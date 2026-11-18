"""Fine-tune a pretrained Transformer for token classification on CoNLL-2003.

Pipeline implemented here (no ``pipeline("ner")`` shortcut):

    words -> tokenizer(is_split_into_words=True) -> input_ids + word_ids
          -> align word-level BIO tags to subword labels
          -> BERT + Linear(768, 9) -> logits per token
          -> argmax -> first-subword collapse -> word-level BIO -> entities

The alignment strategy is selectable:

    --strategy first      Strategy A: first subword keeps the word tag, the other
                          subwords are IGNORE (-100) and contribute no loss.
    --strategy propagate  Strategy B: continuations receive the matching I-X tag.

Usage
-----
    python ner_finetuning.py                       # 3 epochs, Strategy A
    python ner_finetuning.py --strategy propagate --tag propagate --epochs 3
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

from ner_utils import (
    ENTITY_TYPES,
    IGNORE,
    LABEL2ID,
    LABELS,
    MODEL_DIR,
    PRED_DIR,
    ROOT,
    align_tags,
    bio_to_entities,
    entity_prf,
    entity_prf_by_type,
    load_conll,
    predictions_to_word_tags,
    save_predictions,
    tags_to_names,
    token_accuracy_and_confusion,
    token_prf,
    write_json,
)


def encode(tok, sentences, max_len: int, strategy: str):
    """Tokenize + align. Returns token ids, aligned labels and word_ids meta."""
    encoded = []
    truncated = 0
    split_words = 0
    n_subwords = 0
    for sentence in sentences:
        enc = tok(
            sentence["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=max_len,
            padding=False,
        )
        word_ids = enc.word_ids()
        labels = align_tags(word_ids, sentence["ner_tags"], strategy)
        kept_words = len({w for w in word_ids if w is not None})
        if kept_words < len(sentence["tokens"]):
            truncated += 1
        split_words += sum(
            1 for word in set(w for w in word_ids if w is not None)
            if word_ids.count(word) > 1
        )
        n_subwords += len(enc["input_ids"]) - 2  # exclude [CLS]/[SEP]
        encoded.append(
            {
                "id": sentence["id"],
                "tokens": sentence["tokens"],
                "input_ids": enc["input_ids"],
                "attention": enc["attention_mask"],
                "word_ids": word_ids,
                "labels": labels,
                "n_words": len(sentence["tokens"]),
            }
        )
    stats = {
        "n_sentences": len(sentences),
        "truncated_sentences": truncated,
        "split_words": split_words,
        "n_subwords": n_subwords,
        "subwords_per_word": n_subwords / max(1, sum(len(s["tokens"]) for s in sentences)),
    }
    return encoded, stats


def make_tensors(tok, encoded, device):
    input_ids = torch.tensor([e["input_ids"] for e in encoded], dtype=torch.long,
                             device=device)
    attention = torch.tensor([e["attention"] for e in encoded], dtype=torch.long,
                             device=device)
    labels = torch.tensor([e["labels"] for e in encoded], dtype=torch.long, device=device)
    return input_ids, attention, labels


def evaluate(model, loader, device, use_amp: bool):
    model.eval()
    total_loss = 0.0
    batches = 0
    all_word_tags = []
    all_gold_tags = []
    with torch.no_grad():
        for input_ids, attention, labels, metas in loader:
            input_ids = input_ids.to(device)
            attention = attention.to(device)
            labels = labels.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention, labels=labels)
                loss = outputs.loss
            total_loss += float(loss.detach())
            batches += 1
            preds = outputs.logits.argmax(dim=-1).cpu().tolist()
            for pred_ids, meta in zip(preds, metas):
                word_tags = predictions_to_word_tags(meta["word_ids"], pred_ids, meta["n_words"])
                all_word_tags.append(word_tags)
                all_gold_tags.append(tags_to_names(meta["gold_ids"]))
    model.train()
    gold_entities = [bio_to_entities(tags) for tags in all_gold_tags]
    pred_entities = [bio_to_entities(tags) for tags in all_word_tags]
    accuracy, _ = token_accuracy_and_confusion(all_gold_tags, all_word_tags)
    return {
        "loss": total_loss / max(1, batches),
        "token_accuracy": accuracy,
        "token_prf": token_prf(all_gold_tags, all_word_tags),
        "entity": entity_prf(gold_entities, pred_entities),
        "by_type": entity_prf_by_type(gold_entities, pred_entities),
        "gold_tags": all_gold_tags,
        "pred_tags": all_word_tags,
        "gold_entities": gold_entities,
        "pred_entities": pred_entities,
    }


def build_loader(tok, encoded, batch_size: int, device, shuffle: bool, seed: int):
    # Meta is passed through a wrapper dataset; default collate only sees the
    # small token/label lists, so we use a custom batch sampler-free approach:
    # the dataset yields (input_ids, labels, meta) and collate pads input ids.
    dataset = EncodedDataset(encoded)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        collate_fn=lambda batch: encode_collate(batch, tok.pad_token_id),
    )


class EncodedDataset(Dataset):
    def __init__(self, encoded):
        self.encoded = encoded

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, index):
        return self.encoded[index]


def encode_collate(batch, pad_token_id: int):
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.full((len(batch), max_len), IGNORE, dtype=torch.long)
    metas = []
    for row, item in enumerate(batch):
        length = len(item["input_ids"])
        input_ids[row, :length] = torch.tensor(item["input_ids"], dtype=torch.long)
        attention[row, :length] = torch.tensor(item["attention"], dtype=torch.long)
        labels[row, :length] = torch.tensor(item["labels"], dtype=torch.long)
        metas.append(
            {
                "id": item["id"],
                "tokens": item["tokens"],
                "word_ids": item["word_ids"],
                "n_words": item["n_words"],
                "gold_ids": item["gold_ids"],
            }
        )
    return input_ids, attention, labels, metas


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(args) -> dict:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    if use_amp:
        torch.backends.cuda.matmul.allow_tf32 = True

    splits = load_conll()
    train_sentences = splits["train"][: args.limit_train] if args.limit_train else splits["train"]
    dev_sentences = splits["validation"][: args.limit_dev] if args.limit_dev else splits["validation"]

    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    train_enc, train_stats = encode(tok, train_sentences, args.max_len, args.strategy)
    dev_enc, dev_stats = encode(tok, dev_sentences, args.max_len, args.strategy)
    for item, sentence in zip(train_enc, train_sentences):
        item["gold_ids"] = sentence["ner_tags"]
    for item, sentence in zip(dev_enc, dev_sentences):
        item["gold_ids"] = sentence["ner_tags"]

    model = AutoModelForTokenClassification.from_pretrained(
        str(MODEL_DIR),
        num_labels=len(LABELS),
        id2label={i: label for i, label in enumerate(LABELS)},
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    ).to(device)
    model.train()

    train_loader = build_loader(tok, train_enc, args.batch_size, device, True, args.seed)
    dev_loader = build_loader(tok, dev_enc, args.batch_size * 2, device, False, args.seed)

    no_decay = ("bias", "LayerNorm.weight")
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()
                       if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    # bfloat16 autocast is stable on Ampere+ GPUs and needs no gradient scaler.

    history = []
    best_f1 = -1.0
    best_epoch = 0
    best_path = ROOT / "model" / f"best_ner_{args.tag}.pt"
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.perf_counter()
        for step, (input_ids, attention, labels, _metas) in enumerate(train_loader, 1):
            input_ids = input_ids.to(device, non_blocking=True)
            attention = attention.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention, labels=labels)
                loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            epoch_loss += float(loss.detach())
            if step % args.log_every == 0:
                print(f"epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss={epoch_loss / step:.4f} "
                      f"lr={scheduler.get_last_lr()[0]:.2e}", flush=True)
        train_seconds = time.perf_counter() - epoch_start

        dev = evaluate(model, dev_loader, device, use_amp)
        record = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(1, len(train_loader)),
            "dev_loss": dev["loss"],
            "dev_token_accuracy": dev["token_accuracy"],
            "dev_token_f1": dev["token_prf"]["f1"],
            "dev_entity_precision": dev["entity"]["precision"],
            "dev_entity_recall": dev["entity"]["recall"],
            "dev_entity_f1": dev["entity"]["f1"],
            "train_seconds": round(train_seconds, 1),
        }
        history.append(record)
        print(f"epoch {epoch}: train_loss={record['train_loss']:.4f} "
              f"dev_loss={record['dev_loss']:.4f} "
              f"dev_token_f1={record['dev_token_f1']:.4f} "
              f"dev_entity_f1={record['dev_entity_f1']:.4f} "
              f"({record['train_seconds']:.1f}s)", flush=True)

        if dev["entity"]["f1"] > best_f1:
            best_f1 = dev["entity"]["f1"]
            best_epoch = epoch
            if args.save_checkpoint:
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "config": model.config.to_dict(),
                        "epoch": epoch,
                        "dev_entity_f1": best_f1,
                        "strategy": args.strategy,
                        "max_len": args.max_len,
                    },
                    best_path,
                )
                print(f"saved {best_path}", flush=True)

    total_seconds = time.perf_counter() - started
    print(f"training finished in {total_seconds / 60:.1f} min; best dev F1 {best_f1:.4f} "
          f"at epoch {best_epoch}", flush=True)

    if args.save_checkpoint:
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])

    dev_final = evaluate(model, dev_loader, device, use_amp)
    test_sentences = splits["test"][: args.limit_test] if args.limit_test else splits["test"]
    test_enc, test_stats = encode(tok, test_sentences, args.max_len, args.strategy)
    for item, sentence in zip(test_enc, test_sentences):
        item["gold_ids"] = sentence["ner_tags"]
    test_loader = build_loader(tok, test_enc, args.batch_size * 2, device, False, args.seed)
    test = evaluate(model, test_loader, device, use_amp)

    accuracy, confusion = token_accuracy_and_confusion(test["gold_tags"], test["pred_tags"])
    test_block = {
        "entity": test["entity"],
        "by_type": test["by_type"],
        "token_accuracy": accuracy,
        "token_prf": test["token_prf"],
        "n_sentences": len(test_sentences),
        "n_tokens": int(sum(len(t) for t in test["gold_tags"])),
    }
    dev_block = {
        "entity": dev_final["entity"],
        "by_type": dev_final["by_type"],
        "token_accuracy": dev_final["token_accuracy"],
        "token_prf": dev_final["token_prf"],
        "n_sentences": len(dev_sentences),
        "n_tokens": int(sum(len(t) for t in dev_final["gold_tags"])),
    }

    metrics = {
        "model": "google-bert/bert-base-cased",
        "tag": args.tag,
        "strategy": args.strategy,
        "max_len": args.max_len,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "device": str(device),
        "best_epoch": best_epoch,
        "best_dev_entity_f1": best_f1,
        "train_seconds": round(total_seconds, 1),
        "train_stats": train_stats,
        "dev_stats": dev_stats,
        "test_stats": test_stats,
        "history": history,
        "splits": {"validation": dev_block, "test": test_block},
    }

    write_json(PRED_DIR / f"transformer_{args.tag}_metrics.json", metrics)
    write_json(PRED_DIR / f"transformer_{args.tag}_history.json", history)
    np.save(PRED_DIR / f"transformer_{args.tag}_confusion.npy", confusion)

    if args.tag == "main":
        save_predictions(
            PRED_DIR / "test_predictions.json",
            test_sentences, test["gold_tags"], test["pred_tags"],
            test["gold_entities"], test["pred_entities"],
        )
        save_predictions(
            PRED_DIR / "validation_predictions.json",
            dev_sentences, dev_final["gold_tags"], dev_final["pred_tags"],
            dev_final["gold_entities"], dev_final["pred_entities"],
        )
    save_predictions(
        PRED_DIR / f"transformer_{args.tag}_test_predictions.json",
        test_sentences, test["gold_tags"], test["pred_tags"],
        test["gold_entities"], test["pred_entities"],
    )

    print(f"test (best checkpoint): entity P={test['entity']['precision']:.4f} "
          f"R={test['entity']['recall']:.4f} F1={test['entity']['f1']:.4f} | "
          f"token acc={accuracy:.4f} token F1={test['token_prf']['f1']:.4f}", flush=True)
    for etype in ENTITY_TYPES:
        row = test["by_type"][etype]
        print(f"  {etype}: P={row['precision']:.4f} R={row['recall']:.4f} "
              f"F1={row['f1']:.4f} support={row['support']}", flush=True)

    write_json(
        PRED_DIR / "training_environment.json",
        {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "python": os.sys.version,
            "max_len": args.max_len,
            "strategy": args.strategy,
        },
    )
    return metrics


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="main", help="suffix for output files")
    parser.add_argument("--strategy", default="first", choices=["first", "propagate"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-dev", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--no-checkpoint", dest="save_checkpoint", action="store_false")
    return parser.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
