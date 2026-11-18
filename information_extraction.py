"""Information Extraction on top of the fine-tuned NER model.

    raw text -> NER -> entity spans -> structured JSON

Usage
-----
    python information_extraction.py --text "Barack Obama visited Paris."
    python information_extraction.py --file article.txt
    python information_extraction.py --evaluate 30

``--evaluate`` rebuilds 30 CoNLL-2003 test sentences as raw text, runs the same
pipeline used for arbitrary input and scores the structured output against the
human gold annotations at entity level.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

from ner_utils import (
    ID2LABEL,
    LABEL2ID,
    LABELS,
    MODEL_DIR,
    PRED_DIR,
    ROOT,
    Entity,
    bio_to_entities,
    entity_prf,
    entity_prf_by_type,
    load_conll,
    tags_to_names,
    write_json,
)

TYPE_TO_KEY = {
    "PER": "PERSON",
    "ORG": "ORGANIZATION",
    "LOC": "LOCATION",
    "MISC": "MISC",
}
KEY_TO_TYPE = {v: k for k, v in TYPE_TO_KEY.items()}

WORD_RE = re.compile(r"\w+(?:[-'’]\w+)*|[^\w\s]", re.UNICODE)


def load_model(checkpoint: Path | str | None = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    config = AutoConfig.from_pretrained(
        str(MODEL_DIR),
        num_labels=len(LABELS),
        id2label={i: label for i, label in enumerate(LABELS)},
        label2id=LABEL2ID,
    )
    model = AutoModelForTokenClassification.from_config(config)
    if checkpoint is None:
        checkpoint = ROOT / "model" / "best_ner_main.pt"
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"{checkpoint} not found. Run `python ner_finetuning.py` first, "
            "or pass --checkpoint."
        )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["state_dict"])
    model.to(device).eval()
    return tokenizer, model, device


def _predict_words(tokenizer, model, device, words: list[str]) -> list[str]:
    enc = tokenizer(words, is_split_into_words=True, truncation=True,
                    max_length=256, padding=False, return_tensors="pt")
    word_ids = enc.word_ids()
    with torch.no_grad():
        logits = model(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
        ).logits
    preds = logits.argmax(-1)[0].cpu().tolist()
    tags = ["O"] * len(words)
    seen = set()
    for word_id, label_id in zip(word_ids, preds):
        if word_id is None or word_id in seen:
            continue
        seen.add(word_id)
        tags[word_id] = ID2LABEL[int(label_id)]
    return tags


def _chunks(text: str):
    """Yield (chunk, absolute_start) pairs without losing character offsets."""
    chunk_start = 0
    for match in re.finditer(r"(?<=[.!?])\s+", text):
        yield text[chunk_start:match.start()], chunk_start
        chunk_start = match.end()
    yield text[chunk_start:], chunk_start


def extract(text: str, tokenizer=None, model=None, device=None) -> dict:
    """Run NER over arbitrary text and return a structured representation."""
    if tokenizer is None:
        tokenizer, model, device = load_model()

    entities = []
    for chunk, offset in _chunks(text):
        matches = list(WORD_RE.finditer(chunk))
        if not matches:
            continue
        words = [m.group(0) for m in matches]
        tags = _predict_words(tokenizer, model, device, words)
        for entity in bio_to_entities(tags):
            start = offset + matches[entity.start].start()
            end = offset + matches[entity.end - 1].end()
            entities.append(
                {
                    "text": text[start:end],
                    "type": TYPE_TO_KEY[entity.type],
                    "start": start,
                    "end": end,
                }
            )

    grouped: dict[str, list[str]] = {key: [] for key in TYPE_TO_KEY.values()}
    for entity in entities:
        if entity["text"] not in grouped[entity["type"]]:
            grouped[entity["type"]].append(entity["text"])
    return {
        "text": text,
        "entities": grouped,
        "mentions": [
            {
                "text": entity["text"],
                "type": entity["type"],
                "start_char": entity["start"],
                "end_char": entity["end"],
            }
            for entity in entities
        ],
    }


def evaluate(n_sentences: int = 30) -> dict:
    tokenizer, model, device = load_model()
    splits = load_conll()
    rows = []
    gold_sents, pred_sents = [], []
    for sentence in splits["test"][:n_sentences]:
        # Whitespace-joined text keeps a 1:1 mapping between the raw string and
        # the gold word tokens, so span text can be compared exactly.
        text = " ".join(sentence["tokens"])
        result = extract(text, tokenizer, model, device)
        word_offsets = list(re.finditer(r"\S+", text))
        gold = [Entity(word_offsets[e.start].start(), word_offsets[e.end-1].end(), e.type)
                for e in bio_to_entities(tags_to_names(sentence["ner_tags"]))]
        pred = [Entity(m["start_char"],m["end_char"],KEY_TO_TYPE[m["type"]])
                for m in result["mentions"]]
        # Score exact character boundaries from the structured raw-text output.
        gold_sents.append(gold)
        pred_sents.append(pred)
        rows.append(
            {
                "id": sentence["id"],
                "text": text,
                "gold": [
                    {"text": text[e.start:e.end], "start_char": e.start, "end_char": e.end,
                     "type": TYPE_TO_KEY[e.type]}
                    for e in gold
                ],
                "predicted": result["mentions"],
            }
        )

    overall = entity_prf(gold_sents, pred_sents)
    by_type = entity_prf_by_type(gold_sents, pred_sents)
    write_json(PRED_DIR / "ie_evaluation.json",
               {"n_sentences": n_sentences, "overall": overall,
                "by_type": by_type, "examples": rows})
    with (ROOT / "ie_evaluation.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["entity", "support", "predicted", "tp", "fp", "fn",
                         "precision", "recall", "f1"])
        writer.writerow(["OVERALL", overall["support"], overall["predicted"],
                         overall["tp"], overall["fp"], overall["fn"],
                         round(overall["precision"], 4), round(overall["recall"], 4),
                         round(overall["f1"], 4)])
        for etype, stats in by_type.items():
            writer.writerow([TYPE_TO_KEY[etype], stats["support"], stats["predicted"],
                             stats["tp"], stats["fp"], stats["fn"],
                             round(stats["precision"], 4), round(stats["recall"], 4),
                             round(stats["f1"], 4)])
    print(f"IE entity-level P={overall['precision']:.4f} R={overall['recall']:.4f} "
          f"F1={overall['f1']:.4f} on {n_sentences} sentences")
    for etype, stats in by_type.items():
        print(f"  {TYPE_TO_KEY[etype]:<14} F1={stats['f1']:.4f} "
              f"support={stats['support']}")
    return overall


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", type=str)
    parser.add_argument("--file", type=Path)
    parser.add_argument("--evaluate", type=int, metavar="N")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.evaluate is not None:
        evaluate(args.evaluate)
        return 0
    if not args.text and not args.file:
        parser.error("provide --text, --file or --evaluate")

    text = args.text if args.text else args.file.read_text(encoding="utf-8")
    tokenizer, model, device = load_model(args.checkpoint)
    result = extract(text, tokenizer, model, device)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
