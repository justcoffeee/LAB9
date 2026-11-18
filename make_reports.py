"""Build every table, figure and analysis file required by LAB09.

Inputs are the real prediction files written by ``baseline_ner.py`` and
``ner_finetuning.py``; nothing here invents numbers.

    python make_reports.py
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ner_utils import (
    ENTITY_TYPES,
    LABELS,
    MODEL_DIR,
    PRED_DIR,
    ROOT,
    Entity,
    bio_to_entities,
    entity_prf_by_length,
    load_conll,
    read_json,
    tags_to_names,
    write_json,
)


def load_metrics():
    baseline = read_json(PRED_DIR / "baseline_metrics.json")
    main = read_json(PRED_DIR / "transformer_main_metrics.json")
    propagate_path = PRED_DIR / "transformer_propagate_metrics.json"
    propagate = read_json(propagate_path) if propagate_path.exists() else None
    return baseline, main, propagate


def write_results_csv(baseline, main, propagate):
    rows = []
    for name, payload in (
        ("Baseline (token features + logistic regression)", baseline),
        ("BERT-base-cased fine-tuned (Strategy A: first subword)", main),
    ):
        test = payload["splits"]["test"]
        rows.append(
            {
                "model": name,
                "split": "test",
                "precision": round(test["entity"]["precision"], 4),
                "recall": round(test["entity"]["recall"], 4),
                "f1": round(test["entity"]["f1"], 4),
                "token_accuracy": round(test["token_accuracy"], 4),
                "token_f1": round(test["token_prf"]["f1"], 4),
                "support": test["entity"]["support"],
                "predicted": test["entity"]["predicted"],
                "n_sentences": test["n_sentences"],
                "n_tokens": test["n_tokens"],
            }
        )
    if propagate:
        test = propagate["splits"]["test"]
        rows.append(
            {
                "model": "BERT-base-cased fine-tuned (Strategy B: propagate I-X)",
                "split": "test",
                "precision": round(test["entity"]["precision"], 4),
                "recall": round(test["entity"]["recall"], 4),
                "f1": round(test["entity"]["f1"], 4),
                "token_accuracy": round(test["token_accuracy"], 4),
                "token_f1": round(test["token_prf"]["f1"], 4),
                "support": test["entity"]["support"],
                "predicted": test["entity"]["predicted"],
                "n_sentences": test["n_sentences"],
                "n_tokens": test["n_tokens"],
            }
        )
    path = ROOT / "results.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")
    return rows


def write_entity_metrics(main):
    rows = []
    overall = main["splits"]["test"]["entity"]
    rows.append({"group": "type", "entity": "OVERALL", **overall})
    for etype in ENTITY_TYPES:
        rows.append({"group": "type", "entity": etype, **main["splits"]["test"]["by_type"][etype]})
    gold_path = PRED_DIR / "test_predictions.json"
    preds = read_json(gold_path)
    gold_entities = [
        [Entity(e["start"], e["end"], e["type"]) for e in row["gold_entities"]]
        for row in preds
    ]
    pred_entities = [
        [Entity(e["start"], e["end"], e["type"]) for e in row["pred_entities"]]
        for row in preds
    ]
    for bucket, stats in entity_prf_by_length(gold_entities, pred_entities).items():
        rows.append({"group": "length", "entity": bucket, **stats})
    path = ROOT / "entity_metrics.csv"
    fieldnames = ["group", "entity", "support", "predicted", "tp", "fp", "fn",
                  "precision", "recall", "f1"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                             for k, v in row.items()})
    print(f"wrote {path}")
    return rows


def plot_confusion(main):
    matrix = np.load(PRED_DIR / "transformer_main_confusion.npy")
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float),
                           where=row_sums > 0)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=45, ha="right")
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Predicted tag")
    ax.set_ylabel("Gold tag")
    ax.set_title("Word-level token confusion matrix (BERT, test)")
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            if matrix[i, j]:
                ax.text(j, i, f"{int(matrix[i, j])}", ha="center", va="center",
                        fontsize=7,
                        color="white" if normalized[i, j] > 0.5 else "black")
    fig.colorbar(image, ax=ax, label="Row-normalised rate")
    fig.tight_layout()
    fig.savefig(ROOT / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    write_json(
        PRED_DIR / "confusion_matrix.json",
        {"labels": LABELS, "counts": matrix.tolist(),
         "row_normalised": normalized.round(5).tolist()},
    )
    print("wrote confusion_matrix.png")


def plot_entity_length(metrics_rows, main):
    buckets = ["1", "2", "3", "4+"]
    stats = {row["entity"]: row for row in metrics_rows if row["group"] == "length"}
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    f1s = [stats[b]["f1"] for b in buckets]
    supports = [stats[b]["support"] for b in buckets]
    bars = ax.bar(buckets, f1s, color="#2b6cb0")
    for bar, f1, support in zip(bars, f1s, supports):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"F1={f1:.3f}\nn={support}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Gold entity length (tokens)")
    ax.set_ylabel("Entity-level F1")
    ax.set_title("Entity-level F1 by entity length (BERT, test)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "entity_length_analysis.png", dpi=160)
    plt.close(fig)
    write_json(PRED_DIR / "entity_length.json", stats)
    print("wrote entity_length_analysis.png")


def plot_learning_curves(main):
    history = main["history"]
    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(epochs, [h["train_loss"] for h in history], marker="o", label="train loss")
    ax1.plot(epochs, [h["dev_loss"] for h in history], marker="o", label="dev loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-entropy loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.plot(epochs, [h["dev_entity_f1"] for h in history], marker="o", label="dev entity F1")
    ax2.plot(epochs, [h["dev_token_f1"] for h in history], marker="o", label="dev token F1")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1")
    ax2.set_ylim(0, 1)
    ax2.set_title("Validation F1 per epoch")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "learning_curves.png", dpi=160)
    plt.close(fig)
    print("wrote learning_curves.png")


def tokenization_analysis():
    """Experiment 1: how often does WordPiece split words in CoNLL test?"""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    splits = load_conll()
    rows = []
    split_examples: Counter = Counter()
    pieces = Counter()
    entity_words_total = entity_words_split = 0
    other_words_total = other_words_split = 0

    for sentence in splits["test"]:
        # Count the entire sentence; truncation would hide splits and long inputs.
        enc = tok(sentence["tokens"], is_split_into_words=True, truncation=False,
                  padding=False)
        word_ids = enc.word_ids()
        n_subwords = len(enc["input_ids"]) - 2
        split_word_ids = []
        for word_id in sorted({w for w in word_ids if w is not None}):
            count = word_ids.count(word_id)
            if count > 1:
                split_word_ids.append(word_id)
                split_examples[sentence["tokens"][word_id]] += 1
                pieces.update(tok.tokenize(sentence["tokens"][word_id])[1:])
        entity_tokens = set()
        for entity in bio_to_entities(tags_to_names(sentence["ner_tags"])):
            entity_tokens.update(range(entity.start, entity.end))

        n_entity_words = len(entity_tokens)
        n_split_entity = sum(1 for word_id in split_word_ids if word_id in entity_tokens)
        entity_words_total += n_entity_words
        entity_words_split += n_split_entity
        other_words_total += len(sentence["tokens"]) - n_entity_words
        other_words_split += sum(1 for word_id in split_word_ids
                                 if word_id not in entity_tokens)
        rows.append(
            {
                "id": sentence["id"],
                "n_words": len(sentence["tokens"]),
                "n_subwords": n_subwords,
                "n_split_words": len(split_word_ids),
                "ratio": n_subwords / max(1, len(sentence["tokens"])),
                "has_entity": int(bool(entity_tokens)),
            }
        )

    short = [r for r in rows if r["n_words"] <= 15]
    long_ = [r for r in rows if r["n_words"] > 15]
    summary = {
        "n_sentences": len(rows),
        "n_words": sum(r["n_words"] for r in rows),
        "n_subwords": sum(r["n_subwords"] for r in rows),
        "n_split_words": sum(r["n_split_words"] for r in rows),
        "ratio_subwords_per_word": sum(r["n_subwords"] for r in rows) / sum(r["n_words"] for r in rows),
        "split_rate": sum(r["n_split_words"] for r in rows) / sum(r["n_words"] for r in rows),
        "short_sentences": {
            "n": len(short),
            "ratio": sum(r["n_subwords"] for r in short) / sum(r["n_words"] for r in short),
        },
        "long_sentences": {
            "n": len(long_),
            "ratio": sum(r["n_subwords"] for r in long_) / sum(r["n_words"] for r in long_),
        },
        "entity_words_split_rate": entity_words_split / max(1, entity_words_total),
        "non_entity_words_split_rate": other_words_split / max(1, other_words_total),
        "entity_words": entity_words_total,
        "non_entity_words": other_words_total,
        "max_tokens": max(r["n_subwords"] for r in rows),
        "sentences_over_128_subwords": sum(1 for r in rows if r["n_subwords"] + 2 > 128),
        "most_split_words": split_examples.most_common(15),
        "most_common_continuations": pieces.most_common(15),
    }

    with (ROOT / "tokenization_stats.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Figure used in the README / notebook.
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.hist([r["ratio"] for r in rows], bins=30, color="#4a5568")
    ax.axvline(summary["ratio_subwords_per_word"], color="#e53e3e", linestyle="--",
               label=f"mean = {summary['ratio_subwords_per_word']:.3f}")
    ax.set_xlabel("Subword tokens / words per sentence")
    ax.set_ylabel("Number of sentences")
    ax.set_title("WordPiece expansion on CoNLL-2003 test")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "tokenization_ratio.png", dpi=160)
    plt.close(fig)

    write_json(PRED_DIR / "tokenization_analysis.json", summary)
    print(f"tokenization: {summary['ratio_subwords_per_word']:.4f} subwords/word, "
          f"{summary['split_rate']:.4f} of words are split")
    return summary


def _entity_text(row, entity):
    return " ".join(row["tokens"][entity["start"]:entity["end"]])


def _overlap(a, b) -> int:
    return max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))


def _gold_type_of_overlap(entity, gold):
    best, best_overlap = None, 0
    for candidate in gold:
        overlap = _overlap(entity, candidate)
        if overlap > best_overlap:
            best, best_overlap = candidate, overlap
    return best


def error_analysis(baseline, main, tokenization):
    rows = read_json(PRED_DIR / "test_predictions.json")
    splits = load_conll()

    # Frequency of entity strings in the training split (data-driven "rare" flag).
    train_entity_counts = Counter()
    train_surface_counts = Counter()
    for sentence in splits["train"]:
        for entity in bio_to_entities(tags_to_names(sentence["ner_tags"])):
            text = " ".join(sentence["tokens"][entity.start:entity.end])
            train_entity_counts[text.lower()] += 1
            for token in sentence["tokens"][entity.start:entity.end]:
                train_surface_counts[token.lower()] += 1
    train_token_counts = Counter(
        token.lower() for sentence in splits["train"] for token in sentence["tokens"]
    )

    false_positives, false_negatives, boundary_errors = [], [], []
    seen_types = defaultdict(set)

    for row in rows:
        gold = row["gold_entities"]
        pred = row["pred_entities"]
        gold_keys = {(e["start"], e["end"], e["type"]) for e in gold}
        pred_keys = {(e["start"], e["end"], e["type"]) for e in pred}

        # Exact-span mismatches with a different label are type errors, not
        # spurious/missing; report them in both FP and FN tables (as the lab asks
        # for 5 FP and 5 FN) with an explicit "wrong entity type" label.
        for entity in pred:
            key = (entity["start"], entity["end"], entity["type"])
            if key in gold_keys:
                continue
            overlapping = _gold_type_of_overlap(entity, gold)
            text = _entity_text(row, entity)
            if overlapping is not None and overlapping["type"] != entity["type"]:
                error_type = "wrong entity type"
                explanation = (
                    f"The span matches a gold {overlapping['type']} entity "
                    f"('{_entity_text(row, overlapping)}') but the model labelled it "
                    f"{entity['type']}."
                )
            elif overlapping is not None:
                delta_start = entity["start"] - overlapping["start"]
                delta_end = entity["end"] - overlapping["end"]
                error_type = "wrong boundary"
                explanation = (
                    f"Gold '{_entity_text(row, overlapping)}' ({overlapping['type']}, "
                    f"{overlapping['length']} tokens); prediction '{text}' differs by "
                    f"{delta_start:+d} token(s) at the start and {delta_end:+d} at the end."
                )
            else:
                error_type = "spurious entity"
                explanation = (
                    f"No gold entity overlaps this span; the model turned "
                    f"'{text}' into a {entity['type']} entity."
                )
                if train_token_counts[text.lower()] == 0:
                    explanation += " The exact surface form never appears in training."
                elif train_surface_counts[text.lower()] > 0:
                    explanation += (
                        f" The surface form appears {train_surface_counts[text.lower()]}x "
                        f"in training (mostly outside an entity)."
                    )
            false_positives.append(
                {
                    "text": row["text"],
                    "gold": _entity_text(row, overlapping) if overlapping else "O",
                    "prediction": f"{text} ({entity['type']})",
                    "error_type": error_type,
                    "explanation": explanation,
                    "train_entity_frequency": train_entity_counts[text.lower()],
                }
            )

        for entity in gold:
            key = (entity["start"], entity["end"], entity["type"])
            if key in pred_keys:
                continue
            overlapping = None
            best_overlap = 0
            for candidate in pred:
                overlap = _overlap(entity, candidate)
                if overlap > best_overlap:
                    overlapping, best_overlap = candidate, overlap
            text = _entity_text(row, entity)
            if overlapping is not None and overlapping["type"] != entity["type"]:
                error_type = "wrong entity type"
                explanation = (
                    f"Gold {entity['type']} '{text}' overlaps a predicted "
                    f"{overlapping['type']} span '{_entity_text(row, overlapping)}'."
                )
            elif overlapping is not None:
                error_type = "wrong boundary"
                explanation = (
                    f"Gold span [{entity['start']},{entity['end']}) vs predicted "
                    f"[{overlapping['start']},{overlapping['end']}); "
                    f"overlap {best_overlap}/{entity['length']} gold tokens."
                )
            else:
                error_type = "missing entity"
                explanation = (
                    f"No predicted entity overlaps this gold {entity['type']} span."
                )
                if train_entity_counts[text.lower()] == 0:
                    explanation += " This exact entity never appears in the training split."
                elif train_entity_counts[text.lower()] <= 2:
                    explanation += (
                        f" The entity appears only {train_entity_counts[text.lower()]}x "
                        f"in training."
                    )
            false_negatives.append(
                {
                    "text": row["text"],
                    "gold": f"{text} ({entity['type']})",
                    "prediction": (
                        f"{_entity_text(row, overlapping)} ({overlapping['type']})"
                        if overlapping else "O"
                    ),
                    "error_type": error_type,
                    "explanation": explanation,
                    "train_entity_frequency": train_entity_counts[text.lower()],
                }
            )

        # Boundary errors: any gold/pred pair that overlaps but is not identical.
        for gold_entity in gold:
            for pred_entity in pred:
                if _overlap(gold_entity, pred_entity) == 0:
                    continue
                same_span = (
                    gold_entity["start"] == pred_entity["start"]
                    and gold_entity["end"] == pred_entity["end"]
                )
                if same_span:
                    continue
                boundary_errors.append(
                    {
                        "text": row["text"],
                        "gold": f"{_entity_text(row, gold_entity)} ({gold_entity['type']})",
                        "prediction": f"{_entity_text(row, pred_entity)} ({pred_entity['type']})",
                        "error_type": (
                            "wrong boundary + type"
                            if gold_entity["type"] != pred_entity["type"]
                            else "wrong boundary"
                        ),
                        "explanation": (
                            f"Gold span [{gold_entity['start']},{gold_entity['end']}) vs "
                            f"predicted [{pred_entity['start']},{pred_entity['end']}); "
                            f"overlap {_overlap(gold_entity, pred_entity)}/"
                            f"{max(gold_entity['length'], pred_entity['length'])} tokens, "
                            f"types {gold_entity['type']} -> {pred_entity['type']}."
                        ),
                    }
                )

    def pick(examples, n=5):
        chosen, used_types = [], set()
        for example in examples:
            etype = example["gold"].split("(")[-1].rstrip(")") if "(" in example["gold"] else "O"
            if len(chosen) < n and (etype not in used_types or len(chosen) >= 4):
                chosen.append(example)
                used_types.add(etype)
        for example in examples:
            if len(chosen) >= n:
                break
            if example not in chosen:
                chosen.append(example)
        return chosen[:n]

    fp_examples = pick(false_positives)
    fn_examples = pick(false_negatives)
    boundary_examples = pick(boundary_errors)

    # Context analysis: same surface form, different gold type in the test set.
    surface_types = defaultdict(set)
    surface_rows = defaultdict(list)
    for row in rows:
        for entity in row["gold_entities"]:
            text = _entity_text(row, entity)
            key = text
            surface_types[key].add(entity["type"])
            surface_rows[key].append((row, entity))
    context_cases = []
    for surface, types in surface_types.items():
        if len(types) < 2:
            continue
        examples = []
        for row, entity in surface_rows[surface]:
            pred_entity = next(
                (p for p in row["pred_entities"]
                 if p["start"] == entity["start"] and p["end"] == entity["end"]),
                None,
            )
            examples.append(
                {
                    "sentence": row["text"],
                    "gold_type": entity["type"],
                    "predicted": f"{pred_entity['type']}" if pred_entity else "O",
                    "gold_tags": row["gold_tags"][entity["start"]:entity["end"]],
                }
            )
        context_cases.append({"surface": surface, "types": sorted(types),
                              "examples": examples})
    context_cases.sort(key=lambda c: (-len(c["types"]), c["surface"]))
    context_cases = context_cases[:3]

    summary = {
        "n_sentences": len(rows),
        "false_positives": len(false_positives),
        "false_negatives": len(false_negatives),
        "boundary_pairs": len(boundary_errors),
        "fp_error_types": dict(Counter(e["error_type"] for e in false_positives)),
        "fn_error_types": dict(Counter(e["error_type"] for e in false_negatives)),
        "rare_entity_fn": sum(1 for e in false_negatives if e["train_entity_frequency"] == 0),
        "seen_entity_fn": sum(1 for e in false_negatives if e["train_entity_frequency"] > 0),
    }

    write_json(PRED_DIR / "error_examples.json",
               {"false_positives": fp_examples, "false_negatives": fn_examples,
                "boundary_errors": boundary_examples,
                "summary": summary, "context_cases": context_cases})

    md = ["# Error analysis — BERT NER on CoNLL-2003 test", ""]
    md += [
        f"Test sentences: **{summary['n_sentences']:,}**. "
        f"Entity-level disagreements: **{summary['false_positives']:,} false positives**, "
        f"**{summary['false_negatives']:,} false negatives**, "
        f"**{summary['boundary_pairs']:,} overlapping-but-not-identical span pairs**.",
        "",
        "Error categories are assigned mechanically from span overlap: an unmatched "
        "prediction that overlaps a gold entity of another type is `wrong entity type`; "
        "an unmatched span that overlaps a gold entity of the same type is a boundary "
        "error; an unmatched span with no overlap is `spurious` / `missing`. Exact "
        "matches are never listed.",
        "",
        "## False positives (5)",
        "",
        "| # | Text | Gold | Prediction | Error type | Explanation |",
        "|---|---|---|---|---|---|",
    ]
    for i, example in enumerate(fp_examples, 1):
        md.append(
            f"| {i} | {example['text']} | {example['gold']} | "
            f"{example['prediction']} | {example['error_type']} | "
            f"{example['explanation']} |"
        )
    md += ["", "## False negatives (5)", "",
           "| # | Text | Gold | Prediction | Error type | Explanation |",
           "|---|---|---|---|---|---|"]
    for i, example in enumerate(fn_examples, 1):
        md.append(
            f"| {i} | {example['text']} | {example['gold']} | "
            f"{example['prediction']} | {example['error_type']} | "
            f"{example['explanation']} |"
        )
    md += ["", "## Boundary errors (5)", "",
           "| # | Text | Gold | Prediction | Error type | Explanation |",
           "|---|---|---|---|---|---|"]
    for i, example in enumerate(boundary_examples, 1):
        md.append(
            f"| {i} | {example['text']} | {example['gold']} | "
            f"{example['prediction']} | {example['error_type']} | "
            f"{example['explanation']} |"
        )
    md += [
        "",
        "## Context analysis — same surface form, different entity type",
        "",
        "The three forms below were found automatically by grouping gold entities by "
        "their exact surface form and keeping the forms with at least two types.",
        "",
    ]
    for case in context_cases:
        md.append(f"### `{case['surface']}` -> {', '.join(case['types'])}")
        md.append("")
        for example in case["examples"]:
            md.append(
                f"- *{example['sentence']}*  \n"
                f"  gold **{example['gold_type']}**, predicted **{example['predicted']}** "
                f"(tags: {' '.join(example['gold_tags'])})."
            )
        md.append("")
    md += [
        "A static embedding lookup gives the same vector to the same surface form, "
        "while BERT conditions token states on the sentence. The baseline here also "
        "uses neighbouring tokens and gold POS features, so it is not context-free. "
        "These examples document annotation ambiguity; aggregate F1 alone does not "
        "prove which contextual mechanism caused a particular prediction.",
        "",
        "## Tokenization / alignment errors",
        "",
        f"WordPiece expands the test set to "
        f"{tokenization['ratio_subwords_per_word']:.4f} tokens per word and splits "
        f"{tokenization['split_rate']:.2%} of all words. With Strategy A only the first "
        f"subword carries a label; the continuation pieces are trained with "
        f"`ignore_index=-100`. Word-level decoding uses the first piece, while all "
        f"unmasked pieces still participate in attention. Gold entity boundaries are "
        f"at word boundaries, not inside a word. `training_environment.json` / "
        f"`transformer_main_metrics.json` record the exact number of split words and "
        f"truncated sentences for this run.",
    ]
    md.insert(2, "AI-assisted technical evidence, not the student's personal error analysis. "
              "W9 section 28 requires the student to add their own interpretation. "
              "The descriptions below compare spans mechanically, not causal explanations.\n")
    (ROOT / "error_analysis.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote error_analysis.md ({summary['false_positives']} FP, "
          f"{summary['false_negatives']} FN, {summary['boundary_pairs']} boundary pairs)")
    return summary, context_cases


def write_alignment_ablation(main, propagate):
    if propagate is None:
        return
    rows = []
    for name, payload in (("A: first subword (IGNORE rest)", main),
                          ("B: propagate I-X", propagate)):
        test = payload["splits"]["test"]
        rows.append(
            {
                "strategy": name,
                "epochs": payload["epochs"],
                "best_epoch": payload["best_epoch"],
                "best_dev_entity_f1": round(payload["best_dev_entity_f1"], 4),
                "test_entity_f1": round(test["entity"]["f1"], 4),
                "test_precision": round(test["entity"]["precision"], 4),
                "test_recall": round(test["entity"]["recall"], 4),
                "subwords_per_word": round(payload["train_stats"]["subwords_per_word"], 4),
                "split_words": payload["train_stats"]["split_words"],
                "truncated_sentences": payload["train_stats"]["truncated_sentences"],
                "train_seconds": payload["train_seconds"],
                "notes": "3 epochs, same hyperparameters",
            }
        )
    path = ROOT / "alignment_ablation.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")

    delta = rows[1]["test_entity_f1"] - rows[0]["test_entity_f1"]
    write_json(
        PRED_DIR / "alignment_ablation.json",
        {"rows": rows, "f1_delta_B_minus_A": round(delta, 4)},
    )


def main() -> None:
    baseline, main_metrics, propagate = load_metrics()
    rows = write_results_csv(baseline, main_metrics, propagate)
    entity_rows = write_entity_metrics(main_metrics)
    plot_confusion(main_metrics)
    plot_entity_length(entity_rows, main_metrics)
    plot_learning_curves(main_metrics)
    tokenization = tokenization_analysis()
    error_summary, context_cases = error_analysis(baseline, main_metrics, tokenization)
    write_alignment_ablation(main_metrics, propagate)

    summary = {
        "baseline_test_f1": rows[0]["f1"],
        "transformer_test_f1": rows[1]["f1"],
        "improvement_f1": round(rows[1]["f1"] - rows[0]["f1"], 4),
        "tokenization": tokenization,
        "error_summary": error_summary,
        "n_context_cases": len(context_cases),
    }
    write_json(PRED_DIR / "report_summary.json", summary)
    print(json.dumps(summary, indent=2)[:2000])


if __name__ == "__main__":
    main()
