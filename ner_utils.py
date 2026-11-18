"""Shared utilities for LAB09: data loading, BIO handling, alignment and metrics.

Everything here is deterministic and dependency-light so that the baseline, the
transformer fine-tuning script, the notebooks and the reporting script all use
exactly the same label space, alignment rule and evaluation code.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = [
    "O",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
    "B-MISC",
    "I-MISC",
]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}
ENTITY_TYPES = ["PER", "ORG", "LOC", "MISC"]
IGNORE = -100

# CoNLL-2003 POS tag inventory (same order as the ClassLabel in the parquet file).
POS_TAGS = [
    '"', "''", "#", "$", "(", ")", ",", ".", ":", "``",
    "CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS",
    "MD", "NN", "NNP", "NNPS", "NNS", "NN|SYM", "PDT", "POS", "PRP", "PRP$",
    "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG",
    "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB",
]

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PRED_DIR = ROOT / "predictions"
MODEL_DIR = ROOT / "model" / "bert-base-cased"


@dataclass(frozen=True)
class Entity:
    """A single entity span. ``end`` is exclusive, as in Python slices."""

    start: int
    end: int
    type: str

    @property
    def length(self) -> int:
        return self.end - self.start

    def to_dict(self, tokens: list[str] | None = None) -> dict:
        out = {"start": self.start, "end": self.end, "type": self.type,
               "length": self.length}
        if tokens is not None:
            out["text"] = " ".join(tokens[self.start:self.end])
        return out


def load_conll(data_dir: Path | str = DATA_DIR) -> dict[str, list[dict]]:
    """Load the three CoNLL-2003 splits from the parquet files in ``data/``.

    The parquet files come from the Hugging Face parquet conversion of
    ``eriktks/conll2003``; see ``download_data.py`` for the exact URLs and
    checksums.
    """
    data_dir = Path(data_dir)
    splits: dict[str, list[dict]] = {}
    for split, filename in (
        ("train", "train.parquet"),
        ("validation", "validation.parquet"),
        ("test", "test.parquet"),
    ):
        df = pd.read_parquet(data_dir / filename)
        sentences = []
        for row in df.itertuples(index=False):
            sentences.append(
                {
                    "id": str(row.id),
                    "tokens": [str(t) for t in row.tokens],
                    "pos_tags": [int(p) for p in row.pos_tags],
                    "chunk_tags": [int(c) for c in row.chunk_tags],
                    "ner_tags": [int(t) for t in row.ner_tags],
                }
            )
        splits[split] = sentences
    return splits


def tags_to_names(tag_ids) -> list[str]:
    return [ID2LABEL[int(t)] for t in tag_ids]


def bio_to_entities(tags, tokens: list[str] | None = None) -> list[Entity]:
    """Decode a BIO tag sequence into entities.

    An ``I-X`` tag that does not continue an open ``X`` entity is treated as the
    start of a new ``X`` entity. This matches ``seqeval`` (the reference
    implementation used for CoNLL-2003) and makes the decoder robust to
    unconstrained baseline predictions.
    """
    entities: list[Entity] = []
    current_type: str | None = None
    start = 0

    def close(end: int) -> None:
        nonlocal current_type
        if current_type is not None:
            entities.append(Entity(start, end, current_type))
            current_type = None

    for i, tag in enumerate(tags):
        tag = str(tag)
        if tag == "O" or tag == "" or tag.startswith("O-"):
            close(i)
        elif tag.startswith("B-") or tag.startswith("I-"):
            etype = tag[2:]
            if tag.startswith("B-") or current_type != etype:
                close(i)
                current_type, start = etype, i
            # same-type I- extends the open entity
        else:  # unknown tag: treat as outside
            close(i)
    close(len(tags))

    if tokens is not None:
        entities = [Entity(e.start, e.end, e.type) for e in entities
                    if 0 <= e.start < e.end <= len(tokens)]
    return entities


def entities_as_tuples(entities: list[Entity]) -> set[tuple[int, int, str]]:
    return {(e.start, e.end, e.type) for e in entities}


def entity_sets(gold: list[Entity], pred: list[Entity]):
    g, p = entities_as_tuples(gold), entities_as_tuples(pred)
    return g & p, p - g, g - p


def entity_prf(gold_sents, pred_sents) -> dict:
    """Micro entity-level precision/recall/F1 over parallel lists of sentences."""
    tp = fp = fn = 0
    for gold, pred in zip(gold_sents, pred_sents):
        t, f, n = entity_sets(gold, pred)
        tp += len(t)
        fp += len(f)
        fn += len(n)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "support": tp + fn,
        "predicted": tp + fp,
    }


def entity_prf_by_type(gold_sents, pred_sents) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for etype in ENTITY_TYPES:
        tp_all = fp_all = fn_all = 0
        for gold, pred in zip(gold_sents, pred_sents):
            g = {e for e in entities_as_tuples(gold) if e[2] == etype}
            p = {e for e in entities_as_tuples(pred) if e[2] == etype}
            tp_all += len(g & p)
            fp_all += len(p - g)
            fn_all += len(g - p)
        precision = tp_all / (tp_all + fp_all) if tp_all + fp_all else 0.0
        recall = tp_all / (tp_all + fn_all) if tp_all + fn_all else 0.0
        out[etype] = {
            "support": tp_all + fn_all,
            "predicted": tp_all + fp_all,
            "tp": tp_all, "fp": fp_all, "fn": fn_all,
            "precision": precision,
            "recall": recall,
            "f1": (2 * precision * recall / (precision + recall)
                   if precision + recall else 0.0),
        }
    return out


def entity_prf_by_length(gold_sents, pred_sents, buckets=((1, 1), (2, 2), (3, 3), (4, 99))):
    """Entity-level P/R/F1 per gold entity length.

    A predicted entity is attributed to the length bucket of the gold entity it
    overlaps most (same sentence); predictions with no overlap are attributed to
    their own length bucket. This keeps false positives visible instead of
    silently dropping them, which would inflate precision on long entities.
    """
    names = ["1", "2", "3", "4+"]
    stats = {name: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for name in names}

    def bucket_of(length: int) -> str:
        for name, (lo, hi) in zip(names, buckets):
            if lo <= length <= hi:
                return name
        return names[-1]

    for gold, pred in zip(gold_sents, pred_sents):
        gold_set = entities_as_tuples(gold)
        pred_set = entities_as_tuples(pred)
        matched_gold = gold_set & pred_set
        for s, e, t in gold_set:
            b = bucket_of(e - s)
            stats[b]["support"] += 1
            if (s, e, t) in matched_gold:
                stats[b]["tp"] += 1
            else:
                stats[b]["fn"] += 1
        for s, e, t in pred_set - gold_set:
            best, best_overlap = None, 0
            for gs, ge, gt in sorted(gold_set):
                overlap = max(0, min(e, ge) - max(s, gs))
                if overlap > best_overlap:
                    best, best_overlap = (gs, ge, gt), overlap
            b = bucket_of(best[1] - best[0]) if best else bucket_of(e - s)
            stats[b]["fp"] += 1

    out = {}
    for name in names:
        s = stats[name]
        precision = s["tp"] / (s["tp"] + s["fp"]) if s["tp"] + s["fp"] else 0.0
        recall = s["tp"] / (s["tp"] + s["fn"]) if s["tp"] + s["fn"] else 0.0
        out[name] = {
            **s,
            "predicted": s["tp"] + s["fp"],
            "precision": precision,
            "recall": recall,
            "f1": (2 * precision * recall / (precision + recall)
                   if precision + recall else 0.0),
        }
    return out


def token_accuracy_and_confusion(gold_tag_lists, pred_tag_lists) -> tuple[float, np.ndarray]:
    """Word-level token accuracy plus a 9x9 confusion matrix (rows gold, cols pred)."""
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    correct = total = 0
    for gold, pred in zip(gold_tag_lists, pred_tag_lists):
        for g, p in zip(gold, pred):
            gi, pi = LABEL2ID[g], LABEL2ID[p]
            matrix[gi, pi] += 1
            correct += int(gi == pi)
            total += 1
    return (correct / total if total else 0.0), matrix


def token_prf(gold_tag_lists, pred_tag_lists, ignore_o: bool = True) -> dict:
    """Token-level P/R/F1 for entity tags (``O`` excluded by default)."""
    tp = fp = fn = 0
    for gold, pred in zip(gold_tag_lists, pred_tag_lists):
        for g, p in zip(gold, pred):
            if ignore_o and g == "O" and p == "O":
                continue
            if g == p:
                tp += 1
            else:
                fp += int(p != "O")
                fn += int(g != "O")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}


def align_tags(word_ids, word_tags, strategy: str = "first") -> list[int]:
    """Map word-level tags onto subword tokens.

    strategy="first": the first subword of a word receives the word tag, the
    remaining subwords receive ``IGNORE`` (Strategy A of the lab sheet).

    strategy="propagate": the first subword receives the word tag and the
    remaining subwords receive the matching ``I-X`` tag (Strategy B). A word
    tagged ``O`` propagates ``O``.

    Special tokens (``word_id is None``) are always ``IGNORE``.
    """
    if strategy not in ("first", "propagate"):
        raise ValueError(f"unknown strategy: {strategy}")
    labels: list[int] = []
    previous_word = None
    for word_id in word_ids:
        if word_id is None:
            labels.append(IGNORE)
            previous_word = None
            continue
        tag = ID2LABEL[int(word_tags[word_id])]
        if word_id != previous_word:
            labels.append(LABEL2ID[tag])
        else:
            if strategy == "first":
                labels.append(IGNORE)
            else:
                labels.append(LABEL2ID["I-" + tag[2:]] if tag != "O" else LABEL2ID["O"])
        previous_word = word_id
    return labels


def predictions_to_word_tags(word_ids, token_label_ids, n_words: int,
                             strategy: str = "first") -> list[str]:
    """Collapse subword predictions back to one tag per original word."""
    tags = ["O"] * n_words
    seen: set[int] = set()
    for word_id, label_id in zip(word_ids, token_label_ids):
        if word_id is None or word_id in seen:
            continue
        seen.add(word_id)
        tags[word_id] = ID2LABEL[int(label_id)]
    if strategy == "propagate":
        # Continuation predictions were trained as I-X; for word-level decoding
        # the first subword is the word label, so nothing else is required.
        pass
    return tags


def word_shape(word: str) -> str:
    """Collapse character classes, e.g. ``iPhone15`` -> ``aXxxxx99``."""
    out = []
    for ch in word:
        if ch.isupper():
            cls = "X"
        elif ch.islower():
            cls = "x"
        elif ch.isdigit():
            cls = "9"
        else:
            cls = ch
        if out and out[-1] == cls:
            continue
        out.append(cls)
    return "".join(out)


def baseline_features(sentence: dict, index: int) -> dict[str, float | str]:
    """Token features for the feature-based baseline.

    Only the token itself, its immediate context and the gold POS tags are used.
    Chunk tags are deliberately excluded because they already encode noun-phrase
    boundaries and would leak part of the entity-boundary signal.
    """
    tokens = sentence["tokens"]
    pos = sentence["pos_tags"]
    word = tokens[index]
    lower = word.lower()
    feats: dict[str, float | str] = {
        "bias": 1.0,
        "w": lower,
        "w_case": word,
        "shape": word_shape(word),
        "is_title": float(word[:1].isupper()),
        "is_upper": float(word.isupper()),
        "is_digit": float(word.isdigit()),
        "has_digit": float(any(c.isdigit() for c in word)),
        "has_hyphen": float("-" in word),
        "has_period": float("." in word),
        "len": min(len(word), 15),
        "pos": POS_TAGS[pos[index]] if index < len(pos) else "<unk>",
    }
    for n in (1, 2, 3):
        if len(word) >= n:
            feats[f"pre{n}"] = lower[:n]
            feats[f"suf{n}"] = lower[-n:]
    for offset, name in ((-2, "w_m2"), (-1, "w_m1"), (1, "w_p1"), (2, "w_p2")):
        j = index + offset
        feats[name] = tokens[j].lower() if 0 <= j < len(tokens) else "<pad>"
        feats[f"{name}_pos"] = POS_TAGS[pos[j]] if 0 <= j < len(pos) else "<pad>"
    return feats


def sentence_text(tokens: list[str]) -> str:
    """Detokenise with simple spacing rules (enough for the IE demo)."""
    text = ""
    for token in tokens:
        if token in {".", ",", ";", ":", "!", "?", "%", ")", "''", "'s"}:
            text += token
        elif token in {"(", "``", "$", "#"}:
            text += " " + token
        elif token.startswith("'"):
            text += token
        else:
            text += (" " if text else "") + token
    return text.strip()


def write_json(path: Path | str, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def read_json(path: Path | str):
    with Path(path).open(encoding="utf-8") as fh:
        return json.load(fh)


def save_predictions(path: Path | str, sentences, gold_tags, pred_tags, gold_entities, pred_entities):
    payload = []
    for sent, gold, pred, gent, pent in zip(sentences, gold_tags, pred_tags, gold_entities, pred_entities):
        payload.append(
            {
                "id": sent["id"],
                "text": sentence_text(sent["tokens"]),
                "tokens": sent["tokens"],
                "gold_tags": gold,
                "pred_tags": pred,
                "gold_entities": [e.to_dict(sent["tokens"]) for e in gent],
                "pred_entities": [e.to_dict(sent["tokens"]) for e in pent],
            }
        )
    write_json(path, payload)
    return payload


def summarize_counter(counter: Counter) -> dict:
    total = sum(counter.values())
    return {k: {"count": v, "share": v / total if total else 0.0}
            for k, v in counter.most_common()}


def label_distribution(sentences) -> dict[str, dict]:
    counter = Counter()
    entity_counter = Counter()
    for sent in sentences:
        for tag_id in sent["ner_tags"]:
            counter[ID2LABEL[int(tag_id)]] += 1
        for entity in bio_to_entities(tags_to_names(sent["ner_tags"])):
            entity_counter[entity.type] += 1
    return {
        "tokens": summarize_counter(counter),
        "entities": summarize_counter(entity_counter),
        "n_sentences": len(sentences),
        "n_tokens": sum(counter.values()),
        "n_entities": sum(entity_counter.values()),
    }
