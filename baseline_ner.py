"""Feature-based NER baseline: token features -> logistic regression.

This is the reference point required by LAB09 section 13. It deliberately does
*not* use a pretrained language model: every token is classified independently
from its surface form, its two neighbours on each side and the gold POS tag.

    token -> [word, shape, affixes, neighbours, POS] -> LogisticRegression -> BIO

Usage
-----
    python baseline_ner.py                 # train + evaluate all splits
    python baseline_ner.py --max-train 2000  # quick smoke test
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from ner_utils import (
    ID2LABEL,
    PRED_DIR,
    baseline_features,
    bio_to_entities,
    entity_prf,
    entity_prf_by_type,
    load_conll,
    save_predictions,
    tags_to_names,
    token_accuracy_and_confusion,
    token_prf,
    write_json,
)


def featurize(sentences, max_sentences: int | None = None):
    features, tags, owner = [], [], []
    subset = sentences[:max_sentences] if max_sentences else sentences
    for s_index, sentence in enumerate(subset):
        for t_index in range(len(sentence["tokens"])):
            features.append(baseline_features(sentence, t_index))
            tags.append(ID2LABEL[sentence["ner_tags"][t_index]])
            owner.append(s_index)
    return features, tags, owner, subset


def decode_sentences(subset, tags, owner):
    """Group independent token predictions back into per-sentence tag lists."""
    decoded = [[] for _ in subset]
    for tag, s_index in zip(tags, owner):
        decoded[s_index].append(tag)
    return decoded


def train(max_train: int | None = None, C: float = 1.0, max_iter: int = 300):
    splits = load_conll()
    features, tags, owner, train_subset = featurize(splits["train"], max_train)
    print(f"training rows: {len(features):,} "
          f"({len(train_subset):,} sentences, {len(set(tags))} observed tags)")

    vectorizer = DictVectorizer(sparse=True)
    started = time.perf_counter()
    X = vectorizer.fit_transform(features)
    print(f"feature matrix: {X.shape[0]:,} x {X.shape[1]:,} "
          f"({X.nnz:,} non-zeros) in {time.perf_counter() - started:.1f}s")

    classifier = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver="lbfgs",
        n_jobs=-1,
        verbose=0,
    )
    started = time.perf_counter()
    classifier.fit(X, tags)
    train_seconds = time.perf_counter() - started
    print(f"logistic regression fitted in {train_seconds:.1f}s "
          f"(n_iter={int(np.max(classifier.n_iter_))})")

    metrics = {
        "model": "baseline_token_features_logreg",
        "C": C,
        "max_iter": max_iter,
        "n_features": int(X.shape[1]),
        "n_train_tokens": len(features),
        "n_train_sentences": len(train_subset),
        "train_seconds": round(train_seconds, 2),
        "iterations": [int(v) for v in classifier.n_iter_],
        "splits": {},
    }
    outputs = {}
    for split in ("validation", "test"):
        sentences = splits[split]
        feats, gold_tags, sent_owner, subset = featurize(sentences)
        predicted = classifier.predict(vectorizer.transform(feats))
        pred_tags = decode_sentences(subset, predicted, sent_owner)
        gold_per_sentence = [tags_to_names(s["ner_tags"]) for s in subset]

        gold_entities = [bio_to_entities(tags) for tags in gold_per_sentence]
        pred_entities = [bio_to_entities(tags) for tags in pred_tags]
        accuracy, confusion = token_accuracy_and_confusion(gold_per_sentence, pred_tags)
        summary = {
            "entity": entity_prf(gold_entities, pred_entities),
            "by_type": entity_prf_by_type(gold_entities, pred_entities),
            "token_accuracy": accuracy,
            "token_prf": token_prf(gold_per_sentence, pred_tags),
            "n_sentences": len(subset),
            "n_tokens": int(sum(len(t) for t in gold_per_sentence)),
        }
        metrics["splits"][split] = summary
        outputs[split] = (subset, gold_per_sentence, pred_tags, gold_entities, pred_entities)
        print(f"{split}: entity P={summary['entity']['precision']:.4f} "
              f"R={summary['entity']['recall']:.4f} F1={summary['entity']['f1']:.4f} | "
              f"token acc={accuracy:.4f}")

        save_predictions(
            PRED_DIR / f"baseline_{split}_predictions.json",
            subset, gold_per_sentence, pred_tags, gold_entities, pred_entities,
        )
        np.save(PRED_DIR / f"baseline_{split}_confusion.npy", confusion)

    write_json(PRED_DIR / "baseline_metrics.json", metrics)
    write_json(PRED_DIR / "baseline_label_order.json", {"labels": [ID2LABEL[i] for i in range(9)]})
    print(f"wrote {PRED_DIR / 'baseline_metrics.json'}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-train", type=int, default=None,
                        help="limit training sentences (smoke tests only)")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=300)
    args = parser.parse_args()
    train(args.max_train, args.C, args.max_iter)


if __name__ == "__main__":
    main()
