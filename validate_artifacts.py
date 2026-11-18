"""Verify that the LAB09 artifacts are internally consistent and complete.

This is a self-check, not a grader: it re-derives every headline number from the
prediction files and compares it with the published CSV/Markdown/figures.

    python validate_artifacts.py
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from ner_utils import (
    Entity,
    ID2LABEL,
    IGNORE,
    MODEL_DIR,
    PRED_DIR,
    ROOT,
    align_tags,
    bio_to_entities,
    entity_prf,
    load_conll,
    read_json,
    write_json,
)

DELIVERABLES = [
    "README.md",
    "calculations.md",
    "prediction.md",
    "baseline_ner.py",
    "tokenizer_alignment.ipynb",
    "ner_finetuning.ipynb",
    "results.csv",
    "entity_metrics.csv",
    "entity_length_analysis.png",
    "confusion_matrix.png",
    "error_analysis.md",
    "information_extraction.py",
    "reflection.md",
]


def check(condition: bool, message: str) -> dict:
    return {"check": message, "ok": bool(condition)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate() -> dict:
    results = []

    for name in DELIVERABLES:
        path = ROOT / name
        results.append(check(path.exists() and path.stat().st_size > 0,
                             f"deliverable exists and is non-empty: {name}"))

    for name in ("confusion_matrix.png", "entity_length_analysis.png"):
        path = ROOT / name
        if path.exists():
            header = path.read_bytes()[:8]
            results.append(check(header == b"\x89PNG\r\n\x1a\n",
                                 f"{name} is a valid PNG"))

    baseline = read_json(PRED_DIR / "baseline_metrics.json")
    main = read_json(PRED_DIR / "transformer_main_metrics.json")
    rows = read_json(PRED_DIR / "test_predictions.json")

    # Golden integration case: one correct token does not make a correct full entity.
    partial=entity_prf([bio_to_entities(['B-LOC','I-LOC'])],
                       [bio_to_entities(['B-LOC','O'])])
    results.append(check(partial['tp']==0 and partial['fp']==1 and partial['fn']==1,
                         'partial New York prediction is an entity FP and FN'))

    # 1. Prediction files <-> metrics JSON.
    for tag, metrics in (("baseline", baseline), ("", main)):
        prefix = f"{tag}_" if tag else ""
        preds = read_json(PRED_DIR / f"{prefix}test_predictions.json")
        gold_entities = [
            [Entity(e["start"], e["end"], e["type"]) for e in row["gold_entities"]]
            for row in preds
        ]
        pred_entities = [
            [Entity(e["start"], e["end"], e["type"]) for e in row["pred_entities"]]
            for row in preds
        ]
        recomputed = entity_prf(gold_entities, pred_entities)
        expected = metrics["splits"]["test"]["entity"]
        results.append(check(recomputed["tp"] == expected["tp"]
                             and recomputed["fp"] == expected["fp"]
                             and recomputed["fn"] == expected["fn"],
                             f"{tag or 'transformer'} prediction spans match metrics JSON"))
        results.append(check(abs(recomputed["f1"] - expected["f1"]) < 1e-9,
                             f"{tag or 'transformer'} entity F1 recomputes exactly"))

    # 2. results.csv <-> metrics JSON.
    with (ROOT / "results.csv").open(encoding="utf-8") as fh:
        csv_rows = list(csv.DictReader(fh))
    published = {row["model"]: float(row["f1"]) for row in csv_rows}
    baseline_f1 = round(baseline["splits"]["test"]["entity"]["f1"], 4)
    transformer_f1 = round(main["splits"]["test"]["entity"]["f1"], 4)
    results.append(check(
        any(abs(value - baseline_f1) < 1e-9 for value in published.values()),
        "results.csv contains the baseline test F1"))
    results.append(check(
        any(abs(value - transformer_f1) < 1e-9 for value in published.values()),
        "results.csv contains the transformer test F1"))

    # 3. entity_metrics.csv <-> metrics JSON.
    with (ROOT / "entity_metrics.csv").open(encoding="utf-8") as fh:
        entity_rows = list(csv.DictReader(fh))
    overall_rows = [r for r in entity_rows if r["entity"] == "OVERALL"]
    results.append(check(len(overall_rows) == 1, "entity_metrics.csv has one OVERALL row"))
    if overall_rows:
        results.append(check(
            abs(float(overall_rows[0]["f1"]) - transformer_f1) < 1e-4,
            "entity_metrics.csv OVERALL F1 matches results.csv"))
    results.append(check(
        {r["entity"] for r in entity_rows if r["group"] == "type"}
        == {"OVERALL", "PER", "ORG", "LOC", "MISC"},
        "entity_metrics.csv has all four entity types"))
    results.append(check(
        {r["entity"] for r in entity_rows if r["group"] == "length"} == {"1", "2", "3", "4+"},
        "entity_metrics.csv has all four length buckets"))

    # 4. Confusion matrix is a real confusion matrix of the test predictions.
    matrix = np.load(PRED_DIR / "transformer_main_confusion.npy")
    published_cm = read_json(PRED_DIR / "confusion_matrix.json")
    results.append(check(matrix.shape == (9, 9), "confusion matrix is 9x9"))
    results.append(check(int(matrix.sum()) == main["splits"]["test"]["n_tokens"],
                         "confusion matrix totals equal test token count"))
    results.append(check(
        abs(matrix.diagonal().sum() / matrix.sum() - main["splits"]["test"]["token_accuracy"]) < 1e-12,
        "confusion matrix diagonal equals published token accuracy"))
    results.append(check(published_cm["counts"] == matrix.tolist(),
                         "confusion_matrix.json matches the .npy matrix"))

    # 5. Alignment: first subword gets the gold tag, continuations IGNORE.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    splits = load_conll()
    for name in ['test_predictions.json','baseline_test_predictions.json',
                 'transformer_propagate_test_predictions.json']:
        saved=read_json(PRED_DIR/name)
        coherent=len(saved)==len(splits['test'])
        for row,sentence in zip(saved,splits['test']):
            coherent &= (row['tokens']==sentence['tokens'] and
                         len(row['pred_tags'])==len(sentence['tokens']) and
                         row['gold_tags']==[ID2LABEL[t] for t in sentence['ner_tags']])
            for key,tags in [('gold_entities',row['gold_tags']),('pred_entities',row['pred_tags'])]:
                actual={(e.start,e.end,e.type) for e in bio_to_entities(tags)}
                recorded={(e['start'],e['end'],e['type']) for e in row[key]}
                coherent &= actual==recorded
        results.append(check(coherent,f'{name}: labels and spans agree with official test data'))
    bad = 0
    checked_tokens = 0
    for sentence in splits["test"][:100]:
        enc = tokenizer(sentence["tokens"], is_split_into_words=True,
                        truncation=True, max_length=128)
        word_ids = enc.word_ids()
        labels = align_tags(word_ids, sentence["ner_tags"], "first")
        seen = set()
        for word_id, label_id in zip(word_ids, labels):
            if word_id is None:
                bad += int(label_id != IGNORE)
                continue
            if word_id in seen:
                bad += int(label_id != IGNORE)
                continue
            seen.add(word_id)
            if label_id != sentence["ner_tags"][word_id]:
                bad += 1
            checked_tokens += 1
    results.append(check(bad == 0 and checked_tokens > 0,
                         f"Strategy A alignment is exact on {checked_tokens} words"))

    # 6. Split hygiene.
    # CoNLL row IDs restart in each split; compare split-scoped IDs, not raw row numbers.
    results.append(check(
        all(len({s['id'] for s in rows}) == len(rows) for rows in splits.values())
        and {key: len(value) for key,value in splits.items()}
            == {'train':14041,'validation':3250,'test':3453},
        "official split sizes and unique row IDs within each split"))
    from download_data import DATA_FILES
    results.append(check(all(sha256(ROOT/'data'/name)==meta[1] for name,meta in DATA_FILES.items()),
                         "official train/validation/test file hashes match pinned assets"))

    # 7. Notebooks are real executed notebooks.
    for name in ("tokenizer_alignment.ipynb", "ner_finetuning.ipynb"):
        path = ROOT / name
        if not path.exists():
            continue
        notebook = read_json(path)
        code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
        executed = [c for c in code_cells if c.get("execution_count") is not None]
        results.append(check(len(executed) == len(code_cells),
                             f"{name}: every code cell has an execution count"))
        results.append(check(
            any(any(o.get("output_type") in ("stream", "execute_result", "display_data")
                    for o in c.get("outputs", [])) for c in code_cells),
            f"{name}: cells contain real outputs"))
        results.append(check(not any(o.get('output_type')=='error' for c in code_cells
                                    for o in c.get('outputs',[])),f"{name}: no execution errors"))

    # 8. Information extraction evaluation consistency.
    ie_path = PRED_DIR / "ie_evaluation.json"
    if ie_path.exists():
        ie = read_json(ie_path)
        results.append(check(ie['n_sentences']>=30,'IE evaluated on at least 30 gold sentences'))
        ie_gold=[];ie_pred=[]
        reverse={'PERSON':'PER','ORGANIZATION':'ORG','LOCATION':'LOC','MISC':'MISC'}
        for example in ie['examples']:
            ie_gold.append([Entity(e['start_char'],e['end_char'],reverse[e['type']])
                            for e in example['gold']])
            ie_pred.append([Entity(e['start_char'],e['end_char'],reverse[e['type']])
                            for e in example['predicted']])
        results.append(check(entity_prf(ie_gold,ie_pred)==ie['overall'],
                             'IE exact character-span metrics independently recomputed from JSON mentions'))
        results.append(check(ie["overall"]["support"] > 0,
                             "IE evaluation has gold entities"))
        with (ROOT / "ie_evaluation.csv").open(encoding="utf-8") as fh:
            ie_rows = list(csv.DictReader(fh))
        overall = next(r for r in ie_rows if r["entity"] == "OVERALL")
        results.append(check(abs(float(overall["f1"]) - ie["overall"]["f1"]) < 1e-4,
                             "ie_evaluation.csv matches ie_evaluation.json"))

    errors=read_json(PRED_DIR/'error_examples.json')
    results.append(check(all(len(errors[k])>=5 for k in ['false_positives','false_negatives','boundary_errors']),
                         'at least 5 FP, 5 FN and 5 boundary error examples'))
    results.append(check(len(errors['context_cases'])>=3 and all(len(c['types'])>=2
                         for c in errors['context_cases']), '3 exact-surface forms have different gold entity types'))
    for name,key in [('checkpoint_verification.json','all_passed'),
                     ('initial_encoder_provenance.json','all_encoder_weights_match_upstream')]:
        proof=read_json(PRED_DIR/name)
        results.append(check(proof[key],f'{name}: provenance/checkpoint verification passed'))
    tokenizer_proof=read_json(PRED_DIR/'tokenizer_provenance.json')
    results.append(check(all(tokenizer_proof[k] for k in
                         ['same_vocab','same_input_ids','same_attention_masks','same_word_ids']),
                         'prepared and original upstream tokenizers agree on the entire test set'))
    manifest=read_json(ROOT/'assets_manifest.json')
    results.append(check(all(sha256(ROOT/name)==expected
                             for name,expected in manifest['source_sha256'].items()),
                         'source byte hashes match assets_manifest.json'))
    results.append(check(all(sha256(ROOT/name)==info['sha256']
                             for name,info in manifest['assets'].items()),
                         'downloaded model/data hashes match assets_manifest.json'))

    summary = {
        "n_checks": len(results),
        "n_passed": sum(r["ok"] for r in results),
        "all_passed": all(r["ok"] for r in results),
        "checks": results,
        "hashes": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in (ROOT / "results.csv", ROOT / "entity_metrics.csv",
                      ROOT / "error_analysis.md", PRED_DIR / "test_predictions.json")
            if p.exists()
        },
    }
    write_json(PRED_DIR / "validation.json", summary)
    return summary


def main() -> int:
    summary = validate()
    for result in summary["checks"]:
        print(("PASS " if result["ok"] else "FAIL ") + result["check"])
    print(f"\n{summary['n_passed']}/{summary['n_checks']} checks passed")
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
