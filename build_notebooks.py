"""Build and execute the two LAB09 notebooks with real outputs.

    python build_notebooks.py

The notebooks are ordinary nbformat documents; this script only assembles the
cells, executes them with nbclient in the repository root and saves the result.
Re-running it overwrites both notebooks with fresh, really-executed outputs.
"""

from __future__ import annotations

from pathlib import Path
import json
import os
import sys

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent


def md(text: str):
    return new_markdown_cell(text.strip("\n"))


def code(text: str):
    return new_code_cell(text.strip("\n"))


def tokenizer_notebook() -> nbformat.NotebookNode:
    cells = [
        md(
            """
# Tokenizer alignment — LAB09

Notebook này trả lời câu hỏi trung tâm của lab: annotation NER được gán theo
**word**, nhưng Transformer đọc **subword token**. Ta phải nhìn thấy `word_ids`,
`input_ids`, `labels` và quyết định cách căn chỉnh.

Mọi số liệu dưới đây được tính lại từ CoNLL-2003 test trong `data/` bằng
`bert-base-cased`.
"""
        ),
        code(
            """
import sys
from collections import Counter

import pandas as pd
from transformers import AutoTokenizer

from ner_utils import (
    IGNORE, ID2LABEL, LABELS, MODEL_DIR, bio_to_entities,
    align_tags, load_conll, read_json, tags_to_names,
)

pd.set_option("display.max_colwidth", 80)
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
splits = load_conll()
print("tokenizer:", type(tokenizer).__name__, "| vocab:", tokenizer.vocab_size)
print({k: len(v) for k, v in splits.items()})
"""
        ),
        md(
            """
## 1. Word -> subword

`is_split_into_words=True` giữ nguyên ranh giới word. `word_ids()` cho biết mỗi
subword thuộc word nào (`None` = special token).
"""
        ),
        code(
            """
words = ["Barack", "Obama", "visited", "New", "York", "."]
encoded = tokenizer(words, is_split_into_words=True, truncation=True, padding=True)
table = pd.DataFrame({
    "position": range(len(encoded["input_ids"])),
    "token": tokenizer.convert_ids_to_tokens(encoded["input_ids"]),
    "input_id": encoded["input_ids"],
    "attention_mask": encoded["attention_mask"],
    "word_id": encoded.word_ids(),
    "word": [words[w] if w is not None else "<special>" for w in encoded.word_ids()],
})
display(table)
"""
        ),
        md(
            """
## 2. Một từ thật sự bị chia trong corpus

Không phải từ nào cũng bị chia. Ví dụ dưới đây lấy từ test set: một entity word
bị WordPiece tách thành nhiều mảnh `##`.
"""
        ),
        code(
            """
example = None
for sentence in splits["test"]:
    enc = tokenizer(sentence["tokens"], is_split_into_words=True,
                    truncation=True, max_length=128)
    wids = enc.word_ids()
    for word_id in sorted({w for w in wids if w is not None}):
        if wids.count(word_id) > 1 and sentence["ner_tags"][word_id] != 0:
            example = (sentence, enc, word_id)
            break
    if example:
        break

sentence, enc, word_id = example
entity_at_word = {e.start: e for e in bio_to_entities(tags_to_names(sentence["ner_tags"]))}
rows = []
for i, (token, wid) in enumerate(zip(tokenizer.convert_ids_to_tokens(enc["input_ids"]),
                                     enc.word_ids())):
    rows.append({
        "token": token,
        "word_id": wid,
        "word": sentence["tokens"][wid] if wid is not None else "<special>",
        "gold_tag": ID2LABEL[sentence["ner_tags"][wid]] if wid is not None else "-",
    })
split_table = pd.DataFrame(rows)
print("sentence:", " ".join(sentence["tokens"]))
print("words:", sentence["tokens"])
print("split word:", sentence["tokens"][word_id],
      "->", tokenizer.tokenize(sentence["tokens"][word_id]))
display(split_table)
"""
        ),
        md(
            """
## 3. Label alignment

Hai strategy của lab:

* **A — first subword**: `play` nhận `B-X`, `##ing` nhận `IGNORE` (-100).
* **B — propagate**: `play` nhận `B-X`, `##ing` nhận `I-X`.

`IGNORE` không có nghĩa token không tồn tại; nó chỉ bị loại khỏi loss:

$$L = -\\sum_{i:\\,y_i \\ne IGNORE} \\log P(y_i \\mid x)$$
"""
        ),
        code(
            """
alignment_table = pd.DataFrame({
    "token": tokenizer.convert_ids_to_tokens(enc["input_ids"]),
    "word_id": enc.word_ids(),
    "gold_word_tag": [
        ID2LABEL[sentence["ner_tags"][w]] if w is not None else "-"
        for w in enc.word_ids()
    ],
    "strategy_A": [ID2LABEL[l] if l != IGNORE else "IGNORE"
                   for l in align_tags(enc.word_ids(), sentence["ner_tags"], "first")],
    "strategy_B": [ID2LABEL[l] if l != IGNORE else "IGNORE"
                   for l in align_tags(enc.word_ids(), sentence["ner_tags"], "propagate")],
})
display(alignment_table)

n_ignore = sum(1 for l in align_tags(enc.word_ids(), sentence["ner_tags"], "first")
               if l == IGNORE)
print(f"IGNORE positions: {n_ignore} / {len(enc['input_ids'])} tokens in this sentence")
print("IGNORE index =", IGNORE)
"""
        ),
        md(
            """
## 4. Experiment 1 — thống kê trên toàn test set

Yêu cầu của W9 là tối thiểu 20 câu; ở đây dùng toàn bộ 3.453 câu test.
`R = #subword / #word`.
"""
        ),
        code(
            """
stats = read_json("predictions/tokenization_analysis.json")
summary = pd.DataFrame([
    ("sentences", stats["n_sentences"]),
    ("words", stats["n_words"]),
    ("subword tokens (no specials)", stats["n_subwords"]),
    ("subwords / word", round(stats["ratio_subwords_per_word"], 4)),
    ("words split into >1 piece", stats["n_split_words"]),
    ("split rate", round(stats["split_rate"], 4)),
    ("short sentences (<=15 words), ratio", round(stats["short_sentences"]["ratio"], 4)),
    ("long sentences (>15 words), ratio", round(stats["long_sentences"]["ratio"], 4)),
    ("split rate inside gold entities", round(stats["entity_words_split_rate"], 4)),
    ("split rate outside entities", round(stats["non_entity_words_split_rate"], 4)),
    ("max subwords in a sentence", stats["max_tokens"]),
    ("sentences longer than 128 subwords", stats["sentences_over_128_subwords"]),
], columns=["metric", "value"])
display(summary)
"""
        ),
        code(
            """
sentence_stats = pd.read_csv("tokenization_stats.csv")
sentence_stats["length_bucket"] = pd.cut(
    sentence_stats["n_words"], bins=[0, 15, 1000], labels=["short (<=15)", "long (>15)"]
)
display(sentence_stats.groupby("length_bucket", observed=True).agg(
    sentences=("id", "count"),
    mean_subwords_per_word=("ratio", "mean"),
    split_words=("n_split_words", "sum"),
).round(4))
print("Words split most often:")
display(pd.DataFrame(stats["most_split_words"], columns=["word", "sentences"]))
print("Most common continuation pieces:")
display(pd.DataFrame(stats["most_common_continuations"], columns=["##piece", "count"]))
"""
        ),
        md(
            """
## 5. Nhận xét

- WordPiece hiếm khi chia các từ thông dụng; các từ dài, tên riêng, chữ viết hoa
  có thể bị chia nhiều hơn; cần kiểm tra bằng bảng thống kê thực tế.
- Từ bị chia nhiều mảnh không tự động là lỗi alignment; cần kiểm chứng word_ids.
- Đối chiếu tỷ lệ ngắn/dài và entity/non-entity trong bảng; không giả định các
  tỷ lệ bằng nhau hoặc suy ra nguyên nhân lỗi chỉ từ correlation.
- Notebook chọn **Strategy A**: mỗi word có đúng một label, không nhân bản
  nhãn sang các mảnh `##`, giữ mỗi word một vị trí supervision.
  Continuation vẫn tham gia attention, chỉ không trực tiếp đóng góp loss.
"""
        ),
    ]
    return new_notebook(cells=cells)


def finetuning_notebook() -> nbformat.NotebookNode:
    cells = [
        md(
            """
# Fine-tuning BERT cho NER — LAB09

Notebook đọc kết quả **đã chạy thật** trong `predictions/` và `results.csv`.
Phần huấn luyện nằm trong `ner_finetuning.py`; cell cuối chỉ chạy lại khi bật
`RUN_TRAINING = True`.

Kiến trúc: `H ∈ R^(T×768)` -> `Z = H W + b`, `Z ∈ R^(T×9)` -> BIO.
"""
        ),
        code(
            """
import json

import pandas as pd
from IPython.display import Image, display

from ner_utils import LABELS, read_json

pd.set_option("display.width", 140)
main = read_json("predictions/transformer_main_metrics.json")
baseline = read_json("predictions/baseline_metrics.json")
print("model:", main["model"], "| device:", main["device"])
print("strategy:", main["strategy"], "| max_len:", main["max_len"],
      "| batch:", main["batch_size"], "| lr:", main["learning_rate"])
print("best epoch:", main["best_epoch"], "| best dev entity F1:",
      round(main["best_dev_entity_f1"], 4))
"""
        ),
        md("## 1. Learning curves (3 epochs, real log)"),
        code(
            """
history = pd.DataFrame(main["history"])
display(history.round(4))
display(Image("learning_curves.png"))
"""
        ),
        md(
            """
## 2. Baseline vs Transformer

Cùng test set, entity-level P/R/F1. Token accuracy được báo cáo thêm nhưng
không dùng để kết luận vì nó bị chi phối bởi lớp `O`.
"""
        ),
        code(
            """
results = pd.read_csv("results.csv")
display(results)
"""
        ),
        md(
            """
## 3. Entity-level evaluation theo loại entity

Một entity chỉ được tính TP khi **boundary và type** đều đúng.
"""
        ),
        code(
            """
entity_metrics = pd.read_csv("entity_metrics.csv")
display(entity_metrics[entity_metrics["group"] == "type"].round(4))
"""
        ),
        md("## 4. Entity-level F1 theo độ dài entity"),
        code(
            """
display(entity_metrics[entity_metrics["group"] == "length"].round(4))
display(Image("entity_length_analysis.png"))
"""
        ),
        md("## 5. Token-level confusion matrix"),
        code(
            """
display(Image("confusion_matrix.png"))
"""
        ),
        md(
            """
## 6. Alignment ablation — ảnh hưởng của Strategy A vs B

Hai lần chạy cùng hyperparameter, chỉ khác nhãn của các subword tiếp nối
(`IGNORE` so với `I-X`).
"""
        ),
        code(
            """
ablation = pd.read_csv("alignment_ablation.csv")
display(ablation)
"""
        ),
        md("## 7. Ứng dụng: NER -> structured information extraction"),
        code(
            """
from information_extraction import extract, load_model

tokenizer, model, device = load_model()
import torch
example_tokens = tokenizer(["Barack", "Obama", "visited", "Paris", "."],
                           is_split_into_words=True,return_tensors="pt")
with torch.inference_mode():
    logits = model(input_ids=example_tokens["input_ids"].to(device),
                   attention_mask=example_tokens["attention_mask"].to(device)).logits
print("logits shape:",tuple(logits.shape))
print("first-token logits:",logits[0,1].float().cpu().tolist())
print("token predictions:",[LABELS[i] for i in logits[0].argmax(-1).tolist()])
demo = extract(
    "Barack Obama visited Paris to meet Microsoft executives. "
    "The United Nations Security Council met in New York.",
    tokenizer, model, device,
)
print(json.dumps(demo["entities"], ensure_ascii=False, indent=2))
"""
        ),
        md("## 8. Error examples"),
        code(
            """
errors = read_json("predictions/error_examples.json")
print("summary:", json.dumps(errors["summary"], indent=2))
display(pd.DataFrame(errors["false_positives"])[
    ["text", "gold", "prediction", "error_type", "explanation"]
])
display(pd.DataFrame(errors["false_negatives"])[
    ["text", "gold", "prediction", "error_type", "explanation"]
])
"""
        ),
        md(
            """
## 9. Tuỳ chọn chạy lại huấn luyện

Mặc định tắt để notebook mở ra là xem được ngay và không train ngoài ý muốn.
Bật `RUN_TRAINING = True` sẽ chạy 3 epoch (~10 phút trên RTX 4060) và ghi đè
checkpoint/kết quả.
"""
        ),
        code(
            """
RUN_TRAINING = False
if RUN_TRAINING:
    import ner_finetuning
    ner_finetuning.train(ner_finetuning.parse_args([]))
else:
    print("Skipped. Set RUN_TRAINING = True to re-run the 3-epoch fine-tuning.")
"""
        ),
    ]
    return new_notebook(cells=cells)


def execute(notebook, name: str) -> None:
    path = ROOT / name
    client = NotebookClient(
        notebook,
        timeout=1800,
        startup_timeout=60,
        kernel_name="lab9-python",
        resources={"metadata": {"path": str(ROOT)}},
        allow_errors=False,
    )
    print(f'Executing {name}',flush=True)
    client.execute()
    with path.open("w", encoding="utf-8") as fh:
        nbformat.write(notebook, fh)
    print(f"wrote and executed {path}")


def main() -> None:
    # Register the kernel in this workspace, leaving the user's global setup alone.
    jupyter = ROOT / '.jupyter'
    kernel = jupyter / 'kernels' / 'lab9-python'
    kernel.mkdir(parents=True, exist_ok=True)
    (kernel/'kernel.json').write_text(json.dumps({
        'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
        'display_name':'LAB9 Python','language':'python'}),encoding='utf-8')
    os.environ['JUPYTER_PATH']=str(jupyter)
    os.environ['JUPYTER_RUNTIME_DIR']=str(jupyter/'runtime')
    os.environ['IPYTHONDIR']=str(jupyter/'ipython')
    execute(tokenizer_notebook(), "tokenizer_alignment.ipynb")
    execute(finetuning_notebook(), "ner_finetuning.ipynb")


if __name__ == "__main__":
    main()
