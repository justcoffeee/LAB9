# LAB09 - Named Entity Recognition and Information Extraction

Theo W9.pdf, lưu trong `lab9/`. Repository chứa nội dung lab9 ở thư mục gốc.
Mã nguồn và log huấn luyện đã tồn tại khi bắt đầu phiên hoàn thiện; chương trình
`verify_checkpoints.py` chạy lại test từ cả hai checkpoint để kiểm tra kết quả.
Xem `predictions/checkpoint_verification.json` để biết kết quả đối chiếu thực tế.

## Kết quả

CoNLL-2003 official splits: train 14041 / validation 3250 / test 3453 câu.
Nhận diện PER/ORG/LOC/MISC với 9 BIO labels; entity phải đúng **cả span và type**.

| Model | Entity precision | Entity recall | Entity F1 |
|---|---|---|---|
| Token features + Logistic Regression | 0.7142 | 0.7757 | 0.7437 |
| BERT-base-cased, Strategy A | 0.9072 | 0.9189 | 0.9130 |
| BERT-base-cased, Strategy B | 0.9071 | 0.9212 | 0.9141 |

Primary model là Strategy A (first subword); Strategy B là ablation bổ sung,
không đổi primary theo test F1. Mỗi BERT run 3 epoch, max length128 bao gồm
special tokens, batch16, AdamW3e-5, weight_decay0.01, warmup10% + linear decay,
clip-grad1.0, seed42; checkpoint chọn theo validation entity F1. BF16 autocast,
GPU RTX4060 Laptop. Tái lập cùng seed không bảo đảm bitwise trên phần cứng khác.
Train loss là mean theo batch, không phải mean có trọng số theo số valid token.
Thời gian train_seconds ở metrics gồm validation/checkpoint I/O giữa các epoch.

Baseline dùng surface/case/shape/affixes, ngữ cảnh +/-2 token và **gold POS** của
CoNLL, fit DictVectorizer chỉ trên train. Baseline có POS supervision bổ sung mà
BERT không nhận trực tiếp, nên không phải so sánh text-only tuyệt đối; không gọi
baseline là context-free. NER labels/chunk tags không làm feature đầu vào.
Baseline dừng ở giới hạn200 iterations (xem convergence warning trong log);
không khẳng định đã tối ưu hoặc hội tụ hoàn toàn. Hai BERT run chỉ có một seed.

## Chạy lại

Trong thư mục repository:

```powershell
python -m pip install -r requirements.txt
python download_data.py
python -u run_experiments.py
```

Mặc định chạy baseline, train2 BERT run, reports, IE30 câu, docs, notebooks,
validation. Ghi đè artifact tương ứng khi chạy lại. Trên RTX4060, log cũ ghi
khoảng10-11 phút cho mỗi BERT run; đây không phải cam kết thời gian máy khác.

Nếu checkpoint và prediction đã có, chỉ tạo lại báo cáo:

```powershell
python run_experiments.py --skip-baseline --skip-training --skip-ablation
python verify_checkpoints.py
python information_extraction.py --text "Barack Obama visited Paris."
```

Notebook mặc định đọc kết quả sẵn có; đặt RUN_TRAINING=True trong
ner_finetuning.ipynb để train lại main. Kernel phục vụ việc build notebook nằm
trong `.jupyter/` cục bộ, không cài vào profile Jupyter người dùng.

## Alignment và đánh giá

- Strategy A: label ở first subword, continuation/CLS/SEP/PAD=-100.
- Strategy B: continuation của B-X/I-X nhận I-X, O tiếp tục O. Special/PAD=-100.
- Decode ở word level bằng first subword ở cả hai mode; không gọi pipeline("ner").
- Token ngoài đoạn bị truncation được dự đoán O khi chấm toàn câu; gold vẫn
  giữ đủ, nên không loại bỏ entity khó ở phần bị cắt để làm đẹp score.
- Decoder quy ước I-X không nối được thì bắt đầu entity X mới (permissive BIO).
  Entity score vẫn exact-span-and-type, nhưng đây không phải strict IOB2
  transition validation. Báo token accuracy/F1 riêng; không đồng nhất token và entity.
- Length buckets1/2/3/4+ theo số word. TP/FN theo độ dài gold; unmatched FP
  gán vào bucket gold overlap lớn nhất (tie theo span có thứ tự), FP không overlap
  theo độ dài prediction. Đây là thống kê quy lỗi theo length, không chuẩn duy nhất.
- Tokenization statistics dùng **toàn câu không truncation**, tách content
  subwords khỏi CLS/SEP, có so sánh câu ngắn/dài và entity/non-entity.
- Gold30 câu IE lấy từ annotation có sẵn của 30 câu test đầu, không tự nhận
  đã gán nhãn mới. Đánh giá output raw-text bằng exact character offsets và type.
  Đó là sample thuận tiện nhỏ, không phải một test set độc lập hay domain mới.

Raw text IE tách punctuation riêng, giữ offsets, output PERSON/ORGANIZATION/
LOCATION/MISC và danh sách mentions. Tách câu là heuristic; mỗi chunk tối đa
256 subword và có thể truncation với câu rất dài. Mô hình không suy ra relations
hay giải quyết entity linking/coreference.

## Tệp nộp

| Tệp | Nội dung |
|---|---|
| calculations.md / prediction.md | Tài liệu tham chiếu và mục prediction cá nhân chưa cung cấp |
| baseline_ner.py | Feature-based Logistic Regression baseline |
| tokenizer_alignment.ipynb | Words/subwords/IDs/masks/labels, statistics toàn test |
| ner_finetuning.py / ner_finetuning.ipynb | BERT token head, train/eval, notebook đã chạy |
| results.csv / entity_metrics.csv | Entity P/R/F1 tổng, theo type và length |
| entity_length_analysis.png / confusion_matrix.png | Biểu đồ từ prediction thật |
| tokenization_stats.csv / tokenization_ratio.png | Phân tích tokenizer3453 câu, hơn mức20 câu đề yêu cầu |
| error_analysis.md / predictions/error_examples.json | 5 FP,5 FN,5 boundary errors và3 contextual surface cases |
| information_extraction.py / ie_evaluation.csv | Raw text -> structured JSON, chấm30 gold sentences |
| reflection.md | Hướng dẫn reflection300-500 từ và learning check cá nhân |
| history/ / predictions/ | Loss, metrics, prediction, kiểm chứng và log |
| download_data.py / assets_manifest.json | Download/provenance; không đưa dataset/checkpoint lên Git |

## Trạng thái phần cá nhân và AI-assisted code

Theo W9 mục28, sinh viên tự làm BIO exercise, alignment reasoning, prediction,
metric interpretation, error analysis và reflection. `calculations.md` là đáp án
tham chiếu có AI hỗ trợ; cần bản tính/giải thích của sinh viên. `prediction.md`
không điền dự đoán giả sau khi kết quả đã có. Error analysis tự động chỉ mô tả
span disagreement; sinh viên cần tự diễn giải nguyên nhân. `reflection.md` là
dàn ý, chưa phải bài reflection cá nhân300-500 từ đã hoàn tất.

- Tool: OpenAI Codex; Python, PyTorch, Transformers, sklearn, pandas, matplotlib,
  nbformat/nbclient/ipykernel.
- What was generated: code/report/notebooks và tài liệu kỹ thuật trong lab9.
- What was modified trong phiên hoàn thiện: sửa taxonomy boundary, thống kê
  tokenizer không cắt câu, kiểm tra split ID theo namespace, raw-text punctuation
  và chấm IE theo character offsets, thêm kiểm chứng checkpoint/provenance,
  tài liệu tái lập và chuẩn bị GitHub.
- What was verified: xem `predictions/validation.json`,
  `predictions/checkpoint_verification.json` và `assets_manifest.json`; không
  ghi nhận các kết quả validation PASS trước khi chúng được chạy thực tế.

Giới hạn: một seed; labels/annotation có thể mơ hồ; không bảo đảm các type/length
khác nhau có cùng số mẫu; chưa đo domain shift ngoài CoNLL; không coi score cao
trên Reuters news là bảo đảm NER trên văn bản y sinh hay tiếng Việt.

## Nguồn

- [CoNLL-2003 dataset](https://huggingface.co/datasets/eriktks/conll2003).
- [BERT-base-cased](https://huggingface.co/google-bert/bert-base-cased).
- [Hugging Face token classification](https://huggingface.co/docs/transformers/tasks/token_classification).
