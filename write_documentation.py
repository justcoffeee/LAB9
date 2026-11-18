"""Write the lab guide and explicitly separate technical evidence from personal work."""
from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parent


def write(name,text):
    (ROOT/name).write_text(text.strip()+'\n',encoding='utf-8')


def main():
    baseline=json.loads((ROOT/'predictions/baseline_metrics.json').read_text())
    main=json.loads((ROOT/'predictions/transformer_main_metrics.json').read_text())
    propagate=json.loads((ROOT/'predictions/transformer_propagate_metrics.json').read_text())
    b=baseline['splits']['test']['entity'];m=main['splits']['test']['entity']
    p=propagate['splits']['test']['entity']
    from download_data import DATA_FILES, MODEL_FILES, BERT_REVISION
    def digest(path):
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
        return h.hexdigest()
    assets={}
    for name,meta in DATA_FILES.items():
        path=ROOT/'data'/name
        assets[str(path.relative_to(ROOT))]={'url':meta[0],'bytes':path.stat().st_size,'sha256':digest(path)}
    for name,url in MODEL_FILES.items():
        path=ROOT/'model/bert-base-cased'/name
        assets[str(path.relative_to(ROOT))]={'url':url,'bytes':path.stat().st_size,'sha256':digest(path)}
    manifest={'recorded_utc':datetime.now(timezone.utc).isoformat(),
              'model_id':'google-bert/bert-base-cased','model_revision':BERT_REVISION,'assets':assets,
              'source_sha256':{path.name:digest(path) for path in ROOT.glob('*.py')},
              'prepared_initial_encoder':'197/197 encoder tensors matched original upstream exactly; see predictions/initial_encoder_provenance.json',
              'tokenizer_equivalence':'vocabulary, input IDs, attention masks and word_ids matched upstream on all 3453 test sentences'}
    (ROOT/'assets_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    write('README.md',f'''
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
| Token features + Logistic Regression | {b['precision']:.4f} | {b['recall']:.4f} | {b['f1']:.4f} |
| BERT-base-cased, Strategy A | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} |
| BERT-base-cased, Strategy B | {p['precision']:.4f} | {p['recall']:.4f} | {p['f1']:.4f} |

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
''')
    write('calculations.md','''
# W9 calculations - tham chiếu kiểm chứng có AI hỗ trợ

W9 mục28 yêu cầu sinh viên tự gán BIO và giải thích alignment. Phần sau là tham
chiếu; sinh viên cần trình bày lại bằng lời của mình, không coi là bài tự làm.

## Exercise1: BIO

Với giả thiết đề: Elon Musk=PER, SpaceX=ORG, California=LOC:

| Word | BIO label |
|---|---|
| Elon | B-PER |
| Musk | I-PER |
| founded | O |
| SpaceX | B-ORG |
| in | O |
| California | B-LOC |
| . | O |

## Exercise2: entity boundary

`University/B-ORG of/I-ORG California/I-ORG Berkeley/I-ORG`.
Một label ORG cho mọi từ không phân biệt hai ORG liền nhau với một ORG nhiều
từ. B đánh dấu bắt đầu, I là phần tiếp tục của cùng entity. Các spans trong code
có end-exclusive; entity trên có word span[0,4).

## Exercise3: shape

Với B=8,T=128,d=768,C=9:

- Hidden H:8×128×768.
- W:768×9, b:9 nếu viết H@W+b. nn.Linear lưu weight dạng9×768.
- Logits Z:8×128×9.
- Labels Y:8×128, dtype long; IGNORE=-100.
- CrossEntropyLoss có thể dùng Z.reshape(-1,9), Y.reshape(-1).

Sequence classification dùng một representation (thường CLS) ->B×C. Token
classification dùng mọi vị trí ->B×T×C; C của hai task không nhất thiết giống.

## Alignment

Ví dụ giả định một word B-PER bị chia3mảnh:

| Position | word_id | Strategy A | Strategy B |
|---|---|---|---|
| CLS | None | -100 | -100 |
| first piece | 0 | B-PER | B-PER |
| continuation1 | 0 | -100 | I-PER |
| continuation2 | 0 | -100 | I-PER |
| SEP | None | -100 | -100 |

Đây là sơ đồ, không dự đoán tokenizer sẽ chia một từ cụ thể. Notebook in kết
quả thật từ tokenizer cased. IGNORE chỉ loại nhãn khỏi loss; subword vẫn tồn tại
và tham gia attention nếu attention_mask=1. PAD vừa có mask0 vừa label-100.
Mean CE framework chia cho số vị trí không IGNORE; Strategy B có thể làm các
word bị chia nhiều mảnh đóng góp loss nhiều hơn. Kết quả ablation ở
alignment_ablation.csv, sinh viên tự lý giải; không mặc định tăng supervision
luôn cải thiện chất lượng.

## Token correctness và entity correctness

Gold `New/B-LOC York/I-LOC`, prediction `New/B-LOC York/O`: token accuracy=1/2,
nhưng entity gold[0,2) khác pred[0,1), nên TP=0,FP=1,FN=1, entity F1=0.
Precision=TP/(TP+FP), recall=TP/(TP+FN), F1=2PR/(P+R). Nhiều token O có thể
làm accuracy cao dù entity recall thấp. Code quy ước metric0 khi mẫu số0.

Sinh viên tự giải thích: tại sao boundary là phần bắt buộc? First-piece decoding
khác lấy argmax mọi subword rồi xem mỗi subword là một word như thế nào?
''')
    write('prediction.md','''
# Prediction trước experiment - chưa có câu trả lời cá nhân

Khi hoàn thiện thư mục này, log/model training đã có sẵn. Không thể tạo prediction
trước thí nghiệm hồi tố. W9 mục28 yêu cầu sinh viên tự dự đoán; giữ nguyên câu trả
lời gốc nếu có, hoặc ghi rõ đây là trả lời sau experiment.

| # | Câu hỏi | Prediction / Reason cá nhân | Evidence sau experiment |
|---|---|---|---|
| 1 | Pretrained Transformer so với token-feature classifier thay đổi NER thế nào? | Chưa cung cấp | results.csv |
| 2 | Entity dài nhiều token dễ/khó hơn entity1token? Vì sao? | Chưa cung cấp | entity_metrics.csv, entity_length_analysis.png |
| 3 | Entity hiếm trong train có dễ lỗi hơn không? Vì sao? | Chưa cung cấp | error_examples.json, frequency evidence trong error_analysis.md |

Đối chiếu đúng/sai và giải thích cá nhân: sinh viên bổ sung; không đánh giá khi
chưa có prediction gốc. Đếm số lỗi unseen chưa đủ kết luận unseen có error rate
cao hơn seen nếu chưa biết tổng support mỗi nhóm.
''')
    write('reflection.md','''
# Reflection cá nhân - sinh viên hoàn thiện300-500 từ

W9 mục28 không cho AI làm thay reflection. Câu hỏi mục32:

> Tại sao cùng một pretrained Transformer có thể dùng cho sequence classification
> và token classification chỉ bằng cách đổi prediction head?

Nội dung cá nhân: **chưa được cung cấp**. Dàn ý hỗ trợ:

1. So sánh output LAB8 (một nhãn/câu) với LAB9 (BIO trên từng token).
2. Mô tả backbone tạo contextual states H; CLS->classifier so với
   every-token->Linear9. Không đồng nhất đổi head với không cần fine-tune.
3. Giải thích nhãn/loss, word-subword alignment, IGNORE/PAD và decode entity.
4. Trích ít nhất2số từ kết quả thật: entity F1 khác token accuracy, per-type hoặc
   length, ablationA/B. Phân biệt observation, evidence, interpretation.
5. Phân tích một FP/FN/boundary case và một same-surface/different-type context;
   nêu giới hạn rare entity, annotation ambiguity, truncation/domain shift.
6. Vai trò NER->structured IE; LAB10 chuyển sang head dự đoán start/end span.

## Individual learning check - tự trả lời

1. Sequence classification khác token classification như thế nào?
2. BIO dùng làm gì, B-PER khác I-PER ở đâu?
3. Tại sao tokenizer gây alignment problem?
4. IGNORE có ý nghĩa gì với loss và attention?
5. Tại sao không chỉ đánh giá token accuracy?
6. Entity-level F1 khác token-level F1 thế nào?
7. Vì sao boundary khó, đặc biệt với entity dài?
8. Contextual representations giúp gì và chưa bảo đảm điều gì?
9. NER hỗ trợ IE như thế nào; IE có đồng nghĩa hiểu relations không?

Không coi dàn ý này là bài reflection300-500 từ đã hoàn tất.
''')


if __name__=='__main__':main()
