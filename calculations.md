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
