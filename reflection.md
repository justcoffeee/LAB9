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
