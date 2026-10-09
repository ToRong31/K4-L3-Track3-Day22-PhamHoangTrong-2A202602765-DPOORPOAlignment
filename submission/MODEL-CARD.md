---
base_model: unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit
language:
- vi
library_name: peft
tags:
- dpo
- experimental
---

# Lab22 SFT+DPO experimental adapter

Học viên: Phạm Hoàng Trọng (A20-K4 · 2A202602765). Mục tiêu: thực nghiệm căn chỉnh hội thoại tiếng Việt phục vụ học tập.
Adapter cần mô hình SFT riêng đã gộp ở `models/sft-merged`; không nạp trực tiếp trên base gốc.
SFT: saillab/alpaca-vietnamese-cleaned, 1000 mẫu. Preference: sailor2/sea-ultrafeedback-onpolicy, 800 train / 100 held-out.
β=0.1, lr=5e-06, epoch=1.0, seed=42.

Held-out reward accuracy: 0.6500. NB4 win rate: 0.4800, CI: [0.41, 0.55].
Giám khảo: rm-panel:Skywork/Skywork-Reward-V2-Llama-3.2-3B. Kết luận: chưa đủ bằng chứng DPO tốt hơn SFT vì CI chứa 0,5.

Không dùng cho quyết định y tế/pháp lý, triển khai sản phẩm hoặc khẳng định an toàn đầy đủ.
Dữ liệu preference chưa ghi giấy phép rõ ràng; chỉ dùng cho học tập/nghiên cứu, không tự gán giấy phép cho trọng số dẫn xuất.
Giới hạn: mẫu nhỏ, một seed, thiên vị độ dài, giám khảo cùng Skywork với nguồn nhãn. Có thể sai kiến thức hoặc sinh nội dung không phù hợp.
Code hỗ trợ được tạo bằng AI và kiểm tra bằng CPU tests; kết quả GPU và phản tư được ghi từ lần chạy notebook thực tế.
