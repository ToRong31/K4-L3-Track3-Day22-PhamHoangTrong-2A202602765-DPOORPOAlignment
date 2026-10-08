# Bản Colab sẵn chạy — Lab 22

Mở `colab/Lab22_DPO_T4.ipynb` trong Google Colab. Chọn T4 GPU và Run all, cho phép kết nối Drive.
File T4 chính đã chứa toàn bộ quy trình tự động. `Lab22_READY_ALL.ipynb` là bản tương đương để tương thích với hướng dẫn cũ.
Tên/mã học viên đặt theo tên repo; kiểm tra cell cấu hình đầu nếu cần chỉnh.

## Phần đã chuẩn bị

- NB0 có DPO loss, assert và giải thích likelihood displacement.
- NB1 lưu GPU, cấu hình, thời gian, loss và log SFT.
- NB2 in 3 cặp, kiểm tra không trùng prompt, lưu phân bố độ dài.
- NB3 lưu đường train/held-out riêng chosen/rejected, runtime, peak VRAM và metrics.
- NB4 sinh 8 câu cố định + 50 held-out, hội đồng RM, bootstrap CI và kiểm tra độ dài.
- Bonus: 5 biến thể (+8), GGUF (+4), benchmark (+6), GRPO (+8), quét β (+6). Tổng bonus được rubric giới hạn +20.
- Chấm chéo API (+4) và HF Hub (+3) có cấu hình trong cell đầu, chỉ chạy khi bạn cung cấp thông tin/key.
- Tự tạo REFLECTION và model card bằng số liệu thực tế, chạy verify, lưu zip sau mỗi phần.

## Điểm cần chú ý khi chạy

`LOCAL_MODELS=True` (mặc định): mô hình SFT đã gộp và GGUF ở `/content/lab22-runtime`, adapter và kết quả vẫn trên Drive.
Phiên mới gộp lại SFT từ adapter, không cần huấn luyện lại SFT chỉ để khôi phục trọng số. Muốn tải GGUF hoặc mô hình,
tải từ đường dẫn tạm trước khi mất phiên. Bản Drive cũ có thể chuyển bằng `scripts/local_model_storage.py`;
chỉ thêm `--remove-drive-copy` khi muốn xóa riêng `models/sft-merged` sau khi kiểm tra bản sao.

Toàn bộ bonus có thể mất nhiều giờ và vượt hạn mức T4 miễn phí. Lượt huấn luyện bị ngắt phải chạy lại lượt đó;
Drive giữ các mô hình và notebook của phần đã hoàn thành. Cấu hình đổi sẽ làm lại core để tránh dùng kết quả cũ.
`ONLY="core"` chạy phần bắt buộc. `ONLY="variants"`, `"gguf"`, `"benchmark"`, `"grpo"`, `"beta"` chạy riêng bonus.
`ONLY="finish"` cập nhật báo cáo và zip. `RESUME=True` bỏ qua phần hoàn thành có cùng mã nguồn/cấu hình.
Nếu OOM, giảm `GEN_BATCH_SIZE`; giảm `MAX_LEN` cần huấn luyện lại core để bảo đảm dữ liệu và adapter khớp nhau.

GGUF: notebook in tên Q4_K_M và câu trả lời trong cell cuối. Theo rubric, chụp màn hình thủ công và lưu
`submission/screenshots/06-gguf-smoke.png` vào repo. Không tính bonus nếu cell chạy lỗi hoặc thiếu bằng chứng.

## Chấm chéo và HF Hub

Chấm chéo: điền `CROSS_JUDGE_PROVIDER`, `CROSS_JUDGE_MODEL`, thêm key tương ứng trong Colab Secrets.
API có thể tính phí; mặc định không gọi. Kết quả chấm trên đúng file đầu ra NB4, báo agreement và position consistency.
Hub: điền `HF_REPO_ID` và Secret `HF_TOKEN` quyền write. Sẽ tạo hai repo: adapter và SFT reference (~8 GB).
Adapter phụ thuộc mô hình SFT riêng; không thể chỉ upload LoRA và trỏ về mô hình gốc. Dữ liệu preference chưa có
giấy phép rõ ràng, nên chỉ dùng bản thử nghiệm cho học tập/nghiên cứu; tự xem điều khoản trước khi bật công khai.

## Nộp bài

Lấy `Lab22_submission.zip` trong thư mục Drive hoặc tải từ cell cuối, giải nén vào repo hiện tại.
Đọc phản tư tự sinh: kiểm tra 3 cặp NB2, nhận xét cụ thể về 1 ví dụ hữu ích + 1 ví dụ an toàn NB4,
điều chỉnh lập luận theo quan sát của bạn. Các đoạn tự động ghi rõ AI hỗ trợ, không bịa trải nghiệm cá nhân.
Commit notebook với output, ảnh, JSON/parquet và phản tư; nộp link GitHub public vào LMS.
Verify chạy trong Colab nơi còn mô hình tham chiếu; zip không chứa trọng số nên verify trên máy chỉ có zip
sẽ báo thiếu mô hình là điều dự kiến. Không commit trọng số, token hay `.env`.

`BONUS-CHALLENGE.md` là dự án ứng dụng không chấm điểm, yêu cầu ít nhất 100 cặp tự tạo theo chuyên môn của học viên.
Không gán dữ liệu AI sinh thành dữ liệu do học viên tự phán đoán và không đánh dấu dự án này hoàn thành.
