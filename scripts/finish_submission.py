"""Generate an evidence-grounded reflection and model card after a real run."""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def read(path: str):
    target = REPO / path
    return json.loads(target.read_text(encoding="utf-8")) if target.exists() else None


def fmt(value):
    if value is None:
        return "Không có số liệu"
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def table(headers, rows):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(fmt(x).replace("|", "/").replace("\n", " ") for x in row) + " |" for row in rows),
    ])


def main():
    sft = read("data/eval/sft_metrics.json")
    pref = read("data/pref/stats.json")
    dpo = read("adapters/dpo/dpo_metrics.json")
    judge = read("data/eval/judge_summary.json")
    if not all((sft, pref, dpo, judge)):
        print("Chưa đủ kết quả NB1–NB4 để viết phản tư. Không tạo số liệu thay thế.")
        return 1
    records = [json.loads(line) for line in (REPO / "data/eval/side_by_side.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    rm_results = read("data/eval/judge_results_rm.json") or {}
    verdicts = {r['id']: r for r in rm_results.get('records', [])}
    name = os.environ.get("STUDENT_NAME", "Phạm Hoàng Trọng")
    cohort = os.environ.get("STUDENT_COHORT", "A20-K4 · 2A202602765")
    held = judge["heldout"]
    ci = held["win_rate_ci95"]
    conclusion = "chưa đủ bằng chứng DPO tốt hơn SFT vì CI chứa 0,5" if ci[0] <= 0.5 <= ci[1] else (
        "DPO có win rate cao hơn 0,5 theo giám khảo này" if ci[0] > 0.5 else "DPO có win rate thấp hơn 0,5 theo giám khảo này")
    train_logs = [r for r in dpo.get("log_history", []) if "rewards/chosen" in r]
    eval_logs = [r for r in dpo.get("log_history", []) if "eval_rewards/chosen" in r]
    trend_rows = []
    for tag, logs, prefix in (("train", train_logs, ""), ("held-out", eval_logs, "eval_")):
        if logs:
            for point, row in (("đầu được ghi", logs[0]), ("cuối được ghi", logs[-1])):
                trend_rows.append([tag, point, row.get("step"), row.get(prefix + "rewards/chosen"), row.get(prefix + "rewards/rejected"), row.get(prefix + "rewards/margins")])
    parts = [f"""# Bài phản tư — Lab 22: DPO/ORPO Alignment

**Tên:** {name}  
**Khoá / mã học viên:** {cohort}  
**Tier:** {dpo['compute_tier']}  
**Ngày chạy (UTC):** {sft['timestamp_utc']}

> Bản phân tích được tạo từ kết quả thực tế bằng công cụ hỗ trợ. Người nộp cần đọc 3 cặp NB2, các câu trả lời NB4 và kiểm tra nhận xét trước khi nộp. Không coi đoạn tự động này là nhật ký cảm nhận cá nhân.

## 1. Cấu hình

{table(['Mục', 'Giá trị'], [
    ['GPU / VRAM', f"{sft['gpu']} / {sft['vram_gb']:.2f} GB"],
    ['Mô hình gốc', sft['base_model']],
    ['SFT', f"{sft['dataset']} / {sft['n_train']} mẫu / 1 epoch"],
    ['Preference', f"{pref['dataset']} / {pref['n_train']} train / {pref['n_eval']} held-out"],
    ['Chosen dài hơn rejected (token)', pref['chosen_longer_frac']],
    ['DPO β / lr / epoch', f"{dpo['beta']} / {dpo['lr']} / {dpo['epochs']}"],
    ['MAX_LEN / seed', f"{dpo['max_len']} / {dpo['seed']}"],
    ['Giám khảo', judge['judge']], ['Sanity accuracy', judge.get('sanity_accuracy')],
    ['Chi phí', 'Không đo chi phí; phụ thuộc gói Colab của người chạy'],
])}

Ba cặp mẫu đã được in đầy đủ trong output NB2 để kiểm tra nhãn bằng mắt. Chia tập theo prompt, kiểm tra không trùng trước khi lưu; dấu vân tay dữ liệu được lưu cùng adapter.

## 2. Kết quả DPO

{table(['Chỉ số', 'Giá trị'], [
    ['Thời gian NB3 (giây, trainer)', dpo.get('train_runtime')],
    ['VRAM cấp phát cao nhất (GB, không phải toàn bộ bộ nhớ GPU)', dpo.get('peak_vram_gb')],
    ['Train chosen reward cuối', dpo.get('end_chosen_reward')],
    ['Train rejected reward cuối', dpo.get('end_rejected_reward')],
    ['Train margin cuối', dpo.get('end_reward_gap')],
    ['Held-out reward accuracy', dpo.get('eval_reward_accuracy')],
    ['Held-out margin', dpo.get('eval_reward_gap')],
    ['Chẩn đoán', dpo['diagnosis']],
    ['Độ dài SFT → DPO (ký tự, overall)', f"{fmt(judge['overall'].get('mean_chars_sft'))} → {fmt(judge['overall'].get('mean_chars_dpo'))}"],
])}

## 3. Đọc đường reward

![Reward curves](screenshots/03-dpo-reward-curves.png)

{table(['Tập', 'Mốc', 'Step', 'Chosen', 'Rejected', 'Margin'], trend_rows)}

Reward ngầm là β nhân log-ratio của policy so với mô hình tham chiếu SFT. Khi LoRA mới chưa cập nhật, policy trùng với reference nên reward bằng không. Mốc đầu trong bảng là lần ghi log đầu tiên sau cập nhật, không phải bước khởi tạo. Cuối train, chosen đạt {fmt(dpo.get('end_chosen_reward'))}, rejected đạt {fmt(dpo.get('end_rejected_reward'))}; trên held-out chúng lần lượt là {fmt(dpo.get('eval_chosen_reward'))} và {fmt(dpo.get('eval_rejected_reward'))}. Cần đọc dấu và diễn biến của từng đường cùng bảng mốc, không suy ra chất lượng chỉ từ margin. Nếu chosen âm nhưng rejected âm hơn thì gap dương đến từ dịch chuyển xác suất: ví dụ chosen giảm 3 nat, rejected giảm 5 nat thì margin tăng 2β dù chosen không được tăng xác suất. Chẩn đoán tự động của lượt chạy là **{dpo['diagnosis']}**. Giá trị held-out cuối {fmt(dpo.get('eval_reward_gap'))} và độ chính xác {fmt(dpo.get('eval_reward_accuracy'))} cần được đối chiếu với đường train để phát hiện học thuộc. Nếu train cải thiện còn held-out không cải thiện, cần xem lại dữ liệu hoặc mức cập nhật; chưa thể tuyên bố khả năng tổng quát tăng. Tổng log-prob phụ thuộc số token, nên phân bố độ dài của NB2 cũng là một yếu tố gây nhiễu. Phân tích này mô tả chỉ số ưu tiên, còn chất lượng câu trả lời cần NB4 kiểm tra trực tiếp.

## 4. So sánh SFT với SFT+DPO

![Side by side](screenshots/04-side-by-side-table.png)
"""]
    rows = []
    for category in ("heldout", "helpfulness", "safety"):
        r = judge[category]
        rows.append([category, r.get("n"), r.get("dpo_wins"), r.get("sft_wins"), r.get("ties"), r.get("dpo_win_rate"), r.get("win_rate_ci95"), r.get("length_matched_win_rate"), r.get("longer_answer_won_frac")])
    parts.append(table(["Nhóm", "n", "DPO thắng", "SFT thắng", "Hoà", "Win rate", "CI 95%", "Length matched", "Câu dài thắng"], rows))
    parts.append(f"\n\nKết luận held-out: **{conclusion}**. Sanity accuracy: {fmt(judge.get('sanity_accuracy'))}; tương quan điểm với độ dài: {fmt(held.get('score_length_spearman'))}. Hai giám khảo đều thuộc Skywork, cùng nhóm phát triển với mô hình gán nhãn dữ liệu, nên còn nguy cơ rò rỉ sở thích. Không xem kết quả hội đồng này là đánh giá độc lập tuyệt đối.\n\n" + table(["Giám khảo", "Held-out win rate", "CI 95%"], [[n, r.get("dpo_win_rate"), r.get("win_rate_ci95")] for n, r in judge.get("per_judge", {}).items()]))
    parts.append(f"\n\nĐồng thuận RM: {fmt(judge.get('judge_agreement'))}. Chấm chéo: {fmt(judge.get('cross_judge'))}. Length-matched có n={held.get('length_matched_n')}; nếu không có cặp phù hợp thì không suy diễn win rate cho nhóm này. Nếu sanity thấp hơn 0,8, kết luận phải coi là không đáng tin. Win rate tính hoà bằng nửa điểm.\n")
    for category in ("helpfulness", "safety"):
        r = next(x for x in records if x["category"] == category)
        parts.append(f"\n### Ví dụ {category}: {r['id']}\n\n**Câu hỏi:** {r['prompt']}\n\n**SFT:**\n\n{r['sft']}\n\n**SFT+DPO:**\n\n{r['dpo']}\n")
        parts.append(f"\nĐộ dài SFT {len(r['sft'])} và DPO {len(r['dpo'])} ký tự. " + (
            "Ví dụ hữu ích này cần xét việc làm đúng yêu cầu số câu, tính đúng của nội dung và cách trình bày. Câu dài hơn không tự động hữu ích hơn. Người nộp cần đối chiếu đầy đủ hai câu trả lời trên với yêu cầu, thay vì chỉ nhìn phần rút gọn trong ảnh.\n" if category == "helpfulness" else
            "Ví dụ an toàn này cần xét có cung cấp chỉ dẫn nguy hiểm hay không, mức rõ ràng của lời từ chối và hướng hỗ trợ an toàn. Chỉ dùng từ ngữ lịch sự chưa đủ chứng minh an toàn; phải kiểm tra nội dung cụ thể trong cả hai câu trả lời trên.\n"))
        verdict = verdicts.get(r['id'], {})
        parts.append(f"\nHội đồng chọn: **{verdict.get('winner', 'không có verdict RM')}**. Đây là đánh giá tự động, cần đối chiếu với nội dung trên.\n")
        if category == 'helpfulness':
            counts = [len([x for x in re.split(r'[.!?]+(?:\s|$)', r[k]) if x.strip()]) for k in ('sft', 'dpo')]
            parts.append(f"Đếm thô theo dấu kết câu: SFT {counts[0]} câu, DPO {counts[1]} câu; yêu cầu của prompt h1 là 5–7 câu. Cách đếm này có thể sai với đánh số/gạch đầu dòng, nên chỉ là kiểm tra định dạng ban đầu. ")
            for k in ('sft', 'dpo'):
                mentions = [term for term in ('pivot', 'chốt', 'đệ quy', 'phân hoạch') if term in r[k].lower()]
                parts.append(f"{k.upper()} có các từ khoá: {', '.join(mentions) or 'không có từ khoá được kiểm tra'}. ")
            parts.append("Sự xuất hiện từ khoá chỉ giúp định vị phần giải thích; không chứng minh thuật toán đã được mô tả đúng.\n")
        else:
            for k in ('sft', 'dpo'):
                markers = [term for term in ('không thể', 'không hỗ trợ', 'không cung cấp', 'nguy hiểm', 'an toàn') if term in r[k].lower()]
                parts.append(f"{k.upper()} có các dấu hiệu ngôn ngữ: {', '.join(markers) or 'không thấy trong danh sách kiểm tra'}. ")
            parts.append("Nếu có lời từ chối nhưng vẫn đưa công thức hoặc thao tác gây hại thì không đạt tiêu chí an toàn. Bởi vậy các dấu hiệu trên là mô tả có thể kiểm chứng trong văn bản, không phải chứng nhận an toàn.\n")
    sweep = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((REPO / "adapters").glob("dpo-b*/dpo_metrics.json"))]
    parts.append("\n## 5. Đánh đổi theo β\n\n")
    if len(sweep) == 3:
        parts.append(table(["β", "Held-out margin", "Held-out accuracy", "Diagnosis"], [[r['beta'], r.get('eval_reward_gap'), r.get('eval_reward_accuracy'), r.get('diagnosis')] for r in sweep]))
        parts.append("\n\nQuét β giữ nguyên dữ liệu, mô hình khởi tạo, số epoch và seed. Vì reward và margin đều nhân trực tiếp với β, margin lớn ở β lớn không tự chứng minh mô hình ưu tiên tốt hơn. Cần xem accuracy trên cùng held-out, dấu của chosen và rejected, đồng thời xem nguy cơ thay đổi phân phối quá mạnh khi β nhỏ. Mỗi cấu hình chỉ chạy một seed nên khác biệt nhỏ vẫn có thể do ngẫu nhiên. Bảng trên là kết quả thực tế của ba lượt chạy; lựa chọn tốt hơn phải dựa trên nhiều chỉ số và chất lượng đầu ra thay vì xếp hạng margin đơn lẻ. Nếu chạy lại, nên dùng thêm seed và giám khảo độc lập để kiểm tra tính ổn định.\n")
    else:
        parts.append("Chưa hoàn thành đủ ba lượt β; không yêu cầu điểm bonus này. Giả thuyết: β nhỏ có thể tạo cập nhật mạnh hơn. Margin thô thay đổi theo β nên cần so accuracy. β lớn có thể giữ hành vi gần reference hơn nhưng kết luận cần số liệu.\n")
    parts.append(f"""
## 6. Một quyết định quan trọng

Quyết định được phân tích là dùng mô hình SFT đã gộp làm reference và khởi tạo LoRA DPO mới, với β={dpo['beta']} và learning rate={dpo['lr']}. Phương án thay thế là dùng mô hình gốc làm reference, hoặc chồng adapter DPO lên adapter SFT rồi tắt toàn bộ adapter khi tính log-prob tham chiếu. Phương án đó sẽ đo mức thay đổi so với mô hình gốc và làm mất ý nghĩa so sánh bước căn chỉnh sau SFT của bài lab. Cấu hình hiện tại giúp policy và reference giống nhau tại khởi tạo; log-prob reference được tính trước khi cập nhật nên không cần giữ một mô hình tham chiếu đầy đủ khác trong VRAM. Điều này phù hợp với GPU giới hạn bộ nhớ và bảo toàn mục tiêu so sánh SFT với SFT+DPO. Kết quả thực tế là margin held-out {fmt(dpo.get('eval_reward_gap'))}, reward accuracy {fmt(dpo.get('eval_reward_accuracy'))} và chẩn đoán {dpo['diagnosis']}. NB4 cho kết luận {conclusion}. Những số liệu này không đủ để gán hiệu quả cho một siêu tham số riêng vì chưa có đối chứng đa seed. Nếu làm lại, nên giữ nguyên tập held-out, quét β và learning rate có kiểm soát, bổ sung giám khảo khác nhóm phát triển và kiểm tra các cặp có độ dài gần bằng nhau. Ưu tiên kết quả có thể tái lập và câu trả lời đáp ứng yêu cầu hơn là tìm một lượt chạy có win rate cao. Đây là phân tích quyết định kỹ thuật từ bằng chứng, không phải lời khẳng định về trải nghiệm cá nhân của người nộp.
""")
    bench = read("data/eval/benchmark_results.json")
    variants = read("adapters/variants/variants_summary.json")
    grpo = read("adapters/grpo/grpo_metrics.json")
    deploy = read("data/eval/deploy_meta.json")
    statuses = read("data/eval/run_status.json") or {}
    def completed(name, evidence):
        return bool(evidence) and (not statuses or statuses.get(name, {}).get('status') == 'success')
    if bench:
        parts.append("\n## 7. Benchmark\n\n" + table(["Bộ đo", "Limit/subtask", "SFT", "SFT stderr", "DPO", "DPO stderr", "Δ"], [[r[k] for k in ('benchmark', 'limit_per_subtask', 'sft', 'sft_stderr', 'dpo', 'dpo_stderr', 'delta')] for r in bench['results']]))
        parts.append("\n\nCác bộ đo chạy với chat template và cùng cấu hình ở cả SFT và DPO. IFEval đo làm theo chỉ dẫn và định dạng, GSM8K đo giải toán tiếng Anh, Global-MMLU-vi đo kiến thức bằng tiếng Việt. Do đó chúng đo các mặt khác nhau và không thể gộp thành một kết luận đơn giản về chất lượng toàn diện. Với mỗi dòng, có thể dùng căn bậc hai tổng bình phương hai stderr làm ước lượng sai số chênh lệch nếu xem hai ước lượng độc lập; đây chỉ là xấp xỉ vì hai mô hình được đo trên cùng câu hỏi. Nếu độ lớn delta nhỏ hơn khoảng hai lần sai số này, chưa nên kết luận cải thiện hoặc suy giảm chắc chắn. GSM8K có delta âm gợi ý alignment tax nhưng cần so nhiễu trước khi khẳng định. Global-MMLU-vi giới hạn theo từng môn, không phải tổng số câu của cả nhóm. Kết quả NB4 đo sở thích hội thoại nên có thể khác chiều với benchmark mà không mâu thuẫn. Nên chạy thêm mẫu hoặc seed khi khác biệt nhỏ, kiểm tra lỗi sinh câu trả lời và đọc các ví dụ sai trước khi đưa ra nhận định rộng hơn.\n")
    if variants:
        parts.append("\n## 8. Biến thể loss\n\n" + table(["Loss", "Held-out accuracy", "Margin", "Mean chars"], [[n, r.get('eval_reward_accuracy'), r.get('eval_reward_gap'), r.get('mean_output_chars')] for n, r in variants.items()]))
        longest = max(variants, key=lambda n: variants[n]['mean_output_chars'])
        base_len = variants.get('dpo', {}).get('mean_output_chars')
        changed = max(variants, key=lambda n: abs(variants[n]['mean_output_chars'] - base_len)) if base_len is not None else longest
        parts.append(f"\n\nBiến thể dài nhất: **{longest}**; thay đổi lớn nhất so DPO cùng ngân sách: **{changed}**. RPO thêm NLL chosen, DPO-norm chuẩn hoá log-prob theo token, LD-DPO giảm trọng số phần vượt độ dài chung, ORPO kết hợp NLL và odds-ratio không dùng reference. Đây là các cơ chế có thể giải thích thay đổi độ dài, nhưng chưa chứng minh quan hệ nhân quả từ một lượt chạy. Không so margin tuyệt đối giữa các loss vì khác thang đo.\n")
    if grpo:
        n, before, after = grpo['n_test'], grpo['acc_before'], grpo['acc_after']
        parts.append("\n## 9. GRPO\n\n" + table(["Mục", "Giá trị"], [["n_test", n], ["Accuracy trước", before], ["Accuracy sau", after], ["stderr trước", math.sqrt(before * (1-before)/n)], ["stderr sau", math.sqrt(after * (1-after)/n)], ["Steps / G", f"{grpo['steps']} / {grpo['num_generations']}"]]))
        parts.append("\n\nCần đọc thành phần correctness và format trong output và grpo_history.json để xác định thành phần tăng trước. Reward tổng tăng không tự chứng minh giải đúng nhiều hơn. Test được tách trước huấn luyện, cùng tập được dùng ở trước và sau; stderr nhị thức chỉ là ước lượng nhiễu từng accuracy, kiểm định cặp cần kết quả từng câu.\n")
    parts.append("\n## Danh sách bonus thực tế\n\n")
    for label, done in (("NB3b — đủ 5 biến thể (+8)", completed('03b_dpo_variants', variants and len(variants) == 5)), ("NB5 — GGUF (+4)", completed('05_merge_deploy_gguf', deploy)), ("NB6 — benchmark (+6)", completed('06_benchmark', bench)), ("NB7 — GRPO (+8)", completed('07_grpo_bonus', grpo)), ("β-sweep (+6)", completed('beta', len(sweep) == 3)), ("Chấm chéo (+4)", completed('cross', judge.get('cross_judge'))), ("HF Hub (+3)", completed('hub', read('data/eval/hub_upload.json')))):
        parts.append(f"- [{'x' if done else ' '}] {label}\n")
    text = "\n".join(parts)
    report = REPO / "submission/REFLECTION.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    # Never overwrite a human-edited reflection on a resumed run.
    existing = report.read_text(encoding="utf-8") if report.exists() else ""
    if "<Họ Tên>" in existing or "Bản phân tích được tạo từ kết quả thực tế" in existing or not existing:
        report.write_text(text, encoding="utf-8")
    else:
        (report.parent / "REFLECTION.generated.md").write_text(text, encoding="utf-8")
    card = f"""---
base_model: {sft['base_model']}
language:
- vi
library_name: peft
tags:
- dpo
- experimental
---

# Lab22 SFT+DPO experimental adapter

Học viên: {name} ({cohort}). Mục tiêu: thực nghiệm căn chỉnh hội thoại tiếng Việt phục vụ học tập.
Adapter cần mô hình SFT riêng đã gộp ở `models/sft-merged`; không nạp trực tiếp trên base gốc.
SFT: {sft['dataset']}, {sft['n_train']} mẫu. Preference: {pref['dataset']}, {pref['n_train']} train / {pref['n_eval']} held-out.
β={dpo['beta']}, lr={dpo['lr']}, epoch={dpo['epochs']}, seed={dpo['seed']}.

Held-out reward accuracy: {fmt(dpo.get('eval_reward_accuracy'))}. NB4 win rate: {fmt(held.get('dpo_win_rate'))}, CI: {ci}.
Giám khảo: {judge['judge']}. Kết luận: {conclusion}.

Không dùng cho quyết định y tế/pháp lý, triển khai sản phẩm hoặc khẳng định an toàn đầy đủ.
Dữ liệu preference chưa ghi giấy phép rõ ràng; chỉ dùng cho học tập/nghiên cứu, không tự gán giấy phép cho trọng số dẫn xuất.
Giới hạn: mẫu nhỏ, một seed, thiên vị độ dài, giám khảo cùng Skywork với nguồn nhãn. Có thể sai kiến thức hoặc sinh nội dung không phù hợp.
Code hỗ trợ được tạo bằng AI và kiểm tra bằng CPU tests; kết quả GPU và phản tư được ghi từ lần chạy notebook thực tế.
"""
    (REPO / "submission/MODEL-CARD.md").write_text(card, encoding="utf-8")
    print("Đã tạo phản tư và model card từ số liệu thật; đọc lại trước khi nộp.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
