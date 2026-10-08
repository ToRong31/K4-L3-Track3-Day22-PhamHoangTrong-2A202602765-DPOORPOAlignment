# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # NB2 — Dữ liệu sở thích (preference) tiếng Việt
#
# **Bộ dữ liệu mặc định:** `sailor2/sea-ultrafeedback-onpolicy`, lọc `language == "Vietnamese"`
# (khoảng 4.1k cặp). Lab cũ huấn luyện DPO trên UltraFeedback tiếng Anh trong khi SFT và
# đánh giá đều bằng tiếng Việt, nên khó đọc hiệu ứng của DPO.
#
# > **Mục tiêu:** đưa dữ liệu về dạng hội thoại của TRL
# > (`prompt`/`chosen`/`rejected` là list message), lọc cặp vượt `MAX_LEN`,
# > chia **tập huấn luyện/eval không trùng câu hỏi**, đo thiên vị độ dài, lưu Parquet.
# >
# > **Giấy phép:** bộ dữ liệu không ghi giấy phép. Nguồn gốc là UltraFeedback (câu hỏi) và
# > phản hồi do mô hình sinh rồi được chấm, nên chỉ dùng cho học tập và nghiên cứu.
# > Đặt `PREF_DATASET=argilla/ultrafeedback-binarized-preferences-cleaned PREF_LANGUAGE=`
# > để chạy lại bản tiếng Anh làm đối chứng.

# %%
import sys
from pathlib import Path

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "lab22" / "config.py").exists())
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer

from lab22 import config as C
from lab22 import data as D

C.ensure_dirs()
print(C.summary())

# Only the tokenizer is needed here: chat template + length budget. No GPU.
tokenizer = AutoTokenizer.from_pretrained(C.BASE_MODEL)

# %% [markdown]
# ## 1. Nạp, lọc, chia huấn luyện/eval theo câu hỏi

# %%
train_ds, eval_ds = D.load_preference_pairs(
    C.PREF_DATASET,
    tokenizer,
    max_len=C.MAX_LEN,
    n_train=C.PREF_TRAIN,
    n_eval=C.PREF_EVAL,
    language=C.PREF_LANGUAGE or None,
    seed=C.SEED,
    template_kwargs=C.CHAT_TEMPLATE_KWARGS,
)
D.assert_disjoint(list(train_ds), list(eval_ds))
assert len(train_ds) == C.PREF_TRAIN and len(eval_ds) == C.PREF_EVAL, (
    "Không đủ số cặp sau lọc. Kiểm tra dataset/MAX_LEN trước khi huấn luyện; không dùng dữ liệu thiếu để nhận đủ điểm."
)
assert len({D.normalize_prompt(r['prompt'][0]['content']) for r in eval_ds}) >= C.JUDGE_PROMPTS, (
    "Held-out chưa đủ prompt khác nhau để chấm NB4; tăng PREF_EVAL rồi chạy lại NB2–NB4."
)
print(f"train={len(train_ds)}  eval={len(eval_ds)}  (no prompt overlap)")
for i in range(min(3, len(train_ds))):
    print(f"\nCặp mẫu {i + 1}: {train_ds[i]}")

# %% [markdown]
# ## 2. Thiên vị độ dài
#
# Nếu phần lớn `chosen` dài hơn `rejected`, DPO có thể học "viết dài hơn" thay vì
# "trả lời tốt hơn". Ghi con số này vào REFLECTION và so với độ dài đầu ra ở NB4.

# %%
def n_tokens(text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


stats = D.length_stats(list(train_ds), count=n_tokens)
print(f"chosen median {stats['chosen_median']:.0f} tok · rejected median {stats['rejected_median']:.0f} tok")
print(f"chosen longer in {stats['chosen_longer_frac']:.1%} of pairs")

# %%
import matplotlib.pyplot as plt
import numpy as np

chosen = np.array([n_tokens(r["chosen"][0]["content"]) for r in train_ds])
rejected = np.array([n_tokens(r["rejected"][0]["content"]) for r in train_ds])
fig, ax = plt.subplots(figsize=(8, 3.5))
bins = np.linspace(0, C.MAX_LEN, 40)
ax.hist(chosen, bins=bins, alpha=0.6, label="chosen", color="#2e548a")
ax.hist(rejected, bins=bins, alpha=0.6, label="rejected", color="#c83538")
ax.set_xlabel("response tokens")
ax.set_title(f"Length: chosen longer in {stats['chosen_longer_frac']:.0%} of pairs")
ax.legend()
fig.savefig(C.SCREENSHOTS / "02b-pref-length.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. Lưu

# %%
import json

train_ds.to_parquet(str(C.PREF_DIR / "train.parquet"))
eval_ds.to_parquet(str(C.PREF_DIR / "eval.parquet"))
(C.PREF_DIR / "stats.json").write_text(
    json.dumps({"dataset": C.PREF_DATASET, "language": C.PREF_LANGUAGE,
                "n_train": len(train_ds), "n_eval": len(eval_ds), **stats}, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(f"Saved {len(train_ds)} train / {len(eval_ds)} eval pairs → {C.PREF_DIR}")

# %% [markdown]
# ## 4. Ghi chú vibe-coding
#
# Mở 5 cặp ngẫu nhiên và tự chấm: bạn có đồng ý với nhãn `chosen` không? Khoảng 1%
# câu `chosen` trong bộ này lẫn tiếng Anh hoặc mất dấu, một số câu hỏi là bài code.
# Nếu bạn lọc thêm (ví dụ bỏ cặp lệch độ dài > 2×), ghi lại số cặp còn lại và lý do
# vào REFLECTION. `BONUS-CHALLENGE.md` #1 gợi ý tự xây dữ liệu sở thích (preference) tiếng Việt gốc.
#
# **Tiếp theo:** NB3 — huấn luyện DPO.
