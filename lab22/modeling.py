"""GPU-side helpers shared by the notebooks: load, LoRA, generate, DPO, plots.

Import `unsloth` before `trl`/`transformers` in the notebook so Unsloth can
patch them; these helpers import lazily for the same reason.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path

from . import config as C


def load_model(name: str | Path, max_len: int = C.MAX_LEN, load_in_4bit: bool = True):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(name),
        max_seq_length=max_len,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )
    if not tokenizer.chat_template:
        from transformers import AutoTokenizer
        original = AutoTokenizer.from_pretrained(C.BASE_MODEL)
        if tokenizer.get_vocab() != original.get_vocab():
            raise ValueError("Missing chat template and tokenizer vocabulary differs from base; restore original tokenizer files.")
        if not original.chat_template:
            raise ValueError(f"Base tokenizer has no chat template: {C.BASE_MODEL}")
        tokenizer.chat_template = original.chat_template
    # Qwen3 ships a dedicated pad token. Reusing EOS as pad would mask the
    # end-of-turn token out of the loss and teach the model never to stop.
    if tokenizer.pad_token is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        if "<|vision_pad|>" not in tokenizer.get_vocab():
            raise ValueError(f"{name}: tokenizer needs a pad token distinct from EOS")
        tokenizer.pad_token = "<|vision_pad|>"
    return model, tokenizer


def add_lora(model, r: int = C.LORA_R, alpha: int = C.LORA_ALPHA):
    from unsloth import FastLanguageModel

    return FastLanguageModel.get_peft_model(
        model,
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=C.LORA_TARGETS,
        use_gradient_checkpointing="unsloth",
        random_state=C.SEED,
    )


def chat_text(tokenizer, messages: list[dict], add_generation_prompt: bool = True) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **C.CHAT_TEMPLATE_KWARGS,
    )


def generate(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = C.GEN_MAX_NEW_TOKENS,
    batch_size: int = 8,
) -> list[str]:
    """Greedy, batched generation for single-turn user prompts."""
    import torch
    from unsloth import FastLanguageModel

    FastLanguageModel.for_inference(model)
    batch_size = int(os.environ.get("GEN_BATCH_SIZE", str(batch_size)))
    if batch_size < 1:
        raise ValueError("GEN_BATCH_SIZE must be positive")
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    outputs: list[str] = []
    try:
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            texts = [chat_text(tokenizer, [{"role": "user", "content": p}]) for p in batch]
            # The template already contains the special tokens.
            enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            new_tokens = out[:, enc["input_ids"].shape[1] :]
            outputs.extend(t.strip() for t in tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
    finally:
        tokenizer.padding_side = old_side
    return outputs


def cleanup() -> None:
    """Release cached GPU memory. `del` the model/trainer names first."""
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def precision_flags() -> dict:
    import torch

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    return {"bf16": bf16, "fp16": torch.cuda.is_available() and not bf16}


def dpo_config(output_dir: str | Path, loss_type: list[str] | None = None, **overrides):
    """DPOConfig for LoRA-on-merged-SFT training.

    `precompute_ref_log_probs=True` scores every pair under the starting
    model before the first update. The policy starts as merged SFT plus a
    zero-initialised LoRA, so the reference is exactly the SFT model, with
    no dependence on how adapters are toggled during training.
    """
    from trl import DPOConfig

    params = dict(
        output_dir=str(output_dir),
        per_device_train_batch_size=C.TIER.dpo_batch,
        per_device_eval_batch_size=C.TIER.dpo_batch,
        gradient_accumulation_steps=C.TIER.dpo_grad_accum,
        num_train_epochs=C.DPO_EPOCHS,
        learning_rate=C.DPO_LR,
        beta=C.DPO_BETA,
        loss_type=loss_type or C.DPO_LOSS,
        max_length=C.MAX_LEN,
        precompute_ref_log_probs=True,
        warmup_steps=0.1,  # transformers v5: a float < 1 is a ratio; warmup_ratio is gone
        lr_scheduler_type="cosine",
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=25,
        save_strategy="no",
        optim="adamw_8bit",
        seed=C.SEED,
        report_to="none",
        **precision_flags(),
    )
    params.update(overrides)
    return DPOConfig(**params)


def reward_history(log_history: list[dict], prefix: str = ""):
    """Training (prefix="") or eval (prefix="eval_") reward rows as a DataFrame."""
    import pandas as pd

    key = f"{prefix}rewards/chosen"
    rows = [r for r in log_history if key in r]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.rename(columns=lambda c: c[len(prefix):] if prefix and c.startswith(prefix) else c)


def diagnose(df, window: int = 3) -> tuple[str, str]:
    """Classify the reward trajectory. Needs rewards/chosen and rewards/rejected.

    Implicit rewards start at 0 because the policy starts equal to the
    reference, so the sign of the final chosen reward is the direction it moved.
    """
    if df.empty or len(df) < 2:
        return "AMBIGUOUS", "Not enough logged steps to diagnose."
    end = df.tail(window)
    chosen = float(end["rewards/chosen"].mean())
    rejected = float(end["rewards/rejected"].mean())
    margin = chosen - rejected
    if margin <= 0:
        return "FAILURE", (
            f"Margin {margin:+.3f} <= 0: the model does not prefer chosen. Check data "
            "orientation, lr, and that the reference is the SFT model."
        )
    if chosen < 0:
        return "LIKELIHOOD DISPLACEMENT", (
            f"Margin {margin:+.3f} > 0 but chosen reward {chosen:+.3f} < 0: the gap grew "
            "because rejected fell faster. Compare with RPO in NB3b."
        )
    if chosen > 0 and rejected < 0:
        return "INTENDED", f"Chosen {chosen:+.3f} up, rejected {rejected:+.3f}, margin {margin:+.3f}."
    return "AMBIGUOUS", f"Margin {margin:+.3f}, chosen {chosen:+.3f}, rejected {rejected:+.3f}: not the intended opposite directions."


def plot_rewards(train_df, eval_df, title: str, path: Path | None = None):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for df, style, tag in ((train_df, "-", "train"), (eval_df, "o--", "held-out")):
        if df is None or df.empty:
            continue
        axes[0].plot(df["step"], df["rewards/chosen"], style, color="#2e548a", label=f"chosen ({tag})")
        axes[0].plot(df["step"], df["rewards/rejected"], style, color="#c83538", label=f"rejected ({tag})")
        margin = df["rewards/chosen"] - df["rewards/rejected"]
        axes[1].plot(df["step"], margin, style, color="#1a3355", label=f"margin ({tag})")
    for ax, ylabel in zip(axes, ("implicit reward  β·log(π/π_ref)", "margin (chosen − rejected)")):
        ax.axhline(0, color="#888", linestyle=":", linewidth=0.7)
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120, bbox_inches="tight")
    return fig
