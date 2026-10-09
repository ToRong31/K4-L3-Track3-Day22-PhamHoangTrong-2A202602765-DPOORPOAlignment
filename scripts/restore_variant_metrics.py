"""Recover missing DPO/RPO rows by evaluating saved adapters, without training."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))
    paths = root / 'data/eval/storage_paths.json'
    if paths.exists():
        for key, value in json.loads(paths.read_text(encoding='utf-8')).items():
            if key in ('LAB22_MODELS_DIR', 'LAB22_GGUF_DIR'):
                os.environ[key] = value

    import unsloth
    import torch
    from datasets import Dataset
    from trl import DPOTrainer
    from lab22 import config as C, data as D, modeling as MD

    assert torch.cuda.is_available(), 'Choose T4 GPU.'
    summary_path = C.VARIANTS_DIR / 'variants_summary.json'
    results = json.loads(summary_path.read_text(encoding='utf-8'))
    # Keep a copy of the three already-completed rows before adding recovered ones.
    backup = summary_path.with_name('variants_summary.before_recovery.json')
    if not backup.exists():
        backup.write_text(json.dumps(results, indent=2), encoding='utf-8')
    eval_ds = Dataset.from_parquet(str(C.PREF_DIR / 'eval.parquet'))
    # Unsloth's DPOTrainer requires train_dataset even for evaluate-only usage.
    train_ds = Dataset.from_parquet(str(C.PREF_DIR / 'train.parquet'))
    train_ds = train_ds.select(range(min(C.TIER.variant_train, len(train_ds))))
    probes = [r['prompt'][0]['content'] for r in eval_ds.select(range(min(20, len(eval_ds))))]
    for name in ('dpo', 'rpo'):
        if name in results:
            continue
        adapter = C.VARIANTS_DIR / name
        assert (adapter / 'adapter_model.safetensors').exists(), f'Missing adapter: {adapter}'
        mismatch = D.split_mismatch(C.PREF_DIR, adapter)
        assert mismatch is None, mismatch
        print(f'Re-evaluating saved {name}; no training.', flush=True)
        model, tokenizer = MD.load_model(adapter)
        reference, _ = MD.load_model(C.SFT_MERGED)
        overrides = {'loss_weights': [1.0, 1.0]} if name == 'rpo' else {}
        trainer = DPOTrainer(
            model=model, ref_model=reference,
            args=MD.dpo_config(C.VARIANTS_DIR / f'{name}-recovery-eval',
                               loss_type=['sigmoid', 'sft'] if name == 'rpo' else ['sigmoid'],
                               eval_strategy='no', precompute_ref_log_probs=False, **overrides),
            train_dataset=train_ds, eval_dataset=eval_ds, processing_class=tokenizer,
        )
        ev = trainer.evaluate()
        outputs = MD.generate(trainer.model, tokenizer, probes, max_new_tokens=256)
        results[name] = {
            'eval_reward_accuracy': ev.get('eval_rewards/accuracies'),
            'eval_chosen_reward': ev.get('eval_rewards/chosen'),
            'eval_rejected_reward': ev.get('eval_rewards/rejected'),
            'eval_reward_gap': ev.get('eval_rewards/margins'),
            'mean_output_chars': sum(map(len, outputs)) / len(outputs),
            'evaluation_source': 'saved-adapter re-evaluation; explicit SFT reference',
        }
        assert all(results[name][key] is not None for key in ('eval_reward_accuracy', 'eval_reward_gap'))
        summary_path.write_text(json.dumps(results, indent=2), encoding='utf-8')
        print(name, results[name], flush=True)
        del trainer, model, reference
        MD.cleanup()
    expected = ('dpo', 'rpo', 'dpo_norm', 'ld_dpo', 'orpo')
    assert all(name in results for name in expected), 'Still missing variants.'
    results = {name: results[name] for name in expected}
    summary_path.write_text(json.dumps(results, indent=2), encoding='utf-8')
    import matplotlib.pyplot as plt
    import pandas as pd
    table = pd.DataFrame(results).T
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    table['eval_reward_accuracy'].astype(float).plot.bar(ax=axes[0], color='#2e548a')
    axes[0].set_ylim(0, 1)
    axes[0].set_title('held-out reward accuracy')
    table['mean_output_chars'].astype(float).plot.bar(ax=axes[1], color='#c83538')
    axes[1].set_title('mean output length (chars)')
    fig.tight_layout()
    fig.savefig(C.SCREENSHOTS / '03b-variants.png', dpi=120, bbox_inches='tight')
    (C.EVAL_DIR / 'variant_metrics_recovery.json').write_text(json.dumps({
        'method': 'evaluate saved adapters; no training', 'results': results,
    }, indent=2), encoding='utf-8')
    subprocess.run([sys.executable, 'scripts/run_colab.py', '--only', 'finish'], check=True)
    print('COMPLETE: all 5 variant rows and updated chart saved.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
