"""Publish reusable SFT+DPO adapters/model card and a private Kaggle transfer repo.

Credentials come only from HF_TOKEN environment variable. Run beside Drive data.
"""
import argparse
import json
import os
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--name', default='lab22-dpo-experimental')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    from huggingface_hub import HfApi
    token = os.environ.get('HF_TOKEN')
    if not token:
        raise RuntimeError('Set HF_TOKEN using Colab Secrets, not notebook source.')
    for name in ('sft-mini', 'dpo'):
        for file in ('adapter_model.safetensors', 'adapter_config.json'):
            p = root / 'adapters' / name / file
            assert p.exists() and p.stat().st_size, f'Missing/empty {p}'
    api = HfApi(token=token)
    username = api.whoami()['name']
    model_id = f'{username}/{args.name}'
    transfer_id = f'{username}/lab22-continue'
    staging = Path('/tmp/lab22-hf-publish')
    staging.mkdir(parents=True, exist_ok=True)
    # Only copy needed adapter/tokenizer files. No datasets, cached references or keys.
    for name, destination in (('dpo', staging), ('sft-mini', staging / 'sft_adapter')):
        destination.mkdir(exist_ok=True)
        for p in (root / 'adapters' / name).iterdir():
            if p.is_file() and (p.suffix in ('.json', '.safetensors') or p.name == 'tokenizer.model'):
                shutil.copy2(p, destination / p.name)
    cfg = json.loads((staging / 'adapter_config.json').read_text(encoding='utf-8'))
    cfg['base_model_name_or_path'] = './sft-merged'
    (staging / 'adapter_config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    source_dir = Path(__file__).resolve().parent
    shutil.copy2(source_dir / 'reuse_hf_adapter.py', staging / 'reuse_hf_adapter.py')
    shutil.copy2(root / 'requirements.txt', staging / 'requirements.txt')
    metrics_dir = staging / 'evaluation'
    metrics_dir.mkdir(exist_ok=True)
    for relative in ('adapters/dpo/dpo_metrics.json', 'data/eval/judge_summary.json', 'data/eval/sft_metrics.json'):
        source = root / relative
        if source.exists():
            shutil.copy2(source, metrics_dir / source.name)
    card = (root / 'submission/MODEL-CARD.md').read_text(encoding='utf-8')
    card += f'''

## Reuse from Hugging Face

Model repo: https://huggingface.co/{model_id}

The root LoRA is DPO. `sft_adapter/` contains the preceding SFT adapter.
The DPO reference is **merged SFT**, not the original base model. Therefore
loading the root adapter directly on the base model is incorrect. The bundled
helper reconstructs SFT locally, updates a local copy of the DPO config, and
loads the final model. CUDA GPU and sufficient temporary disk are required.

```python
from huggingface_hub import hf_hub_download
import subprocess, sys
helper = hf_hub_download("{model_id}", "reuse_hf_adapter.py")
subprocess.run([sys.executable, helper, "--repo", "{model_id}",
                "--prompt", "Giải thích quicksort trong 5 câu."], check=True)
```

Install the training packages in `requirements.txt` before running the helper.
The helper does no new training; it merges the saved SFT LoRA before applying DPO.
Reference reconstruction/reuse is provided as code, not a tested deployment guarantee.
Known output defects: stray tool_call markers and imperfect instruction following.
The model card and evaluation files report the actual lab run, not benchmark claims.
Do not infer a redistribution license for preference data or derivative weights.
'''
    (staging / 'README.md').write_text(card, encoding='utf-8')
    api.create_repo(repo_id=model_id, repo_type='model', private=False, exist_ok=True)
    api.upload_folder(repo_id=model_id, repo_type='model', folder_path=str(staging),
                      commit_message='Publish saved SFT/DPO adapters, model card and reuse helper')
    published = set(api.list_repo_files(model_id, repo_type='model'))
    for file in ('adapter_model.safetensors', 'adapter_config.json', 'sft_adapter/adapter_model.safetensors', 'README.md', 'reuse_hf_adapter.py'):
        assert file in published, f'Upload verification failed: {file}'
    print('Model uploaded:', f'https://huggingface.co/{model_id}', flush=True)
    # Separate private transfer dataset: preserve all adapters and evaluation evidence.
    api.create_repo(repo_id=transfer_id, repo_type='dataset', private=True, exist_ok=True)
    info = api.repo_info(transfer_id, repo_type='dataset')
    if not info.private:
        raise RuntimeError('Transfer dataset is public; set it private before uploading data.')
    for folder in ('adapters', 'data', 'notebooks', 'submission'):
        path = root / folder
        if path.exists():
            api.upload_folder(repo_id=transfer_id, repo_type='dataset', folder_path=str(path), path_in_repo=folder,
                allow_patterns=['*.safetensors', '*.json', '*.jsonl', '*.parquet', '*.ipynb', '*.md', '*.txt', '*.png', '*.jpg'],
                ignore_patterns=['**/.ipynb_checkpoints/**', '**/ref/**', '**/*-checkpoints/**'],
                commit_message=f'Transfer Lab22 {folder} for Kaggle continuation')
    expected = {'adapters/sft-mini/adapter_model.safetensors', 'adapters/dpo/adapter_model.safetensors', 'data/pref/eval.parquet'}
    assert expected <= set(api.list_repo_files(transfer_id, repo_type='dataset')), 'Transfer incomplete.'
    evidence = {'adapter': model_id, 'reference': 'reconstruct from sft_adapter/ using reuse_hf_adapter.py',
                'model_url': f'https://huggingface.co/{model_id}', 'transfer_dataset': transfer_id,
                'upload_verified': True, 'reuse_gpu_tested': False}
    (root / 'data/eval/hub_upload.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    reflection = root / 'submission/REFLECTION.md'
    text = reflection.read_text(encoding='utf-8')
    text = text.replace('- [ ] HF Hub (+3)', '- [x] HF Hub (+3)')
    note = f'\n\n## HF Hub — adapter và model card\n\nĐã upload và kiểm tra các file trên [{model_id}](https://huggingface.co/{model_id}). Repo chứa adapter DPO, adapter SFT, model card (mô hình gốc, dữ liệu, siêu tham số, kết quả thật) và script khôi phục SFT để tái sử dụng. Repo dataset {transfer_id} là gói riêng tư phục vụ chuyển sang Kaggle, không thay thế model card. Chưa kiểm thử suy luận GPU từ repo HF sau upload.\n'
    if '## HF Hub — adapter và model card' not in text:
        text += note
    reflection.write_text(text, encoding='utf-8')
    print('Kaggle HF_DATASET_ID =', transfer_id, flush=True)
    print('Upload complete; download hub_upload.json and updated REFLECTION.md for GitHub submission.', flush=True)


if __name__ == '__main__':
    main()
