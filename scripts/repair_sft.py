"""Rebuild a damaged SFT shard from its saved LoRA, without retraining.

Run in a separate GPU process. Validate reconstruction against every readable
original tensor before publishing only the damaged shards to the same path.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--staging', default='/content/lab22-sft-repair')
    args = parser.parse_args()
    root, staging = Path(args.root).resolve(), Path(args.staging).resolve()
    adapter, target = root / 'adapters/sft-mini', root / 'models/sft-merged'
    if not (adapter / 'adapter_model.safetensors').exists():
        raise RuntimeError('Missing saved SFT adapter; cannot reconstruct without training.')
    if staging == target or staging in target.parents or target in staging.parents:
        raise RuntimeError('Staging must be separate from the original model.')

    import unsloth  # Must patch transformers first, in a fresh process.
    import torch
    from safetensors import safe_open
    from unsloth import FastLanguageModel

    assert torch.cuda.is_available(), 'Choose a T4 GPU before repair.'
    index_path = target / 'model.safetensors.index.json'
    index = json.loads(index_path.read_text(encoding='utf-8'))
    shards = sorted(set(index['weight_map'].values()))
    damaged = []
    for name in shards:
        try:
            with safe_open(str(target / name), framework='pt', device='cpu') as f:
                assert list(f.keys()), f'No tensors in {name}'
        except Exception:
            damaged.append(name)
    if not damaged:
        print('All referenced SFT shards are readable; no repair needed.', flush=True)
        return 0
    if len(damaged) == len(shards):
        raise RuntimeError('No readable original shard to verify reconstruction; refusing to replace the reference blindly.')
    print('Damaged shards:', damaged, flush=True)
    staging.mkdir(parents=True, exist_ok=True)
    if list(staging.glob('*.safetensors')):
        raise RuntimeError('Staging contains a previous reconstruction. Use a fresh --staging directory.')
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter), max_seq_length=768, dtype=None, load_in_4bit=True,
    )
    model.save_pretrained_merged(str(staging), tokenizer, save_method='merged_16bit')
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    rebuilt = json.loads((staging / 'model.safetensors.index.json').read_text(encoding='utf-8'))
    if rebuilt['weight_map'] != index['weight_map']:
        raise RuntimeError('Rebuilt shard layout differs from the original; keep originals and inspect before replacing.')
    for name in shards:
        with safe_open(str(staging / name), framework='pt', device='cpu') as fresh:
            keys = set(fresh.keys())
            expected = {key for key, shard in index['weight_map'].items() if shard == name}
            if keys != expected:
                raise RuntimeError(f'Rebuilt shard has unexpected tensor keys: {name}')
            if name not in damaged:
                with safe_open(str(target / name), framework='pt', device='cpu') as original:
                    for key in fresh.keys():
                        old, new = original.get_slice(key), fresh.get_slice(key)
                        if old.get_shape() != new.get_shape() or old.get_dtype() != new.get_dtype():
                            raise RuntimeError(f'Reconstruction differs from original: {key}')
                        # Compare small row chunks to avoid allocating whole embedding tensors.
                        shape = old.get_shape()
                        if not shape:
                            equal = torch.equal(original.get_tensor(key), fresh.get_tensor(key))
                            if not equal:
                                raise RuntimeError(f'Reconstruction differs from original: {key}')
                        else:
                            for start in range(0, shape[0], 256):
                                if not torch.equal(old[start:start+256], new[start:start+256]):
                                    raise RuntimeError(f'Reconstruction differs from original: {key}')
        print('Verified rebuilt shard:', name, flush=True)

    # Publish only damaged shards. Readable original weights and all existing
    # adapters, evaluation results, and notebook outputs stay at their paths.
    for name in damaged:
        source = staging / name
        temp = target / (name + '.repairing')
        shutil.copyfile(source, temp)
        if source.stat().st_size != temp.stat().st_size or digest(source) != digest(temp):
            raise RuntimeError(f'Drive copy incomplete: {name}; original was not replaced. Check Drive quota.')
        with safe_open(str(temp), framework='pt', device='cpu') as f:
            list(f.keys())
        os.replace(temp, target / name)
        print('Repaired:', name, flush=True)
    print('SFT reference repaired without retraining. Run each failed bonus separately.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
