"""Keep merged SFT and GGUF on Colab's temporary disk, preserving Drive evidence.

Also supports migrating an older notebook workspace without re-running training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def validate_model(folder):
    from safetensors import safe_open
    cfg = folder / 'config.json'
    if not cfg.exists():
        raise RuntimeError(f'Missing config: {folder}')
    index = folder / 'model.safetensors.index.json'
    if index.exists():
        mapping = json.loads(index.read_text(encoding='utf-8'))['weight_map']
    else:
        mapping = None
    files = sorted({folder / x for x in mapping.values()}) if mapping else sorted(folder.glob('*.safetensors'))
    if not files:
        raise RuntimeError(f'No model weights: {folder}')
    for path in files:
        with safe_open(str(path), framework='pt', device='cpu') as f:
            keys = set(f.keys())
            if not keys:
                raise RuntimeError(f'Empty tensors: {path}')
            if mapping and keys != {k for k, v in mapping.items() if v == path.name}:
                raise RuntimeError(f'Index mismatch: {path}')
    return files


def valid(folder):
    try:
        validate_model(folder)
        return True
    except Exception:
        return False


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def patch_workspace(root, merged):
    # These replacements allow use with the user's already-running old notebook.
    config = root / 'lab22/config.py'
    text = config.read_text(encoding='utf-8')
    text = text.replace('MODELS = REPO_ROOT / "models"', 'MODELS = Path(_env("LAB22_MODELS_DIR", str(REPO_ROOT / "models")))')
    text = text.replace('GGUF_DIR = REPO_ROOT / "gguf"', 'GGUF_DIR = Path(_env("LAB22_GGUF_DIR", str(REPO_ROOT / "gguf")))')
    config.write_text(text, encoding='utf-8')
    notebook = root / 'notebooks/05_merge_deploy_gguf.py'
    text = notebook.read_text(encoding='utf-8')
    text = text.replace('C.REPO_ROOT.glob("gguf*/**/*.gguf")', 'C.GGUF_DIR.rglob("*.gguf")')
    text = text.replace('gguf_path.relative_to(C.REPO_ROOT)', 'gguf_path')
    notebook.write_text(text, encoding='utf-8')
    verifier = root / 'scripts/verify.py'
    text = verifier.read_text(encoding='utf-8')
    text = text.replace('import json\n', 'import json\nimport os\n') if '\nimport os\n' not in text else text
    text = text.replace('(REPO / "models" / "sft-merged").resolve()', '(Path(os.environ.get("LAB22_MODELS_DIR", str(REPO / "models"))) / "sft-merged").resolve()')
    text = text.replace('{rel(expected)}', '{expected}')
    text = text.replace('need(REPO / "models" / "sft-merged" / "config.json",', 'need(Path(os.environ.get("LAB22_MODELS_DIR", str(REPO / "models"))) / "sft-merged" / "config.json",')
    # need() can report paths outside the repo when running with temporary weights.
    text = text.replace('return str(path.relative_to(REPO))', 'return os.path.relpath(path, REPO)')
    verifier.write_text(text, encoding='utf-8')
    runner = root / 'scripts/run_colab.py'
    text = runner.read_text(encoding='utf-8')
    if 'storage_file = REPO /' not in text:
        text = text.replace('    args = parser.parse_args()\n',
            '    args = parser.parse_args()\n'
            '    storage_file = REPO / "data/eval/storage_paths.json"\n'
            '    if storage_file.exists():\n'
            '        for key, value in json.loads(storage_file.read_text(encoding="utf-8")).items():\n'
            '            if key in ("LAB22_MODELS_DIR", "LAB22_GGUF_DIR"):\n'
            '                os.environ[key] = value\n')
    text = text.replace("folder_path=REPO / 'models/sft-merged'", "folder_path=Path(os.environ.get('LAB22_MODELS_DIR', str(REPO / 'models'))) / 'sft-merged'")
    runner.write_text(text, encoding='utf-8')
    old = root / 'models/sft-merged'
    for cfg in (root / 'adapters').rglob('adapter_config.json'):
        data = json.loads(cfg.read_text(encoding='utf-8'))
        base = str(data.get('base_model_name_or_path', ''))
        if base in (str(old), str(merged), 'models/sft-merged'):
            data['base_model_name_or_path'] = str(merged)
            cfg.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--local', default='/content/lab22-runtime')
    parser.add_argument('--remove-drive-copy', action='store_true')
    parser.add_argument('--build', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    root, local = Path(args.root).resolve(), Path(args.local).resolve()
    drive_model = root / 'models/sft-merged'
    merged = local / 'models/sft-merged'
    if root == local or root in local.parents or local in root.parents:
        raise RuntimeError('Temporary model storage must be outside the Drive workspace.')
    if args.build:
        import unsloth
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(model_name=str(root / 'adapters/sft-mini'), max_seq_length=768, dtype=None, load_in_4bit=True)
        model.save_pretrained_merged(str(merged), tokenizer, save_method='merged_16bit')
        validate_model(merged)
        return 0
    if not valid(merged):
        # Prefer the original when healthy; otherwise reuse the previously verified repair.
        candidates = (drive_model, Path('/content/lab22-sft-repair'))
        source = next((p for p in candidates if valid(p)), None)
        if source:
            print('Copying verified SFT to temporary disk:', source, flush=True)
            shutil.copytree(source, merged, dirs_exist_ok=True)
            for file in validate_model(source):
                if digest(file) != digest(merged / file.name):
                    raise RuntimeError('Temporary model copy differs; Drive files were not removed.')
        else:
            print('Reconstructing SFT from its saved adapter; no training.', flush=True)
            subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), '--root', str(root), '--local', str(local), '--build'], check=True)
    validate_model(merged)
    patch_workspace(root, merged)
    # Save paths for future pipeline invocations in this workspace (not secrets).
    storage = {'LAB22_MODELS_DIR': str(local / 'models'), 'LAB22_GGUF_DIR': str(local / 'gguf')}
    (root / 'data/eval').mkdir(parents=True, exist_ok=True)
    (root / 'data/eval/storage_paths.json').write_text(json.dumps(storage, indent=2), encoding='utf-8')
    if args.remove_drive_copy and drive_model.exists():
        # Delete only the specifically named, verified-copy model folder.
        if drive_model.is_symlink() or drive_model.resolve() != root / 'models/sft-merged':
            raise RuntimeError('Unexpected model path; refusing deletion.')
        for file in drive_model.glob('*.safetensors'):
            if file.stat().st_size > 0 and (not (merged / file.name).exists() or digest(file) != digest(merged / file.name)):
                raise RuntimeError('Original nonempty shard differs; refusing deletion.')
        shutil.rmtree(drive_model)
        print('Removed only models/sft-merged from Drive after verifying the local copy.', flush=True)
    print('READY: model on temporary Colab disk; adapters and evidence remain on Drive.', flush=True)
    print('Paths:', storage, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
