"""Execute saved notebooks in fresh kernels; save evidence after every stage."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CORE = ['00_dpo_loss_from_scratch', '01_sft_mini', '02_preference_data', '03_dpo_train', '04_compare_and_eval']
BONUS = {'variants': '03b_dpo_variants', 'gguf': '05_merge_deploy_gguf', 'benchmark': '06_benchmark', 'grpo': '07_grpo_bonus'}
STATE = REPO / 'data/eval/run_status.json'
ARTIFACTS = {
    CORE[0]: 'notebooks/00_dpo_loss_from_scratch.ipynb',
    CORE[1]: 'models/sft-merged/config.json', CORE[2]: 'data/pref/eval.parquet',
    CORE[3]: 'adapters/dpo/dpo_metrics.json', CORE[4]: 'data/eval/judge_summary.json',
    BONUS['variants']: 'submission/screenshots/03b-variants.png',
    BONUS['gguf']: 'data/eval/deploy_meta.json',
    BONUS['benchmark']: 'data/eval/benchmark_results.json',
    BONUS['grpo']: 'adapters/grpo/grpo_metrics.json',
}


def command(args):
    print('Đang chạy:', ' '.join(map(str, args)), flush=True)
    subprocess.run([str(x) for x in args], cwd=REPO, check=True)


def export_submission():
    """Include only known evidence/source types; never include model weights/secrets."""
    out = REPO / 'Lab22_submission.zip'
    files = set()
    for folder in ('lab22', 'scripts', 'notebooks', 'colab', 'docs'):
        for suffix in ('*.py', '*.md', '*.ipynb'):
            files.update((REPO / folder).rglob(suffix))
    for folder in ('submission', 'data/eval', 'data/pref', 'adapters'):
        for suffix in ('*.json', '*.jsonl', '*.md', '*.png', '*.parquet'):
            files.update((REPO / folder).rglob(suffix))
    files.update(p for p in REPO.glob('*') if p.name in ('README.md', 'rubric.md', 'requirements.txt', 'requirements-biggpu.txt', 'Makefile', '.gitignore', 'LICENSE', 'HARDWARE-GUIDE.md', 'BONUS-CHALLENGE.md', 'COLAB-RUN.md'))
    temp = out.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temp, 'w', zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(files):
            if '__pycache__' not in p.parts and '.ipynb_checkpoints' not in p.parts:
                archive.write(p, p.relative_to(REPO).as_posix())
    temp.replace(out)
    print(f'Đã lưu gói nộp: {out} ({out.stat().st_size / 1e6:.1f} MB)', flush=True)
    return out


def execute(stem):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    import nbformat
    from nbclient import NotebookClient
    from build_colab import percent_cells

    cells = percent_cells(REPO / 'notebooks' / f'{stem}.py')
    notebook = nbformat.v4.new_notebook(cells=[
        (nbformat.v4.new_code_cell if c['cell_type'] == 'code' else nbformat.v4.new_markdown_cell)(''.join(c['source']))
        for c in cells
    ], metadata={'kernelspec': {'display_name': 'Python 3', 'name': 'python3', 'language': 'python'}})
    output = REPO / 'notebooks' / f'{stem}.ipynb'

    def saved(cell, cell_index, **kwargs):
        nbformat.write(notebook, output)
        print(f'[{stem}] cell {cell_index + 1}/{len(notebook.cells)} đã lưu', flush=True)

    class StreamingClient(NotebookClient):
        def process_message(self, msg, cell, cell_index):
            content = msg.get('content', {})
            if msg.get('msg_type') == 'stream':
                print(content.get('text', ''), end='', flush=True)
            elif msg.get('msg_type') == 'error':
                print('\n'.join(content.get('traceback', [])), flush=True)
            return super().process_message(msg, cell, cell_index)

    client = StreamingClient(notebook, timeout=None, kernel_name='python3',
                            resources={'metadata': {'path': str(REPO)}}, on_cell_executed=saved)
    try:
        client.execute()
    finally:
        nbformat.write(notebook, output)


def cross_judge():
    # NB4 RM results are retained. Judge exactly the saved answers, no regeneration.
    sys.path.insert(0, str(REPO))
    from lab22 import judge as J
    from lab22 import config as C
    provider = os.environ.get('CROSS_JUDGE_PROVIDER', '')
    model_id = os.environ.get('CROSS_JUDGE_MODEL', '')
    if not provider or not model_id or not J.has_judge_key(provider):
        raise RuntimeError('Chấm chéo cần provider, model ID và API key trong Colab Secrets.')
    path = C.EVAL_DIR / 'side_by_side.jsonl'
    records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    caller = J.make_caller(provider, model_id)
    rows = [{**r, **J.judge_pair(r['prompt'], r['sft'], r['dpo'], caller)} for r in records]
    name = f'{provider}:{model_id}'
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    result = {'judge': name, 'outputs_sha256': sha, 'records': rows}
    (C.EVAL_DIR / 'judge_results_api.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    summary_path = C.EVAL_DIR / 'judge_summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    rm = json.loads((C.EVAL_DIR / 'judge_results_rm.json').read_text(encoding='utf-8'))
    assert rm['outputs_sha256'] == sha == summary['outputs_sha256'], 'Chấm chéo khác tập đầu ra'
    summary['cross_judge'] = {'other_judge': name, **J.agreement(rm['records'], rows)}
    summary['api_summary'] = {c: J.summarize([r for r in rows if r['category'] == c], seed=C.SEED) for c in ('heldout', 'helpfulness', 'safety')}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')


def upload_hub():
    from huggingface_hub import HfApi
    repo_id = os.environ.get('HF_REPO_ID', '')
    if not repo_id or '/' not in repo_id or not os.environ.get('HF_TOKEN'):
        raise RuntimeError('HF Hub cần HF_REPO_ID=tài-khoản/tên-experimental và HF_TOKEN trong Secrets.')
    api = HfApi()
    reference_id = repo_id + '-sft-reference'
    api.create_repo(repo_id=reference_id, exist_ok=True)
    api.upload_folder(repo_id=reference_id, folder_path=Path(os.environ.get('LAB22_MODELS_DIR', str(REPO / 'models'))) / 'sft-merged')
    # Publish a copy of the config; the local config must continue pointing to the local SFT.
    staging = REPO / 'hub-export'
    staging.mkdir(exist_ok=True)
    import shutil
    for p in (REPO / 'adapters/dpo').iterdir():
        if p.is_file():
            shutil.copy2(p, staging / p.name)
    cfg = json.loads((staging / 'adapter_config.json').read_text(encoding='utf-8'))
    cfg['base_model_name_or_path'] = reference_id
    (staging / 'adapter_config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    card = (REPO / 'submission/MODEL-CARD.md').read_text(encoding='utf-8')
    import re
    card = re.sub(r'^base_model: .*$', f'base_model: {reference_id}', card, count=1, flags=re.MULTILINE)
    card += f'\n\nHF merged SFT reference: {reference_id}. Nạp adapter từ {repo_id}; PEFT tự nạp reference này.\n'
    (staging / 'README.md').write_text(card, encoding='utf-8')
    api.create_repo(repo_id=repo_id, exist_ok=True)
    api.upload_folder(repo_id=repo_id, folder_path=staging)
    (REPO / 'data/eval/hub_upload.json').write_text(json.dumps({'adapter': repo_id, 'reference': reference_id}, indent=2), encoding='utf-8')


def signature(stem):
    sources = sorted((REPO / 'lab22').glob('*.py')) + [REPO / 'notebooks' / f'{name}.py' for name in CORE]
    if stem not in CORE:
        sources.append(REPO / 'notebooks' / f'{stem}.py')
    settings = {k: os.environ.get(k, '') for k in ('COMPUTE_TIER', 'BASE_MODEL', 'MAX_LEN', 'SFT_DATASET', 'SFT_SLICE', 'PREF_DATASET', 'PREF_LANGUAGE', 'PREF_TRAIN', 'PREF_EVAL', 'DPO_BETA', 'DPO_LR', 'DPO_EPOCHS', 'DPO_LOSS', 'LORA_R', 'LORA_ALPHA', 'JUDGE_PROMPTS', 'JUDGE_RM_MODELS', 'GEN_MAX_NEW_TOKENS', 'SEED', 'VARIANTS', 'ORPO_FROM_BASE', 'GRPO_DATASET', 'BENCH_BATCH_SIZE')}
    return hashlib.sha256(b''.join(p.read_bytes() for p in sources) + json.dumps(settings, sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bonuses', default='variants,gguf,benchmark,grpo,beta')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--only', choices=['core', *BONUS, 'beta', 'cross', 'hub', 'finish'])
    args = parser.parse_args()
    storage_file = REPO / 'data/eval/storage_paths.json'
    if storage_file.exists():
        for key, value in json.loads(storage_file.read_text(encoding='utf-8')).items():
            if key in ('LAB22_MODELS_DIR', 'LAB22_GGUF_DIR'):
                os.environ[key] = value
    ARTIFACTS[CORE[1]] = str(Path(os.environ.get('LAB22_MODELS_DIR', str(REPO / 'models'))) / 'sft-merged/config.json')
    selected = [s.strip() for s in args.bonuses.split(',') if s.strip()]
    unknown = set(selected) - set(BONUS) - {'beta', 'cross', 'hub'}
    if unknown:
        parser.error(f'Unknown bonuses: {sorted(unknown)}')
    os.environ.setdefault('JUDGE_PROVIDER', 'rm')
    STATE.parent.mkdir(parents=True, exist_ok=True)
    statuses = json.loads(STATE.read_text(encoding='utf-8')) if STATE.exists() else {}

    def stage(name, action, required=False, sig=None):
        if args.resume and sig and statuses.get(name, {}).get('status') == 'success' and statuses[name].get('signature') == sig and (REPO / ARTIFACTS[name]).exists():
            print(f'Giữ kết quả đã hoàn thành: {name}', flush=True)
            return True
        print(f'\n=== BẮT ĐẦU {name} ===', flush=True)
        entry = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(), 'signature': sig}
        statuses[name] = entry
        STATE.write_text(json.dumps(statuses, ensure_ascii=False, indent=2), encoding='utf-8')
        success = False
        try:
            action()
            entry['status'] = 'success'
            success = True
        except Exception as exc:
            entry.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            print(f'{name} chưa hoàn thành: {exc}', flush=True)
            if required:
                raise
        finally:
            entry['ended_utc'] = datetime.now(timezone.utc).isoformat()
            STATE.write_text(json.dumps(statuses, ensure_ascii=False, indent=2), encoding='utf-8')
            export_submission()
        return success

    if args.only in (None, 'core'):
        for index, stem in enumerate(CORE):
            sig = signature(stem)
            previous = statuses.get(stem, {})
            skip = args.resume and previous.get('status') == 'success' and previous.get('signature') == sig and (REPO / ARTIFACTS[stem]).exists()
            if not skip:
                # A rebuilt reference/split invalidates downstream evidence even with the same seed.
                downstream = set(CORE[index + 1:]) | set(BONUS.values()) | {'beta', 'cross', 'hub'}
                for name in downstream:
                    if name in statuses:
                        statuses[name]['status'] = 'stale'
            stage(stem, lambda s=stem: execute(s), required=True, sig=sig)
        command([sys.executable, 'scripts/finish_submission.py'])
        command([sys.executable, 'scripts/verify.py'])
        export_submission()
    if args.only == 'core':
        return 0
    for name in selected if args.only is None else [args.only]:
        if name in BONUS:
            def run_bonus(n=name):
                if n == 'gguf':
                    command([sys.executable, '-m', 'pip', 'install', 'llama-cpp-python>=0.3.16,<1.0'])
                elif n == 'benchmark':
                    command([sys.executable, '-m', 'pip', 'install', 'lm-eval[ifeval,math]>=0.4.13,<0.5'])
                execute(BONUS[n])
            stage(BONUS[name], run_bonus, sig=signature(BONUS[name]))
        elif name == 'beta':
            def sweep():
                for beta in (0.05, 0.1, 0.5):
                    command([sys.executable, 'scripts/train_dpo.py', '--beta', str(beta), '--output-dir', f'adapters/dpo-b{beta:.2f}'])
                    export_submission()
                command([sys.executable, 'scripts/eval_judge.py', '--plot-sweep'])
            stage('beta', sweep)
        elif name == 'cross':
            stage('cross', cross_judge)
        elif name == 'hub':
            command([sys.executable, 'scripts/finish_submission.py'])
            stage('hub', upload_hub)
    command([sys.executable, 'scripts/finish_submission.py'])
    verification = subprocess.run([sys.executable, 'scripts/verify.py'], cwd=REPO)
    export_submission()
    failed = [k for k, v in statuses.items() if v['status'] != 'success']
    print(f'Bonus/chặng chưa hoàn thành: {failed or "không"}', flush=True)
    return verification.returncode


if __name__ == '__main__':
    raise SystemExit(main())
