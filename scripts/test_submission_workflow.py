"""Check evidence packaging, source bundling, and generated report without GPU."""
import ast
import base64
import gzip
import json
import re
import sys
import subprocess
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_colab as B
import run_colab as R
import finish_submission as F


def test_ready_bundle_is_self_contained_and_core_first():
    payload = json.loads(gzip.decompress(base64.b64decode(B.ready_payload())))
    for path in ('scripts/run_colab.py', 'scripts/verify.py', 'scripts/finish_submission.py', 'notebooks/00_dpo_loss_from_scratch.py', 'lab22/config.py'):
        assert path in payload
    assert '.env' not in payload
    assert 'return None' not in payload['notebooks/00_dpo_loss_from_scratch.py']
    assert B.STAGES[4][0] == '04_compare_and_eval'
    nb = B.render_ready()
    assert len(nb['cells']) == 7
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            ast.parse(''.join(cell['source']))
    target = B.REPO / 'colab/Lab22_READY_ALL.ipynb'
    assert json.loads(target.read_text(encoding='utf-8')) == nb
    canonical = B.REPO / 'colab/Lab22_DPO_T4.ipynb'
    assert json.loads(canonical.read_text(encoding='utf-8')) == nb


def test_export_includes_evidence_and_excludes_secrets_and_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(R, 'REPO', tmp_path)
    for path in ('notebooks/nb.ipynb', 'submission/REFLECTION.md', 'data/pref/stats.json', 'data/pref/train.parquet', 'adapters/dpo/dpo_metrics.json', 'models/sft-merged/model.safetensors', 'adapters/dpo/adapter_model.safetensors', '.env', 'data/eval/random.token'):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('fixture', encoding='utf-8')
    out = R.export_submission()
    with zipfile.ZipFile(out) as archive:
        names = set(archive.namelist())
        assert 'notebooks/nb.ipynb' in names
        assert 'data/pref/stats.json' in names
        assert 'data/pref/train.parquet' in names
        assert all(not n.endswith(('.token', '.safetensors')) for n in names)
        assert '.env' not in names


def test_builder_runs_in_freshly_extracted_colab_payload(tmp_path):
    payload = json.loads(gzip.decompress(base64.b64decode(B.ready_payload())))
    for relative, contents in payload.items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(contents, encoding='utf-8')
    assert not (tmp_path / 'colab').exists()
    result = subprocess.run([sys.executable, 'scripts/build_colab.py'], cwd=tmp_path, capture_output=True, text=True, encoding='utf-8')
    assert result.returncode == 0, result.stderr
    assert (tmp_path / 'colab/Lab22_DPO_T4.ipynb').exists()


def test_finish_does_not_create_reflection_without_real_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(F, 'REPO', tmp_path)
    assert F.main() == 1
    assert not (tmp_path / 'submission/REFLECTION.md').exists()


def test_generated_report_uses_fixture_numbers_and_meets_word_limits(tmp_path, monkeypatch):
    monkeypatch.setattr(F, 'REPO', tmp_path)
    from lab22 import judge as J
    def put(path, value):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
    put('data/eval/sft_metrics.json', dict(gpu='test GPU', vram_gb=16, base_model='test/base', dataset='test/sft', n_train=1000, timestamp_utc='2026-10-08T00:00:00Z'))
    put('data/pref/stats.json', dict(dataset='test/pref', n_train=800, n_eval=100, chosen_longer_frac=0.61))
    put('adapters/dpo/dpo_metrics.json', dict(compute_tier='T4', beta=0.1, lr=5e-6, epochs=1, max_len=768, seed=42, diagnosis='INTENDED', eval_reward_gap=0.2345, eval_reward_accuracy=0.65))
    records = [dict(id='e1', category=c, prompt='fixture prompt', sft='fixture SFT answer', dpo='fixture DPO answer', winner='tie') for c in ('heldout', 'helpfulness', 'safety')]
    summary = {c: J.summarize([r for r in records if r['category'] == c]) for c in ('heldout', 'helpfulness', 'safety')}
    summary.update(judge='rm:test', overall=J.summarize(records), sanity_accuracy=1)
    put('data/eval/judge_summary.json', summary)
    (tmp_path / 'data/eval/side_by_side.jsonl').write_text('\n'.join(json.dumps(r) for r in records), encoding='utf-8')
    assert F.main() == 0
    text = (tmp_path / 'submission/REFLECTION.md').read_text(encoding='utf-8')
    assert '0.2345' in text and 'fixture DPO answer' in text
    assert 'chưa đủ bằng chứng' in text
    for section, minimum in ((3, 100), (6, 150)):
        body = text.split(f'## {section}.', 1)[1].split('\n## ', 1)[0]
        assert len(body.split()) >= minimum
    assert '_Trả lời ở đây._' not in text
    # A later generation must preserve a human's revision.
    report = tmp_path / 'submission/REFLECTION.md'
    report.write_text('Human edited reflection', encoding='utf-8')
    assert F.main() == 0
    assert report.read_text(encoding='utf-8') == 'Human edited reflection'
    assert (tmp_path / 'submission/REFLECTION.generated.md').exists()


def test_failed_notebook_keeps_actual_output(tmp_path, monkeypatch):
    pytest.importorskip('nbclient')
    nbformat = pytest.importorskip('nbformat')
    monkeypatch.setattr(R, 'REPO', tmp_path)
    folder = tmp_path / 'notebooks'
    folder.mkdir()
    (folder / 'failure.py').write_text('# %%\nprint("actual output before error")\n# %%\nraise RuntimeError("intentional fixture failure")\n', encoding='utf-8')
    with pytest.raises(Exception, match='intentional fixture failure'):
        R.execute('failure')
    nb = nbformat.read(folder / 'failure.ipynb', as_version=4)
    assert nb.cells[0].outputs[0].text.strip() == 'actual output before error'
    assert nb.cells[1].outputs[0].output_type == 'error'
