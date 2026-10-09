import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('limit,expected_calls', [(200, 0), (100, 1)])
def test_resume_only_uses_completed_matching_evaluation(tmp_path, limit, expected_calls):
    source = Path(__file__).resolve().parent.parent / 'notebooks/06_benchmark.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    functions = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)], type_ignores=[])
    folder = tmp_path / 'lm_eval/sft-ifeval'
    folder.mkdir(parents=True)
    result = {'config': {'model_args': 'pretrained=/tmp/model,dtype=float16,enable_thinking=False',
                         'limit': limit, 'apply_chat_template': True},
              'configs': {'ifeval': {'num_fewshot': 0}},
              'results': {'ifeval': {'prompt_level_strict_acc,none': 0.5}}}
    (folder / 'results-old.json').write_text(json.dumps(result))
    calls = []
    def run(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0, stderr='')
    ns = dict(json=json, Path=Path, C=SimpleNamespace(SFT_MERGED='/tmp/model', EVAL_DIR=tmp_path, SEED=42),
              DTYPE='float16', BATCH='1', BENCHMARKS={'IFEval': ('ifeval', 0, 200, 'prompt_level_strict_acc,none')},
              subprocess=SimpleNamespace(run=run))
    exec(compile(functions, str(source), 'exec'), ns)
    ns['run_lm_eval']('sft', 'ifeval', 0, 200)
    assert len(calls) == expected_calls
