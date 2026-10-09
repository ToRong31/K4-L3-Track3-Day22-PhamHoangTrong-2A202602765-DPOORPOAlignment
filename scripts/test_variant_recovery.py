"""The evaluation-only recovery must satisfy Unsloth's required dataset API."""
import ast
from pathlib import Path


def test_recovery_supplies_training_dataset_but_never_trains():
    path = Path(__file__).resolve().parent / 'restore_variant_metrics.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    trainers = [n for n in calls if isinstance(n.func, ast.Name) and n.func.id == 'DPOTrainer']
    assert len(trainers) == 1
    assert {'train_dataset', 'eval_dataset', 'ref_model'} <= {k.arg for k in trainers[0].keywords}
    assert not any(isinstance(n.func, ast.Attribute) and n.func.attr == 'train' for n in calls)
