"""Build a Kaggle continuation notebook with current lab source embedded."""
import json
from pathlib import Path
from build_colab import code, md, ready_payload, requirements

ROOT = Path(__file__).resolve().parent.parent


def render():
    specs = [s for s in requirements() if not s.startswith(('llama-cpp-python', 'lm-eval'))]
    cells = [
        md('# Lab22 — Chạy tiếp trên Kaggle\n\n'
           'Upload notebook này lên Kaggle. Chọn GPU và bật Internet. '
           'Thêm private Dataset có `Lab22_continue.zip`, gồm adapter trọng số và kết quả từ Drive. '
           'Hoặc điền HF_DATASET_ID để tải trực tiếp từ repo HF dataset riêng tư, thêm Secret HF_TOKEN trên Kaggle. '
           'Zip bài nộp không đủ vì không chứa trọng số adapter. '
           'Chạy một phần mỗi lần. Đây là bản chuẩn bị, chưa kiểm chứng GPU Kaggle.'),
        code('ONLY = "variants-recovery"  # hoặc "benchmark", "grpo", "beta", "gguf", "finish"\n'
             'HF_DATASET_ID = ""  # tài-khoản/lab22-continue; ưu tiên HF nếu điền\n'
             'INPUT_ZIP = ""  # để trống để tìm Lab22_continue.zip dưới /kaggle/input\n'),
        code('import os, sys, json, gzip, base64, zipfile, subprocess\n'
             'from pathlib import Path\n'
             'WORK = Path("/kaggle/working/lab22")\n'
             'WORK.mkdir(parents=True, exist_ok=True)\n'
             'if not (WORK / "adapters/sft-mini/adapter_model.safetensors").exists():\n'
             '    if HF_DATASET_ID:\n'
             '        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=True)\n'
             '        from huggingface_hub import snapshot_download\n'
             '        from kaggle_secrets import UserSecretsClient\n'
             '        hf_token = UserSecretsClient().get_secret("HF_TOKEN")\n'
             '        snapshot_download(repo_id=HF_DATASET_ID, repo_type="dataset", token=hf_token, local_dir=str(WORK))\n'
             '        del hf_token\n'
             '    else:\n'
             '        candidates = [Path(INPUT_ZIP)] if INPUT_ZIP else list(Path("/kaggle/input").rglob("Lab22_continue.zip"))\n'
             '        assert len(candidates) == 1, "Thêm Dataset chứa Lab22_continue.zip hoặc điền INPUT_ZIP"\n'
             '        with zipfile.ZipFile(candidates[0]) as z:\n'
             '            for name in z.namelist():\n'
             '                path = (WORK / name).resolve()\n'
             '                assert path == WORK or WORK in path.parents, "Đường dẫn zip không hợp lệ"\n'
             '            z.extractall(WORK)\n'
             f'payload = json.loads(gzip.decompress(base64.b64decode("{ready_payload()}")))\n'
             'for relative, contents in payload.items():\n'
             '    if relative.startswith("submission/"): continue\n'
             '    path = WORK / relative\n'
             '    path.parent.mkdir(parents=True, exist_ok=True)\n'
             '    path.write_text(contents, encoding="utf-8")\n'
             'assert (WORK / "adapters/sft-mini/adapter_model.safetensors").exists(), "Zip thiếu adapter SFT"\n'
             'os.chdir(WORK)\n'
             '# Dùng 1 GPU cho pipeline hiện tại, kể cả máy Kaggle có 2 GPU.\n'
             'os.environ.update(CUDA_VISIBLE_DEVICES="0", COMPUTE_TIER="T4", GEN_BATCH_SIZE="2", JUDGE_PROVIDER="rm", STUDENT_NAME="Phạm Hoàng Trọng", STUDENT_COHORT="A20-K4 · 2A202602765")\n'),
        code(f'subprocess.run([sys.executable, "-m", "pip", "install", "-q", *{specs!r}, "nbclient>=0.10,<1", "nbformat>=5.10,<6", "ipykernel>=6,<8"], check=True)\n'
             'subprocess.run([sys.executable, "-c", "import torch; assert torch.cuda.is_available(), \'Chọn GPU trong Settings\'; print(torch.cuda.get_device_name(0))"], check=True)\n'
             'subprocess.run([sys.executable, "-u", "scripts/local_model_storage.py", "--root", str(WORK), "--local", "/kaggle/temp/lab22-runtime"], check=True)\n'),
        code('if ONLY == "variants-recovery":\n'
             '    for name in ("dpo", "rpo"):\n'
             '        assert (WORK / "adapters/variants" / name / "adapter_model.safetensors").exists(), f"Zip thiếu adapter variants/{name}"\n'
             '    command = [sys.executable, "-u", "scripts/restore_variant_metrics.py", "--root", str(WORK)]\n'
             'else:\n'
             '    command = [sys.executable, "-u", "scripts/run_colab.py", "--only", ONLY]\n'
             'result = subprocess.run(command)\n'
             'print("Exit code:", result.returncode)\n'
             'status = WORK / "data/eval/run_status.json"\n'
             'if status.exists():\n'
             '    print({k:v["status"] for k,v in json.loads(status.read_text()).items()})\n'),
        code('from IPython.display import FileLink, display\n'
             'sys.path.insert(0, str(WORK / "scripts"))\n'
             'from run_colab import export_submission\n'
             'archive = export_submission()\n'
             'display(FileLink(str(archive)))\n'
             'print("Tải zip về, giải nén vào repo rồi commit/push. Muốn chạy phiên mới, giữ cả adapter bằng cách tải continuation zip dưới đây.")\n'
             'continue_zip = Path("/kaggle/working/Lab22_continue.zip")\n'
             'with zipfile.ZipFile(continue_zip, "w", zipfile.ZIP_DEFLATED) as z:\n'
             '    for folder in ("adapters", "data", "notebooks", "submission"):\n'
             '        for p in (WORK / folder).rglob("*"):\n'
             '            if p.is_file() and p.suffix in (".safetensors", ".json", ".jsonl", ".parquet", ".ipynb", ".md", ".txt", ".png", ".jpg"):\n'
             '                z.write(p, p.relative_to(WORK).as_posix())\n'
             'display(FileLink(str(continue_zip)))\n'),
        md('Trước khi dừng, tải zip và lưu phiên notebook/output theo chức năng Save Version của Kaggle. '
           'Mô hình lớn trong /kaggle/temp không phải bằng chứng nộp bài; phiên mới khôi phục từ adapter. '
           'Đánh giá đang dở chưa resume ở cấp câu hỏi; benchmark giữ các lượt hoàn thành có cấu hình khớp. '
           'Giữ Dataset riêng tư vì chứa adapter/dữ liệu thực nghiệm.'),
    ]
    return {'cells': cells, 'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}, 'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 5}


if __name__ == '__main__':
    folder = ROOT / 'kaggle'
    folder.mkdir(exist_ok=True)
    (folder / 'Lab22_Kaggle_Continue.ipynb').write_text(json.dumps(render(), ensure_ascii=False, indent=1), encoding='utf-8')
    print('Created kaggle/Lab22_Kaggle_Continue.ipynb')
