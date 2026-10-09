"""Create CPU-only Colab publishing notebook; no tokens embedded."""
import json
from pathlib import Path
from build_colab import md, code

ROOT = Path(__file__).resolve().parent.parent


def render():
    cells = [md('# Lab22 — Publish reusable adapters to HF + transfer to Kaggle\n\n'
                'CPU runtime is enough. Add a new write token to Colab Secrets as HF_TOKEN and enable notebook access. '
                'The notebook creates a **public Model** repo with SFT/DPO adapters and a model card, '
                'plus a **private Dataset** for Kaggle transfer. No merged model upload is needed. '
                'Use the Drive folder containing actual adapter weights, not the submission zip.'),
             code('DRIVE_FOLDER = "Lab22_PhamHoangTrong_2A202602765"\nMODEL_NAME = "lab22-dpo-experimental"\n'),
             code('from google.colab import drive, userdata\nfrom pathlib import Path\nimport os, sys, subprocess\n'
                  'drive.mount("/content/drive")\nWORK = Path("/content/drive/MyDrive") / DRIVE_FOLDER\n'
                  'assert (WORK / "adapters/sft-mini/adapter_model.safetensors").exists(), "Không thấy adapter SFT"\n'
                  'os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")\n'
                  'subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=True)\n'),
             code('local_scripts = Path("/content/lab22-hf-tools")\nlocal_scripts.mkdir(exist_ok=True)\n' +
                  ''.join(f'(local_scripts / {name!r}).write_text({(ROOT / "scripts" / name).read_text(encoding="utf-8")!r}, encoding="utf-8")\n' for name in ('publish_hf.py', 'reuse_hf_adapter.py')) +
                  'subprocess.run([sys.executable, "-u", str(local_scripts / "publish_hf.py"), "--root", str(WORK), "--name", MODEL_NAME], check=True)\n'),
             code('from google.colab import files\nimport json\n'
                  'evidence = json.loads((WORK / "data/eval/hub_upload.json").read_text())\n'
                  'print("Model:", evidence["model_url"])\nprint("Kaggle HF_DATASET_ID:", evidence["transfer_dataset"])\n'
                  'files.download(str(WORK / "data/eval/hub_upload.json"))\n'
                  'files.download(str(WORK / "submission/REFLECTION.md"))\n')]
    return dict(cells=cells, metadata={'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}, 'language_info': {'name': 'python'}, 'colab': {'provenance': []}}, nbformat=4, nbformat_minor=5)


if __name__ == '__main__':
    path = ROOT / 'colab/Lab22_HF_Publish.ipynb'
    path.write_text(json.dumps(render(), ensure_ascii=False, indent=1), encoding='utf-8')
    print('Created', path.name)
