"""Reconstruct SFT from its saved adapter, then load DPO. Requires CUDA.

python reuse_hf_adapter.py --repo account/lab22-dpo-experimental --prompt "Xin chào"
"""
import argparse
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True)
    parser.add_argument('--work', default='./lab22-hf-runtime')
    parser.add_argument('--prompt', default='Giải thích thuật toán quicksort trong 5 câu.')
    args = parser.parse_args()
    import unsloth
    import torch
    from huggingface_hub import snapshot_download
    from unsloth import FastLanguageModel
    assert torch.cuda.is_available(), 'A CUDA GPU is required.'
    work = Path(args.work).resolve()
    snapshot = Path(snapshot_download(args.repo))
    sft = work / 'sft-merged'
    dpo = work / 'dpo-adapter'
    work.mkdir(parents=True, exist_ok=True)
    if not (sft / 'config.json').exists():
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(snapshot / 'sft_adapter'), max_seq_length=768, load_in_4bit=True,
        )
        if not tokenizer.chat_template:
            from transformers import AutoTokenizer
            config = json.loads((snapshot / 'sft_adapter/adapter_config.json').read_text())
            original = AutoTokenizer.from_pretrained(config['base_model_name_or_path'])
            assert tokenizer.get_vocab() == original.get_vocab(), 'Tokenizer mismatch'
            assert original.chat_template, 'Missing base chat template'
            tokenizer.chat_template = original.chat_template
        model.save_pretrained_merged(str(sft), tokenizer, save_method='merged_16bit')
        del model, tokenizer
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    dpo.mkdir(exist_ok=True)
    for p in snapshot.iterdir():
        if p.is_file() and p.suffix in ('.json', '.safetensors', '.jinja', '.model'):
            shutil.copy2(p, dpo / p.name)
    cfg = json.loads((dpo / 'adapter_config.json').read_text(encoding='utf-8'))
    cfg['base_model_name_or_path'] = str(sft)
    (dpo / 'adapter_config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=str(dpo), max_seq_length=768, load_in_4bit=True)
    if not tokenizer.chat_template:
        from transformers import AutoTokenizer
        original = AutoTokenizer.from_pretrained(str(sft))
        assert tokenizer.get_vocab() == original.get_vocab(), 'Tokenizer mismatch'
        assert original.chat_template, 'Missing SFT chat template'
        tokenizer.chat_template = original.chat_template
    FastLanguageModel.for_inference(model)
    prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': args.prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=384, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    print(tokenizer.decode(output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True))


if __name__ == '__main__':
    main()
