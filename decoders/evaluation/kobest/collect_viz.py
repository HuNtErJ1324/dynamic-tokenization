"""Collect plain + entropy-split tokenization for one fixed sentence across all models."""
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

SENTENCE = "심혈관 질환의 예방을 위해 규칙적인 운동이 필요합니다."

# Models to run full (forward pass for entropy): small/medium only
MODELS_FULL = [
    "mistralai/Mistral-7B-v0.1",
    "EleutherAI/polyglot-ko-1.3b",
    "skt/ko-gpt-trinity-1.2B-v0.5",
]
# Models to run tokenizer-only (no forward pass, just plain tokens)
MODELS_TOK_ONLY = [
    "EleutherAI/polyglot-ko-5.8b",
    "kfkas/Llama-2-ko-7b-Chat",
    "beomi/Llama-3-KoEn-8B",
    "Qwen/Qwen2.5-7B",
]

THRESHOLD = 3.0

def get_entropy_splits(model, tokenizer, text, threshold, device):
    ids = tokenizer.encode(text, return_tensors="pt").to(device)
    seq = ids[0].tolist()
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0]
    log_p = torch.log_softmax(logits.float(), dim=-1)
    entropy = (-(torch.exp(log_p) * log_p).sum(-1)).tolist()

    orig_tokens = [tokenizer.decode([t]) for t in seq]
    split_log = []
    new_ids = []
    for pos, tid in enumerate(seq):
        ent = entropy[pos]
        tok_str = tokenizer.decode([tid])
        if ent > threshold:
            # char-level split
            stripped = tok_str.strip()
            has_space = tok_str.startswith(' ')
            if len(stripped) > 1:
                chars = list(stripped)
                char_ids = []
                for i, ch in enumerate(chars):
                    piece = (' ' if (i == 0 and has_space) else '') + ch
                    try:
                        char_ids.extend(tokenizer.encode(piece, add_special_tokens=False))
                    except:
                        char_ids.append(tid)
                if len(char_ids) > 1:
                    new_ids.extend(char_ids)
                    split_log.append({
                        "token": tok_str,
                        "entropy": round(ent, 3),
                        "split_into": [tokenizer.decode([i]) for i in char_ids],
                    })
                    continue
        new_ids.append(tid)

    split_tokens = [tokenizer.decode([t]) for t in new_ids]
    return orig_tokens, split_tokens, split_log

results = {}
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Full run (plain + entropy split)
for model_name in MODELS_FULL:
    print(f"\n>>> {model_name}", flush=True)
    try:
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16, trust_remote_code=True
        ).to(device)
        model.eval()
        orig, split, log = get_entropy_splits(model, tok, SENTENCE, THRESHOLD, device)
        results[model_name] = {
            "original_tokens": orig, "n_orig": len(orig),
            "split_tokens": split, "n_split": len(split),
            "splits_applied": log,
        }
        print(f"  orig={len(orig)} split={len(split)} splits_applied={len(log)}", flush=True)
        del model
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)
        results[model_name] = {"error": str(e)}

# Tokenizer-only (plain tokens only, no entropy)
for model_name in MODELS_TOK_ONLY:
    print(f"\n>>> {model_name} (tokenizer only)", flush=True)
    try:
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        ids = tok.encode(SENTENCE, add_special_tokens=True)
        tokens = [tok.decode([i]) for i in ids]
        results[model_name] = {
            "original_tokens": tokens, "n_orig": len(tokens),
            "split_tokens": None, "n_split": None,
            "splits_applied": [],
            "note": "tokenizer-only (no entropy split — model too large for local MPS)"
        }
        print(f"  orig={len(tokens)} (no split)", flush=True)
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)
        results[model_name] = {"error": str(e)}

out = "/Users/hanbyulkang/Desktop/cse481m/extracted/dynamic-tokenization-main/results/viz_all_models.json"
json.dump(results, open(out, "w"), ensure_ascii=False, indent=2)
print(f"\nSaved: {out}")
