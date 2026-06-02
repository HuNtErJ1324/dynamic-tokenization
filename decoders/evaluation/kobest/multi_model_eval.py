"""
Multi-model KoBEST evaluation with entropy-guided splitting.

Supports any CausalLM decoder model. Runs:
  - plain: original tokenization
  - entropy_split: split high-entropy tokens back to char level

Usage:
  python multi_model_eval.py --model mistralai/Mistral-7B-v0.1 --task copa --max_examples 50
"""

import argparse
import json
import time
import random
import sys
import os
from pathlib import Path
from collections import defaultdict
from typing import List, Optional, Tuple

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

RESULTS_DIR = Path(__file__).resolve().parents[3] / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Universal char-level split ─────────────────────────────────────────────────

def split_token_to_char_ids(token_id: int, tokenizer) -> Optional[List[int]]:
    """
    Decode a single token to text, split into Unicode chars, re-encode each.
    Returns list of token IDs (possibly > 1) or None if can't split further.
    Works for GPT-2 byte-level, SentencePiece, and tiktoken-style tokenizers.
    """
    try:
        text = tokenizer.decode([token_id])
    except Exception:
        return None

    stripped = text.strip()
    if len(stripped) <= 1:
        return None

    has_space = text.startswith(' ')
    chars = list(stripped)
    all_ids = []

    for idx, ch in enumerate(chars):
        piece = (' ' if (idx == 0 and has_space) else '') + ch
        try:
            ids = tokenizer.encode(piece, add_special_tokens=False)
            all_ids.extend(ids)
        except Exception:
            all_ids.append(token_id)

    if len(all_ids) <= 1:
        return None
    # Verify the decoded result is the same text
    try:
        check = tokenizer.decode(all_ids)
        if check.strip() != stripped:
            # fallback: just split by individual char encoding
            pass
    except Exception:
        pass

    return all_ids


# ── Entropy calculation ────────────────────────────────────────────────────────

def compute_per_token_entropy(
    model, input_ids: torch.Tensor, device
) -> torch.Tensor:
    """Forward pass → per-token next-token entropy. Shape: (seq_len,)"""
    with torch.no_grad():
        out = model(input_ids=input_ids)
        logits = out.logits[0]          # (seq_len, vocab)
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        entropy = -(probs * log_probs).sum(dim=-1)  # (seq_len,)
    return entropy


# ── KoBEST COPA helpers ────────────────────────────────────────────────────────

CHOICE_LABELS = ["A", "B", "C", "D"]
KMMLU_SUBJECT = "Korean-History"

KMMLU_ALL_SUBJECTS = [
    "Accounting","Agricultural-Sciences","Aviation-Engineering-and-Maintenance",
    "Biology","Chemical-Engineering","Chemistry","Civil-Engineering",
    "Computer-Science","Construction","Criminal-Law","Ecology","Economics",
    "Education","Electrical-Engineering","Electronics-Engineering",
    "Energy-Management","Environmental-Science","Fashion","Food-Processing",
    "Gas-Technology-and-Engineering","Geomatics","Health","Industrial-Engineer",
    "Information-Technology","Interior-Architecture-and-Design","Law",
    "Machine-Design-and-Manufacturing","Management","Maritime-Engineering",
    "Marketing","Materials-Engineering","Mechanical-Engineering",
    "Nondestructive-Testing","Patent","Political-Science-and-Sociology",
    "Psychology","Public-Safety","Railway-and-Automotive-Engineering",
    "Real-Estate","Refrigerating-Machinery","Social-Welfare","Taxation",
    "Telecommunications-and-Wireless-Technology","Korean-History","Math",
]


def format_kmmlu(row, answer=None, score_by_text=False):
    q = row["question"]
    choices = [row["A"], row["B"], row["C"], row["D"]]
    gold = row["answer"] - 1  # 1-indexed → 0-indexed

    if score_by_text:
        prompt = q + "\n정답:"
        return prompt, gold, choices

    prompt = (f"{q}\nA. {row['A']}\nB. {row['B']}\nC. {row['C']}\nD. {row['D']}\nAnswer:")
    if answer is not None:
        prompt += f" {answer}"
    return prompt, gold, [row["A"], row["B"], row["C"], row["D"]]


def load_kmmlu(subject=KMMLU_SUBJECT, max_examples=None, seed=42):
    from datasets import concatenate_datasets
    subjects = KMMLU_ALL_SUBJECTS if subject == "all" else [subject]
    test_parts, dev_parts = [], []
    for s in subjects:
        ds = load_dataset("HAERAE-HUB/KMMLU", s)
        test_parts.append(ds["test"])
        dev_parts.append(ds["dev"])
    test = concatenate_datasets(test_parts)
    dev = concatenate_datasets(dev_parts)
    if max_examples:
        test = test.select(range(min(max_examples, len(test))))
    return test, dev

def format_copa(row, answer=None, korean_prompt=False, score_by_text=False):
    premise = row["premise"]
    q = row["question"]
    c1 = row["alternative_1"]
    c2 = row["alternative_2"]
    gold = row["label"]

    if score_by_text:
        # For base models: prompt ends before the blank, choices are the actual texts
        if korean_prompt:
            connective = "왜냐하면" if q == "cause" else "그래서"
            prompt = f"{premise} {connective}"
        else:
            connective = "because" if q == "cause" else "therefore"
            prompt = f"{premise} {connective.capitalize()},"
        return prompt, gold, [c1, c2]

    if korean_prompt:
        connective = "왜냐하면" if q == "cause" else "그 결과"
        prompt = f"{premise} {connective} _____\nA. {c1}\nB. {c2}\n정답:"
    else:
        connective = "because" if q == "cause" else "therefore"
        prompt = f"{premise} {connective.capitalize()}, _____\nA. {c1}\nB. {c2}\nAnswer:"
    if answer is not None:
        prompt += f" {answer}"
    return prompt, gold, [c1, c2]

KOBEST_TASKS = ["copa", "boolq", "wic", "hellaswag", "sentineg"]

def format_kobest(task, row, score_by_text=False):
    """Universal formatter for all KoBEST tasks. Returns (prompt, gold_idx, choices)."""
    if task == "copa":
        return None  # handled separately
    elif task == "boolq":
        prompt = f"{row['paragraph']}\n질문: {row['question']}\n정답:"
        choices = ["예", "아니오"]
        return prompt, row["label"], choices
    elif task == "wic":
        prompt = (f"단어 '{row['word']}'이 두 문장에서 같은 의미로 쓰였나요?\n"
                  f"문장1: {row['context_1']}\n문장2: {row['context_2']}\n정답:")
        choices = ["예", "아니오"]
        return prompt, row["label"], choices
    elif task == "hellaswag":
        endings = [row["ending_1"], row["ending_2"], row["ending_3"], row.get("ending_4", "")]
        choices = [e for e in endings if e]
        prompt = f"{row['context']}\n다음으로 가장 자연스러운 것은?"
        return prompt, row["label"], choices
    elif task == "sentineg":
        prompt = f"다음 문장의 감정은?\n{row['sentence']}\n정답:"
        choices = ["부정", "긍정"]
        return prompt, row["label"], choices
    raise ValueError(f"Unknown task: {task}")

def load_kobest_task(task, max_examples=None, seed=42):
    ds = load_dataset("skt/kobest_v1", task)
    test = ds["test"]
    val = ds["validation"]
    if max_examples:
        test = test.select(range(min(max_examples, len(test))))
    return test, val

def load_copa(num_shots=5, max_examples=None, seed=42):
    ds = load_dataset("skt/kobest_v1", "copa")
    test = ds["test"]
    val = ds["validation"]
    random.seed(seed)
    if max_examples:
        indices = list(range(min(max_examples, len(test))))
        test = test.select(indices)
    return test, val

def build_few_shot_prefix(val_ds, num_shots=5, seed=42, korean_prompt=False, score_by_text=False):
    random.seed(seed)
    indices = random.sample(range(len(val_ds)), min(num_shots, len(val_ds)))
    prefix = ""
    for i in indices:
        row = val_ds[i]
        prompt, gold, choice_texts = format_copa(row, korean_prompt=korean_prompt, score_by_text=score_by_text)
        if score_by_text:
            # Show full answer text in few-shot
            full = prompt + " " + choice_texts[gold]
        else:
            answer = CHOICE_LABELS[gold]
            full, _, _ = format_copa(row, answer=answer, korean_prompt=korean_prompt)
        prefix += full + "\n\n"
    return prefix


# ── Scoring: compare log-probs for each choice ────────────────────────────────

def score_choices_plain(model, tokenizer, prompt: str, choices: List[str], device) -> int:
    """Return the index of the highest-scoring choice (continuation likelihood)."""
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_len = prompt_ids.shape[1]
    scores = []
    for ch in choices:
        full = prompt + " " + ch
        full_ids = tokenizer.encode(full, return_tensors="pt").to(device)
        # Score only the choice tokens, not the prompt
        with torch.no_grad():
            out = model(input_ids=full_ids)
            logits = out.logits[0]  # (seq_len, vocab)
            choice_len = full_ids.shape[1] - prompt_len
            if choice_len <= 0:
                scores.append(-999.0)
                continue
            # Log-prob of choice tokens
            log_probs = torch.log_softmax(logits, dim=-1)
            choice_ids = full_ids[0, prompt_len:]
            choice_log_prob = sum(
                log_probs[prompt_len - 1 + j, choice_ids[j]].item()
                for j in range(len(choice_ids))
            )
            scores.append(choice_log_prob / len(choice_ids))
    return int(np.argmax(scores))


def score_choices_entropy_split(
    model, tokenizer, prompt: str, choices: List[str],
    device, threshold: float, visualize: bool = False
) -> Tuple[int, Optional[dict]]:
    """
    1. Tokenize prompt
    2. Forward pass → entropy per token
    3. Split high-entropy tokens
    4. Re-score with split tokens
    """
    scores = []
    viz = None

    for ch_idx, ch in enumerate(choices):
        full = prompt + " " + ch
        input_ids = tokenizer.encode(full, return_tensors="pt").to(device)
        seq_ids = input_ids[0].tolist()

        entropy = compute_per_token_entropy(model, input_ids, device)

        # Split high-entropy tokens
        new_ids = []
        split_log = [] if (visualize and ch_idx == 0) else None

        for pos, tid in enumerate(seq_ids):
            ent = entropy[pos].item()
            if ent > threshold:
                splits = split_token_to_char_ids(tid, tokenizer)
                if splits and len(splits) > 1:
                    new_ids.extend(splits)
                    if split_log is not None:
                        orig_tok = tokenizer.decode([tid])
                        split_toks = tokenizer.decode(splits)
                        split_log.append({
                            "pos": pos,
                            "token": orig_tok,
                            "entropy": round(ent, 3),
                            "split_into": [tokenizer.decode([s]) for s in splits],
                        })
                else:
                    new_ids.append(tid)
            else:
                new_ids.append(tid)

        if visualize and ch_idx == 0 and split_log is not None:
            orig_decoded = [tokenizer.decode([t]) for t in seq_ids]
            new_decoded = [tokenizer.decode([t]) for t in new_ids]
            viz = {
                "original_tokens": orig_decoded,
                "split_tokens": new_decoded,
                "splits_applied": split_log,
                "n_orig": len(seq_ids),
                "n_split": len(new_ids),
            }

        new_ids_tensor = torch.tensor([new_ids], device=device)
        with torch.no_grad():
            out = model(input_ids=new_ids_tensor, labels=new_ids_tensor)
        scores.append(-out.loss.item())

    return int(np.argmax(scores)), viz


# ── Main evaluation ────────────────────────────────────────────────────────────

def evaluate(
    model_name: str,
    exp_type: str,
    task: str = "copa",
    max_examples: int = None,
    threshold: float = 3.0,
    num_shots: int = 5,
    device_str: str = "auto",
    seed: int = 42,
    korean_prompt: bool = False,
    score_by_text: bool = False,
    kmmlu_subject: str = "all",
):
    print(f"\n{'='*60}", flush=True)
    print(f"Model: {model_name}", flush=True)
    print(f"Exp type: {exp_type} | task: {task} | max: {max_examples} | thr: {threshold}", flush=True)
    print(f"{'='*60}", flush=True)

    if device_str == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = torch.device(device_str)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if device.type in ("cuda", "mps") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, trust_remote_code=True
    ).to(device)
    model.eval()

    if task == "kmmlu":
        test_ds, val_ds = load_kmmlu(subject=kmmlu_subject, max_examples=max_examples, seed=seed)
        prefix = ""
        random.seed(seed)
        for row in list(val_ds)[:num_shots]:
            p, g, ch = format_kmmlu(row, score_by_text=score_by_text)
            if score_by_text:
                prefix += p + " " + ch[g] + "\n\n"
            else:
                full, _, _ = format_kmmlu(row, answer=CHOICE_LABELS[g])
                prefix += full + "\n\n"
    elif task in KOBEST_TASKS and task != "copa":
        test_ds, val_ds = load_kobest_task(task, max_examples=max_examples, seed=seed)
        prefix = ""
        random.seed(seed)
        for row in list(val_ds)[:num_shots]:
            p, g, ch = format_kobest(task, row)
            prefix += p + " " + ch[g] + "\n\n"
    else:
        test_ds, val_ds = load_copa(num_shots=num_shots, max_examples=max_examples, seed=seed)
        prefix = build_few_shot_prefix(val_ds, num_shots=num_shots, seed=seed, korean_prompt=korean_prompt, score_by_text=score_by_text)

    correct = 0
    total = 0
    viz_examples = []

    t0 = time.time()
    for i, row in enumerate(test_ds):
        if task == "kmmlu":
            prompt, gold, choice_texts = format_kmmlu(row, score_by_text=score_by_text)
            full_prompt = prefix + prompt
            choices = choice_texts if score_by_text else ["A", "B", "C", "D"]
        elif task in KOBEST_TASKS and task != "copa":
            prompt, gold, choice_texts = format_kobest(task, row)
            full_prompt = prefix + prompt
            choices = choice_texts
        else:
            prompt, gold, choice_texts = format_copa(row, korean_prompt=korean_prompt, score_by_text=score_by_text)
            full_prompt = prefix + prompt
            choices = choice_texts if score_by_text else ["A", "B"]
        gold_idx = gold

        if exp_type == "plain":
            pred = score_choices_plain(model, tokenizer, full_prompt, choices, device)
            if i < 2:
                ids = tokenizer.encode(full_prompt + " " + choices[0], add_special_tokens=True)
                toks = [tokenizer.decode([t]) for t in ids]
                viz_examples.append({
                    "example_idx": i,
                    "sentence": full_prompt[-80:],
                    "tokens": toks,
                    "n_tokens": len(toks),
                })
        elif exp_type == "entropy_split":
            do_viz = (i < 2)
            pred, viz = score_choices_entropy_split(
                model, tokenizer, full_prompt, choices, device,
                threshold=threshold, visualize=do_viz
            )
            if do_viz and viz:
                viz["example_idx"] = i
                viz_examples.append(viz)
        else:
            raise ValueError(f"Unknown exp_type: {exp_type}")

        if pred == gold_idx:
            correct += 1
        total += 1

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(test_ds)}] acc={correct/total:.3f}", flush=True)

    elapsed = time.time() - t0
    acc = correct / total if total > 0 else 0.0

    print(f"\nResult: {acc:.4f} ({correct}/{total}) | {elapsed:.1f}s", flush=True)

    return {
        "model": model_name,
        "exp_type": exp_type,
        "task": task,
        "max_examples": max_examples,
        "threshold": threshold,
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "elapsed_s": round(elapsed, 1),
        "viz_examples": viz_examples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--exp_type", default="plain",
                        choices=["plain", "entropy_split"])
    parser.add_argument("--task", default="copa")
    parser.add_argument("--max_examples", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=3.0)
    parser.add_argument("--num_shots", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--korean_prompt", action="store_true",
                        help="Use Korean-language prompt format for Korean-only models")
    parser.add_argument("--score_by_text", action="store_true",
                        help="Score by full choice text instead of A/B labels (better for base models)")
    parser.add_argument("--kmmlu_subject", type=str, default="all",
                        help="KMMLU subject, or 'all' for all subjects")
    args = parser.parse_args()

    result = evaluate(
        model_name=args.model,
        exp_type=args.exp_type,
        task=args.task,
        max_examples=args.max_examples if args.max_examples > 0 else None,
        threshold=args.threshold,
        num_shots=args.num_shots,
        device_str=args.device,
        korean_prompt=args.korean_prompt,
        score_by_text=args.score_by_text,
        kmmlu_subject=args.kmmlu_subject,
    )

    out_path = args.output or str(RESULTS_DIR / f"multi_model_{args.task}_{args.exp_type}.json")

    # Load existing results and append
    if os.path.exists(out_path):
        with open(out_path) as f:
            all_results = json.load(f)
    else:
        all_results = []

    all_results.append(result)
    with open(out_path, "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
