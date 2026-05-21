"""
Utility functions for HRM8K evaluation.

HRM8K contains 8,011 English-Korean parallel bilingual math problems.
We evaluate both the English and Korean versions with the same model to
measure whether the English-Korean performance gap widens under different
tokenization schemes.

Evaluation is generation-based: the model generates a free-form answer, and
we extract the last numerical value from the output to compare to the gold.

Dataset: HAERAE-HUB/HRM8K (fallback: HRM8K/HRM8K)
"""

import re
import time
import statistics
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from typing import List, Optional


# ── GPT-2 byte decoder (for char-level split) ─────────────────────────────────

def _build_gpt2_byte_decoder():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_GPT2_BYTE_DECODER = _build_gpt2_byte_decoder()
_GPT2_BYTE_ENCODER = {b: c for c, b in _GPT2_BYTE_DECODER.items()}


def _gpt2_decode(tok: str):
    """GPT-2 스타일 토큰(Ġ/▁ prefix 포함) → 실제 텍스트. 실패하면 None."""
    t = tok.replace("Ġ", " ").replace("▁", " ")
    try:
        bts = bytes([_GPT2_BYTE_DECODER.get(c, ord(c) & 0xFF) for c in t])
        return bts.decode("utf-8")
    except (UnicodeDecodeError, KeyError):
        return None


def _gpt2_encode_text(text: str) -> str:
    """텍스트 → GPT-2 byte 인코딩 문자열 (예: ' 인' → 'ĠìĿ¸')"""
    return "".join(_GPT2_BYTE_ENCODER[b] for b in text.encode("utf-8"))


def _char_level_split(tok: str) -> List[str]:
    """
    BPE-merged GPT-2 token을 최소 1 Unicode 글자 단위로 split.
    tokenize() 미사용 → spurious space 없음.
    예: Ġìĭ¬íĺĪê´Ģ (= ' 심혈관') → ['Ġìĭ¬', 'íĺĪ', 'ê´Ģ']  (3 tokens)
    """
    text = _gpt2_decode(tok)
    if text is None:
        return [tok]
    chars = list(text)
    if len(chars) <= 1:
        return [tok]

    result = []
    i = 0
    if chars[0] == " " and len(chars) > 1:
        result.append(_gpt2_encode_text(" " + chars[1]))
        i = 2
    while i < len(chars):
        result.append(_gpt2_encode_text(chars[i]))
        i += 1
    return result if result else [tok]


# ── Latency tracking ─────────────────────────────────────────────────────────

class LatencyTracker:
    def __init__(self):
        self.encode_times: List[float] = []
        self.forward_times: List[float] = []
        self._wall_start = None
        self._wall_end = None

    def start(self):
        self._wall_start = time.perf_counter()

    def stop(self):
        self._wall_end = time.perf_counter()

    @property
    def total_wall_time(self):
        if self._wall_start is None or self._wall_end is None:
            return 0.0
        return self._wall_end - self._wall_start

    @staticmethod
    def _summarize(times):
        if not times:
            return {}
        srt = sorted(times)
        p95_idx = max(0, int(round(0.95 * len(srt))) - 1)
        return {
            "total_s": sum(times),
            "mean_ms": statistics.mean(times) * 1000.0,
            "median_ms": statistics.median(times) * 1000.0,
            "p95_ms": srt[p95_idx] * 1000.0,
            "n_batches": len(times),
        }

    def report(self, args, total_examples, label="hrm8k"):
        encode = self._summarize(self.encode_times)
        forward = self._summarize(self.forward_times)
        wall = self.total_wall_time
        thru = total_examples / wall if wall > 0 else 0.0
        ms_per_ex = (wall / total_examples * 1000.0) if total_examples > 0 else 0.0
        print(f"\n--- {label.upper()} latency ({args.exp_type}, batch_size={args.batch_size}) ---", flush=True)
        print(f"[latency] total_wall_time={wall:.2f}s   examples={total_examples}   "
              f"throughput={thru:.2f} ex/s   per_example={ms_per_ex:.2f} ms", flush=True)
        if encode:
            print(f"[latency] encode  total={encode['total_s']:.2f}s   mean={encode['mean_ms']:.2f}ms   "
                  f"median={encode['median_ms']:.2f}ms   p95={encode['p95_ms']:.2f}ms   "
                  f"n_batches={encode['n_batches']}", flush=True)
        if forward:
            print(f"[latency] generate total={forward['total_s']:.2f}s   mean={forward['mean_ms']:.2f}ms   "
                  f"median={forward['median_ms']:.2f}ms   p95={forward['p95_ms']:.2f}ms   "
                  f"n_batches={forward['n_batches']}", flush=True)


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Answer extraction ─────────────────────────────────────────────────────────

def extract_answer(text: str) -> Optional[float]:
    """
    Extract the last number (integer or decimal) from generated text.
    Returns None if no number is found.
    """
    # Remove commas in numbers (e.g. "1,234" → "1234")
    text = text.replace(",", "")
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not numbers:
        return None
    try:
        return float(numbers[-1])
    except ValueError:
        return None


def answers_match(pred: Optional[float], gold: Optional[float], tol: float = 1e-6) -> bool:
    if pred is None or gold is None:
        return False
    if gold == 0:
        return abs(pred) < tol
    return abs(pred - gold) / (abs(gold) + tol) < tol or abs(pred - gold) < tol


# ── Prompt formatting ─────────────────────────────────────────────────────────

_FEW_SHOT_EN = [
    ("A store sells apples for $2 each and oranges for $3 each. "
     "If you buy 4 apples and 3 oranges, how much do you spend in total?",
     "17"),
    ("A rectangle has a length of 8 cm and a width of 5 cm. "
     "What is the area of the rectangle?",
     "40"),
    ("If a car travels at 60 km/h for 2.5 hours, how far does it travel?",
     "150"),
]

_FEW_SHOT_KO = [
    ("한 가게에서 사과는 개당 2,000원, 오렌지는 개당 3,000원에 팔고 있습니다. "
     "사과 4개와 오렌지 3개를 구매하면 총 얼마를 지불해야 합니까?",
     "17000"),
    ("직사각형의 가로가 8cm이고 세로가 5cm일 때, 직사각형의 넓이는 얼마입니까?",
     "40"),
    ("자동차가 시속 60km로 2.5시간 동안 달리면 이동한 거리는 얼마입니까?",
     "150"),
]


def format_prompt(question: str, lang: str, shots: list, include_answer: bool = False,
                  answer: str = "") -> str:
    if lang == "en":
        instruction = "Solve the following math problem. Write only the final numerical answer."
        answer_prefix = "Answer:"
    else:
        instruction = "다음 수학 문제를 풀어 최종 숫자 답만 쓰세요."
        answer_prefix = "정답:"

    prompt = f"{instruction}\n\n"
    for q, a in shots:
        prompt += f"문제: {q}\n{answer_prefix} {a}\n\n" if lang == "ko" else f"Problem: {q}\n{answer_prefix} {a}\n\n"

    if lang == "ko":
        prompt += f"문제: {question}\n{answer_prefix}"
    else:
        prompt += f"Problem: {question}\n{answer_prefix}"

    if include_answer:
        prompt += f" {answer}\n\n"

    return prompt


# ── Dataset ───────────────────────────────────────────────────────────────────

def _get_question_field(row, lang):
    """Flexible column name lookup for Korean and English question fields."""
    if lang == "ko":
        for col in ("question_ko", "question", "kor_question", "korean_question", "ko"):
            if col in row and row[col]:
                return str(row[col])
    else:
        for col in ("question_en", "question_english", "eng_question", "english_question", "en"):
            if col in row and row[col]:
                return str(row[col])
        # If only one question column, use it for English too
        if "question" in row:
            return str(row["question"])
    return None


def _get_answer_field(row):
    for col in ("answer", "gold", "label", "ans", "solution"):
        if col in row:
            val = row[col]
            try:
                return str(val)
            except Exception:
                pass
    return None


class HRM8KDataset(Dataset):
    """
    Dataset wrapper for HRM8K. Each item returns a Korean prompt and its
    paired English prompt for the same problem, enabling direct comparison.
    """

    def __init__(self, dataset, lang: str, five_shot: bool = False, num_shots: int = 3):
        self.dataset = dataset
        self.lang = lang
        self.five_shot = five_shot
        self.num_shots = num_shots
        self._shots = (_FEW_SHOT_KO if lang == "ko" else _FEW_SHOT_EN)[:num_shots]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        question = _get_question_field(row, self.lang)
        if question is None:
            question = str(list(row.values())[0])

        answer_str = _get_answer_field(row) or "0"
        shots = self._shots if self.five_shot else []
        prompt = format_prompt(question, self.lang, shots)
        return prompt, answer_str, question


def collate_fn(batch):
    prompts = [item[0] for item in batch]
    answers = [item[1] for item in batch]
    questions = [item[2] for item in batch]
    return prompts, answers, questions


# ── Data loading ──────────────────────────────────────────────────────────────

def load_hrm8k(lang: str = "ko", max_retries: int = 3):
    """
    Load math benchmark from HAERAE-HUB/HRM8K.
    lang='ko' uses the KSM (Korean School Math) config.
    lang='en' uses the GSM8K config.
    """
    from datasets import load_dataset

    config = "KSM" if lang == "ko" else "GSM8K"
    print(f"Loading HAERAE-HUB/HRM8K config={config} ...", flush=True)

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            ds = load_dataset("HAERAE-HUB/HRM8K", config)
            split = "test" if "test" in ds else list(ds.keys())[0]
            print(f"Loaded split={split}, columns={ds[split].column_names}", flush=True)
            return ds[split]
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(2 ** attempt)

    raise RuntimeError(
        f"Could not load HAERAE-HUB/HRM8K config={config}. Last error: {last_err}"
    )


# ── Evaluator ─────────────────────────────────────────────────────────────────

def _cuda_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _run_generation(dataloader, model, tokenizer, args, lang, label_suffix=""):
    """Shared generation loop for plain and entropy_split modes."""
    device = next(model.parameters()).device

    total_correct = 0
    total_seen = 0
    total_no_answer = 0

    latency = LatencyTracker()
    latency.start()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            prompts, gold_answers, _ = batch

            t0 = time.perf_counter()
            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_len,
            ).to(device)
            input_len = enc["input_ids"].shape[1]
            _cuda_sync(device)
            latency.encode_times.append(time.perf_counter() - t0)

            t1 = time.perf_counter()
            output_ids = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            _cuda_sync(device)
            latency.forward_times.append(time.perf_counter() - t1)

            new_tokens = output_ids[:, input_len:]
            generated_texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

            for gen_text, gold_str in zip(generated_texts, gold_answers):
                pred = extract_answer(gen_text)
                gold = extract_answer(gold_str)
                if pred is None:
                    total_no_answer += 1
                if answers_match(pred, gold):
                    total_correct += 1
                total_seen += 1

            if batch_idx % 20 == 0:
                running = total_correct / max(total_seen, 1)
                print(
                    f"[batch {batch_idx}] running acc = {running:.4f} "
                    f"({total_correct}/{total_seen}), no_answer={total_no_answer}",
                    flush=True,
                )

    latency.stop()
    label = f"hrm8k_{lang}{label_suffix}"
    latency.report(args, total_seen, label=label)

    acc = total_correct / max(total_seen, 1)
    print(f"\nHRM8K [{lang}]{label_suffix} accuracy: {acc:.4f} ({total_correct}/{total_seen})", flush=True)
    print(f"  No-answer rate: {total_no_answer/max(total_seen,1):.4f} ({total_no_answer}/{total_seen})", flush=True)
    return acc


def _run_entropy_split_generation(dataloader, model, tokenizer, args, lang):
    """Generation-based evaluation with entropy-guided token splitting."""
    import sys
    from pathlib import Path
    _home = str(Path(__file__).resolve().parents[3])
    if _home not in sys.path:
        sys.path.insert(0, _home)
    from decoders.evaluation.mmlu.split_utils import process_prompts_with_split, minimal_split

    device = next(model.parameters()).device

    total_correct = 0
    total_seen = 0
    total_no_answer = 0
    entropy_threshold = getattr(args, "entropy_threshold", 3.0)

    latency = LatencyTracker()
    latency.start()

    prev_padding_side = tokenizer.padding_side
    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                prompts, gold_answers, _ = batch

                # Entropy scan: right-padding required for position alignment
                t0 = time.perf_counter()
                tokenizer.padding_side = "right"
                new_ids_list = process_prompts_with_split(
                    model, tokenizer, list(prompts),
                    minimal_split,
                    entropy_threshold=entropy_threshold,
                    device=device,
                )
                _cuda_sync(device)
                latency.encode_times.append(time.perf_counter() - t0)

                # Build left-padded input for generation
                max_len = min(args.max_len, max(len(ids) for ids in new_ids_list))
                pad_id = tokenizer.pad_token_id
                input_ids = torch.full(
                    (len(new_ids_list), max_len), pad_id, dtype=torch.long, device=device
                )
                attn_mask = torch.zeros(
                    len(new_ids_list), max_len, dtype=torch.long, device=device
                )
                for i, ids in enumerate(new_ids_list):
                    seq = ids[-max_len:]
                    input_ids[i, -len(seq):] = torch.tensor(seq, dtype=torch.long)
                    attn_mask[i, -len(seq):] = 1

                t1 = time.perf_counter()
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                _cuda_sync(device)
                latency.forward_times.append(time.perf_counter() - t1)

                new_tokens = output_ids[:, max_len:]
                generated_texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

                for gen_text, gold_str in zip(generated_texts, gold_answers):
                    pred = extract_answer(gen_text)
                    gold = extract_answer(gold_str)
                    if pred is None:
                        total_no_answer += 1
                    if answers_match(pred, gold):
                        total_correct += 1
                    total_seen += 1

                if batch_idx % 20 == 0:
                    running = total_correct / max(total_seen, 1)
                    print(
                        f"[batch {batch_idx}] running acc = {running:.4f} "
                        f"({total_correct}/{total_seen}), no_answer={total_no_answer}",
                        flush=True,
                    )
    finally:
        tokenizer.padding_side = prev_padding_side

    latency.stop()
    latency.report(args, total_seen, label=f"hrm8k_{lang}_entropy_split")

    acc = total_correct / max(total_seen, 1)
    print(f"\nHRM8K [{lang}] entropy_split accuracy: {acc:.4f} ({total_correct}/{total_seen})", flush=True)
    print(f"  No-answer rate: {total_no_answer/max(total_seen,1):.4f} ({total_no_answer}/{total_seen})", flush=True)
    return acc


def _build_embeds_from_tokens(new_batch_tokens, datasetEncoder, tokenizer, max_len, device):
    """Build left-padded inputs_embeds from token sequences using the hypernetwork cache."""
    unique_tokens = set()
    for seq in new_batch_tokens:
        unique_tokens.update(seq)
    unique_tokens.add(tokenizer.pad_token)

    datasetEncoder.compute_tokens_batch_embeddings(unique_tokens, task="mmlu")
    hp = datasetEncoder.embeddings_cache.hypernet_preds
    t2i = datasetEncoder.embeddings_cache.token2idx
    emb_size = datasetEncoder.embeddings_cache.emb_size

    max_seq_len = min(max_len, max(len(seq) for seq in new_batch_tokens))
    batch_size = len(new_batch_tokens)
    inputs_embeds = torch.zeros(batch_size, max_seq_len, emb_size, dtype=torch.bfloat16, device=device)
    attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.long, device=device)

    for i, tokens_seq in enumerate(new_batch_tokens):
        seq = tokens_seq[-max_seq_len:]
        pad_len = max_seq_len - len(seq)
        for j, tok in enumerate(seq):
            if tok in t2i:
                inputs_embeds[i, pad_len + j] = hp[t2i[tok]]
        attention_mask[i, pad_len:pad_len + len(seq)] = 1

    return inputs_embeds, attention_mask


def _run_dynamic_bpe_generation(dataloader, model, tokenizer, args, lang, datasetEncoder):
    """HRM8K generation with dynamic BPE + hypernetwork input embeddings."""
    device = next(model.parameters()).device
    model.eval()

    total_correct = 0
    total_seen = 0
    total_no_answer = 0
    latency = LatencyTracker()
    latency.start()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            prompts, gold_answers, _ = batch

            t0 = time.perf_counter()
            encoded = datasetEncoder.encode_examples_unique_tokens_lru(
                examples=list(prompts),
                max_length=args.max_len,
                merges=args.merges,
                task="mmlu",
            )
            inputs_embeds = encoded["inputs_embeds"].to(torch.bfloat16)
            attention_mask = encoded["attention_mask"]
            _cuda_sync(device)
            latency.encode_times.append(time.perf_counter() - t0)

            t1 = time.perf_counter()
            output_ids = model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            _cuda_sync(device)
            latency.forward_times.append(time.perf_counter() - t1)

            generated_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            for gen_text, gold_str in zip(generated_texts, gold_answers):
                pred = extract_answer(gen_text)
                gold = extract_answer(gold_str)
                if pred is None:
                    total_no_answer += 1
                if answers_match(pred, gold):
                    total_correct += 1
                total_seen += 1

            if batch_idx % 20 == 0:
                running = total_correct / max(total_seen, 1)
                print(
                    f"[batch {batch_idx}] running acc = {running:.4f} "
                    f"({total_correct}/{total_seen}), no_answer={total_no_answer}",
                    flush=True,
                )

    latency.stop()
    latency.report(args, total_seen, label=f"hrm8k_{lang}_dynamic_bpe")
    acc = total_correct / max(total_seen, 1)
    print(f"\nHRM8K [{lang}] dynamic_bpe accuracy: {acc:.4f} ({total_correct}/{total_seen})", flush=True)
    print(f"  No-answer rate: {total_no_answer/max(total_seen,1):.4f} ({total_no_answer}/{total_seen})", flush=True)
    return acc


def _run_dynamic_bpe_entropy_split_generation(dataloader, model, tokenizer, args, lang, datasetEncoder):
    """dynamic BPE merging → entropy scan → split high-entropy merged tokens → generate."""
    device = next(model.parameters()).device
    model.eval()

    entropy_threshold = getattr(args, "entropy_threshold", 3.0)
    total_correct = 0
    total_seen = 0
    total_no_answer = 0
    latency = LatencyTracker()
    latency.start()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            prompts, gold_answers, _ = batch

            # Step 1: Dynamic BPE encoding
            t0 = time.perf_counter()
            encoded = datasetEncoder.encode_examples_unique_tokens_lru(
                examples=list(prompts),
                max_length=args.max_len,
                merges=args.merges,
                task="mmlu",
            )
            inputs_embeds = encoded["inputs_embeds"].to(torch.bfloat16)
            attention_mask = encoded["attention_mask"]
            batch_tokens = encoded["batch_tokens"]
            _cuda_sync(device)
            latency.encode_times.append(time.perf_counter() - t0)

            # Step 2: Entropy scan with merged embeddings
            hidden = model.model(
                inputs_embeds=inputs_embeds, attention_mask=attention_mask
            ).last_hidden_state
            logits = model.lm_head(hidden).float()
            probs = torch.softmax(logits, dim=-1)
            entropy_matrix = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).cpu()

            # Step 3: Split high-entropy merged tokens back into sub-tokens
            new_batch_tokens = []
            for i, tokens_seq in enumerate(batch_tokens):
                attn = attention_mask[i].tolist()
                pad_offset = attn.index(1) if 1 in attn else 0
                new_seq = []
                for j, tok in enumerate(tokens_seq):
                    pos = pad_offset + j
                    ent = entropy_matrix[i, pos].item() if pos < entropy_matrix.shape[1] else 0.0
                    if ent > entropy_threshold and tok not in tokenizer.all_special_tokens:
                        sub = _char_level_split(tok)
                        new_seq.extend(sub)
                    else:
                        new_seq.append(tok)
                new_batch_tokens.append(new_seq)

            # Step 4: Re-embed split sequences with hypernetwork
            t1 = time.perf_counter()
            new_inputs_embeds, new_attn_mask = _build_embeds_from_tokens(
                new_batch_tokens, datasetEncoder, tokenizer, args.max_len, device
            )
            _cuda_sync(device)

            # Step 5: Generate from split embeddings
            output_ids = model.generate(
                inputs_embeds=new_inputs_embeds,
                attention_mask=new_attn_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            _cuda_sync(device)
            latency.forward_times.append(time.perf_counter() - t1)

            generated_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            for gen_text, gold_str in zip(generated_texts, gold_answers):
                pred = extract_answer(gen_text)
                gold = extract_answer(gold_str)
                if pred is None:
                    total_no_answer += 1
                if answers_match(pred, gold):
                    total_correct += 1
                total_seen += 1

            if batch_idx % 20 == 0:
                running = total_correct / max(total_seen, 1)
                print(
                    f"[batch {batch_idx}] running acc = {running:.4f} "
                    f"({total_correct}/{total_seen}), no_answer={total_no_answer}",
                    flush=True,
                )

    latency.stop()
    latency.report(args, total_seen, label=f"hrm8k_{lang}_dynamic_bpe_entropy_split")
    acc = total_correct / max(total_seen, 1)
    print(f"\nHRM8K [{lang}] dynamic_bpe+entropy_split accuracy: {acc:.4f} ({total_correct}/{total_seen})", flush=True)
    print(f"  No-answer rate: {total_no_answer/max(total_seen,1):.4f} ({total_no_answer}/{total_seen})", flush=True)
    return acc


def evaluate_model(dataloader, model, tokenizer, args, lang: str,
                   datasetEncoder=None, **_):
    model.eval()
    if args.exp_type == "plain":
        return _run_generation(dataloader, model, tokenizer, args, lang)
    if args.exp_type == "entropy_split":
        return _run_entropy_split_generation(dataloader, model, tokenizer, args, lang)
    if args.exp_type == "dynamic_bpe":
        return _run_dynamic_bpe_generation(dataloader, model, tokenizer, args, lang, datasetEncoder)
    if args.exp_type == "dynamic_bpe_entropy_split":
        return _run_dynamic_bpe_entropy_split_generation(
            dataloader, model, tokenizer, args, lang, datasetEncoder
        )
    raise NotImplementedError(
        f"HRM8K evaluation: exp_type={args.exp_type!r} is not supported. "
        "Use 'plain', 'entropy_split', 'dynamic_bpe', or 'dynamic_bpe_entropy_split'."
    )
