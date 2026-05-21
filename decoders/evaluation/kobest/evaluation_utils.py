"""
Utility functions for KoBEST evaluation.

Supports all 5 KoBEST tasks (BoolQ, COPA, WiC, HellaSwag, SentiNeg) with a
uniform A/B/C/D multiple-choice interface so the same logit-based scorer used
for MMLU and KMMLU works unchanged.

Dataset: skt/kobest_v1
  Configs: kobest_boolq, kobest_copa, kobest_wic, kobest_hellaswag, kobest_sentineg
"""

import time
import statistics
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from typing import List
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tokenizations.dynamic_bpe import CHARS_TO_BYTES as _CHARS_TO_BYTES


# ── GPT-2 byte helpers (char-level split) ─────────────────────────────────────

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
    t = tok.replace("Ġ", " ").replace("▁", " ")
    try:
        bts = bytes([_GPT2_BYTE_DECODER.get(c, ord(c) & 0xFF) for c in t])
        return bts.decode("utf-8")
    except (UnicodeDecodeError, KeyError):
        return None


def _gpt2_encode_text(text: str) -> str:
    return "".join(_GPT2_BYTE_ENCODER[b] for b in text.encode("utf-8"))


def _char_level_split(tok: str) -> List[str]:
    """BPE-merged GPT-2 token → Unicode 글자 단위 GPT-2 토큰 리스트."""
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


def _build_embeds_from_tokens(new_batch_tokens, datasetEncoder, tokenizer, max_len, device):
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

KOBEST_TASKS = ["boolq", "copa", "wic", "hellaswag", "sentineg"]


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
    def total_wall_time(self) -> float:
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

    def report(self, args, total_examples, label="kobest"):
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
            print(f"[latency] forward total={forward['total_s']:.2f}s   mean={forward['mean_ms']:.2f}ms   "
                  f"median={forward['median_ms']:.2f}ms   p95={forward['p95_ms']:.2f}ms   "
                  f"n_batches={forward['n_batches']}", flush=True)


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Prompt formatting ─────────────────────────────────────────────────────────

def _format_boolq(row, answer=""):
    paragraph = row["paragraph"].strip()
    question = row["question"].strip()
    prompt = (
        f"다음 지문을 읽고 질문에 답하세요.\n\n"
        f"지문: {paragraph}\n"
        f"질문: {question}\n"
        f"(A) 예\n(B) 아니오\n"
        f"정답:"
    )
    if answer:
        prompt += f" {answer}\n\n"
    # label 1=True=예=A, label 0=False=아니오=B
    gold = int(row["label"])  # 0 or 1
    gold_idx = 1 - gold       # label 1 → choice A (idx 0), label 0 → choice B (idx 1)
    return prompt, gold_idx


def _format_copa(row, answer=""):
    premise = row["premise"].strip()
    question_type = row.get("question", "effect")
    alt1 = row["alternative_1"].strip()
    alt2 = row["alternative_2"].strip()
    if question_type == "cause":
        instruction = "다음 결과의 원인으로 가장 알맞은 것을 고르세요."
        connector = "원인:"
    else:
        instruction = "다음 원인의 결과로 가장 알맞은 것을 고르세요."
        connector = "결과:"
    prompt = (
        f"{instruction}\n\n"
        f"전제: {premise}\n"
        f"(A) {alt1}\n(B) {alt2}\n"
        f"정답:"
    )
    if answer:
        prompt += f" {answer}\n\n"
    gold_idx = int(row["label"])  # 0 → A, 1 → B
    return prompt, gold_idx


def _format_wic(row, answer=""):
    word = row["target_word"].strip()
    ctx1 = row["context_1"].strip()
    ctx2 = row["context_2"].strip()
    prompt = (
        f"다음 두 문장에서 '{word}'이(가) 같은 의미로 사용되었습니까?\n\n"
        f"문장 1: {ctx1}\n"
        f"문장 2: {ctx2}\n"
        f"(A) 예, 같은 의미입니다\n(B) 아니오, 다른 의미입니다\n"
        f"정답:"
    )
    if answer:
        prompt += f" {answer}\n\n"
    # label 1=same=예=A, label 0=different=아니오=B
    gold = int(row["label"])
    gold_idx = 1 - gold
    return prompt, gold_idx


def _format_hellaswag(row, answer=""):
    context = row["context"].strip()
    endings = [row["ending_1"], row["ending_2"], row["ending_3"], row["ending_4"]]
    prompt = (
        f"다음 문장을 가장 자연스럽게 완성하는 것을 고르세요.\n\n"
        f"{context}\n"
        f"(A) {endings[0]}\n(B) {endings[1]}\n(C) {endings[2]}\n(D) {endings[3]}\n"
        f"정답:"
    )
    if answer:
        prompt += f" {answer}\n\n"
    gold_idx = int(row["label"])  # 0-3 → A-D
    return prompt, gold_idx


def _format_sentineg(row, answer=""):
    sentence = row["sentence"].strip()
    prompt = (
        f"다음 문장의 감정을 분류하세요.\n\n"
        f"문장: {sentence}\n"
        f"(A) 긍정\n(B) 부정\n"
        f"정답:"
    )
    if answer:
        prompt += f" {answer}\n\n"
    # label 1=positive=긍정=A, label 0=negative=부정=B
    gold = int(row["label"])
    gold_idx = 1 - gold
    return prompt, gold_idx


_FORMATTERS = {
    "boolq": _format_boolq,
    "copa": _format_copa,
    "wic": _format_wic,
    "hellaswag": _format_hellaswag,
    "sentineg": _format_sentineg,
}

_NUM_CHOICES = {
    "boolq": 2,
    "copa": 2,
    "wic": 2,
    "hellaswag": 4,
    "sentineg": 2,
}

_CHOICE_LABELS = ["A", "B", "C", "D"]


# ── Dataset ───────────────────────────────────────────────────────────────────

class KoBESTDataset(Dataset):
    """
    Uniform wrapper for all 5 KoBEST tasks.

    Returns (prompt, choices_text, gold_idx, context, init_prompt, task) tuples
    matching the shape expected by the collate_fn and evaluator.
    """

    def __init__(self, dataset, validation_dataset, task, num_shots=5):
        self.dataset = dataset
        self.validation_dataset = validation_dataset
        self.task = task
        self.num_shots = num_shots
        self.five_shot = False
        self._formatter = _FORMATTERS[task]
        self._num_choices = _NUM_CHOICES[task]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        context = ""

        if self.five_shot:
            shot_pool = list(range(len(self.validation_dataset)))
            random.shuffle(shot_pool)
            shots_added = 0
            for shot_idx in shot_pool:
                if shots_added >= self.num_shots:
                    break
                shot_row = self.validation_dataset[shot_idx]
                shot_prompt, shot_gold = self._formatter(shot_row)
                answer_letter = _CHOICE_LABELS[shot_gold]
                shot_prompt_with_ans, _ = self._formatter(shot_row, answer=answer_letter)
                context += shot_prompt_with_ans
                shots_added += 1

        prompt, gold_idx = self._formatter(row)
        full_prompt = context + prompt

        choices_text = _CHOICE_LABELS[:self._num_choices]
        init_prompt, _ = self._formatter(row)
        return full_prompt, choices_text, gold_idx, context, init_prompt, self.task


def collate_fn(batch):
    prompts = [item[0] for item in batch]
    choices = [item[1] for item in batch]
    gold_indices = [item[2] for item in batch]
    contexts = [item[3] for item in batch]
    init_prompts = [item[4] for item in batch]
    tasks = [item[5] for item in batch]
    return prompts, choices, gold_indices, contexts, init_prompts, tasks


# ── Data loading ──────────────────────────────────────────────────────────────

def load_kobest_splits(task):
    from datasets import load_dataset

    config = task  # configs are 'boolq', 'copa', 'wic', 'hellaswag', 'sentineg'
    print(f"Downloading KoBEST config {config} ...", flush=True)
    ds = load_dataset("skt/kobest_v1", config)

    test_split = "test" if "test" in ds else "validation"
    dev_split = "validation" if "validation" in ds else "train"

    return ds[test_split], ds[dev_split]


# ── Token helpers ─────────────────────────────────────────────────────────────

def _letter_token_id(tokenizer, letter):
    ids = tokenizer.encode(f" {letter}", add_special_tokens=False)
    return ids[-1]


def _cuda_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def get_hn_embeddings_for_tokens(tokens, tokenizer, lang_index, hypernet,
                                  source_embeddings, device,
                                  base_input_embeddings, base_output_embeddings):
    from tokenizations.tokenization_utils import get_surface_form_matrix
    surface_forms, _ = get_surface_form_matrix(
        tokens,
        maxlen=hypernet.config.hn_surface_maxlen,
        tokenizer_to_use=tokenizer,
    )
    target_surface_forms = torch.from_numpy(surface_forms).to(device)
    special_mask = torch.isin(
        target_surface_forms[:, 0],
        torch.tensor(tokenizer.all_special_ids, device=device),
    )
    pred_in, pred_out, _ = hypernet(
        target_surface_forms,
        lang_index=lang_index,
        source_embeddings=source_embeddings,
    )
    pred_in[special_mask] = base_input_embeddings[target_surface_forms[special_mask, 0]]
    pred_out[special_mask] = base_output_embeddings[target_surface_forms[special_mask, 0]]
    return pred_in.to(torch.bfloat16), pred_out.to(torch.bfloat16)


# ── Evaluators ────────────────────────────────────────────────────────────────

def _print_results(total_correct, total_seen, correct_per_task, total_per_task):
    acc = total_correct / max(total_seen, 1)
    print(f"\nOverall KoBEST accuracy: {acc:.4f} ({total_correct}/{total_seen})", flush=True)
    for task in sorted(correct_per_task.keys()):
        c = correct_per_task[task]
        t = total_per_task[task]
        print(f"  {task}: {c/max(t,1):.4f} ({c}/{t})", flush=True)
    return acc


def _evaluate_plain(dataloader, model, tokenizer, args):
    device = next(model.parameters()).device
    model.eval()

    # Build choice token IDs for each possible number of choices
    choice_ids_2 = torch.tensor(
        [_letter_token_id(tokenizer, L) for L in ("A", "B")], device=device
    )
    choice_ids_4 = torch.tensor(
        [_letter_token_id(tokenizer, L) for L in ("A", "B", "C", "D")], device=device
    )

    correct_per_task = defaultdict(int)
    total_per_task = defaultdict(int)
    total_correct = 0
    total_seen = 0

    prev_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    latency = LatencyTracker()
    latency.start()

    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                prompts, choices_batch, gold_indices, _, _, tasks_in_batch = batch

                t0 = time.perf_counter()
                enc = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.max_len,
                ).to(device)
                _cuda_sync(device)
                latency.encode_times.append(time.perf_counter() - t0)

                t1 = time.perf_counter()
                logits = model(**enc).logits
                _cuda_sync(device)
                latency.forward_times.append(time.perf_counter() - t1)

                last_logits = logits[:, -1, :]

                for i, (gold, task) in enumerate(zip(gold_indices, tasks_in_batch)):
                    choice_ids = choice_ids_4 if _NUM_CHOICES[task] == 4 else choice_ids_2
                    pred = last_logits[i][choice_ids[:_NUM_CHOICES[task]]].argmax().item()
                    correct = int(pred == gold)
                    correct_per_task[task] += correct
                    total_per_task[task] += 1
                    total_correct += correct
                    total_seen += 1

                if batch_idx % 50 == 0:
                    print(
                        f"[batch {batch_idx}] running acc = {total_correct/max(total_seen,1):.4f} "
                        f"({total_correct}/{total_seen})",
                        flush=True,
                    )
    finally:
        tokenizer.padding_side = prev_padding_side
        latency.stop()

    latency.report(args, total_seen, label="kobest")
    return _print_results(total_correct, total_seen, correct_per_task, total_per_task)


def _evaluate_hypernet(dataloader, model, tokenizer, args,
                        hypernet, lang_index, source_embeddings, datasetEncoder):
    from transformers import AutoTokenizer
    CHARS_TO_BYTES = _CHARS_TO_BYTES

    device = next(model.parameters()).device
    model.eval()

    H = model.config.hidden_size
    base_input_emb = source_embeddings[:, :H]
    base_output_emb = source_embeddings[:, H:]

    bytes_to_chars = {v: k for k, v in CHARS_TO_BYTES.items()}

    def _to_bbpe(s):
        return "".join(bytes_to_chars[b] for b in s.encode("utf-8"))

    if args.use_original_emb_for_choices:
        original_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        ids_4 = [_letter_token_id(original_tok, L) for L in ("A", "B", "C", "D")]
        choice_out_emb_4 = base_output_emb[torch.tensor(ids_4, device=device)]
        ids_2 = [_letter_token_id(original_tok, L) for L in ("A", "B")]
        choice_out_emb_2 = base_output_emb[torch.tensor(ids_2, device=device)]
    else:
        bbpe_4 = [_to_bbpe(s) for s in (" A", " B", " C", " D")]
        _, choice_out_emb_4 = get_hn_embeddings_for_tokens(
            bbpe_4, tokenizer, lang_index, hypernet,
            source_embeddings, device, base_input_emb, base_output_emb,
        )
        bbpe_2 = [_to_bbpe(s) for s in (" A", " B")]
        _, choice_out_emb_2 = get_hn_embeddings_for_tokens(
            bbpe_2, tokenizer, lang_index, hypernet,
            source_embeddings, device, base_input_emb, base_output_emb,
        )

    choice_out_emb_4 = choice_out_emb_4.to(torch.bfloat16)
    choice_out_emb_2 = choice_out_emb_2.to(torch.bfloat16)

    correct_per_task = defaultdict(int)
    total_per_task = defaultdict(int)
    total_correct = 0
    total_seen = 0

    latency = LatencyTracker()
    latency.start()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            prompts, choices_batch, gold_indices, _, _, tasks_in_batch = batch

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
            outputs = model.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            last_hidden = outputs.last_hidden_state[:, -1, :]
            _cuda_sync(device)
            latency.forward_times.append(time.perf_counter() - t1)

            for i, (gold, task) in enumerate(zip(gold_indices, tasks_in_batch)):
                n = _NUM_CHOICES[task]
                choice_emb = choice_out_emb_4[:n] if n == 4 else choice_out_emb_2
                scores = last_hidden[i].float() @ choice_emb.float().T
                pred = scores.argmax().item()
                correct = int(pred == gold)
                correct_per_task[task] += correct
                total_per_task[task] += 1
                total_correct += correct
                total_seen += 1

            if batch_idx % 50 == 0:
                print(
                    f"[batch {batch_idx}] running acc = {total_correct/max(total_seen,1):.4f} "
                    f"({total_correct}/{total_seen})",
                    flush=True,
                )

    latency.stop()
    latency.report(args, total_seen, label="kobest")
    return _print_results(total_correct, total_seen, correct_per_task, total_per_task)


def _evaluate_entropy_split(dataloader, model, tokenizer, args):
    import sys
    from pathlib import Path
    _home = str(Path(__file__).resolve().parents[3])
    if _home not in sys.path:
        sys.path.insert(0, _home)
    from decoders.evaluation.mmlu.split_utils import process_prompts_with_split, minimal_split

    device = next(model.parameters()).device
    model.eval()

    choice_ids_2 = torch.tensor(
        [_letter_token_id(tokenizer, L) for L in ("A", "B")], device=device
    )
    choice_ids_4 = torch.tensor(
        [_letter_token_id(tokenizer, L) for L in ("A", "B", "C", "D")], device=device
    )

    correct_per_task = defaultdict(int)
    total_per_task = defaultdict(int)
    total_correct = 0
    total_seen = 0
    entropy_threshold = getattr(args, "entropy_threshold", 3.0)

    prev_padding_side = tokenizer.padding_side
    latency = LatencyTracker()
    latency.start()

    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                prompts, choices_batch, gold_indices, _, _, tasks_in_batch = batch

                # Entropy scan requires right-padding so token positions align
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

                # Final forward pass with left-padding (standard for causal LM eval)
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
                logits = model(input_ids=input_ids, attention_mask=attn_mask).logits
                _cuda_sync(device)
                latency.forward_times.append(time.perf_counter() - t1)

                last_logits = logits[:, -1, :]

                for i, (gold, task) in enumerate(zip(gold_indices, tasks_in_batch)):
                    n = _NUM_CHOICES[task]
                    choice_ids = choice_ids_4[:n] if n == 4 else choice_ids_2
                    pred = last_logits[i][choice_ids].argmax().item()
                    correct = int(pred == gold)
                    correct_per_task[task] += correct
                    total_per_task[task] += 1
                    total_correct += correct
                    total_seen += 1

                if batch_idx % 50 == 0:
                    print(
                        f"[batch {batch_idx}] running acc = {total_correct/max(total_seen,1):.4f} "
                        f"({total_correct}/{total_seen})",
                        flush=True,
                    )
    finally:
        tokenizer.padding_side = prev_padding_side
        latency.stop()

    latency.report(args, total_seen, label="kobest_entropy_split")
    return _print_results(total_correct, total_seen, correct_per_task, total_per_task)


def _evaluate_dynamic_bpe_entropy_split(dataloader, model, tokenizer, args,
                                         hypernet, lang_index, source_embeddings, datasetEncoder):
    """dynamic BPE → entropy scan → char-level split → re-embed → classify."""
    CHARS_TO_BYTES = _CHARS_TO_BYTES
    from transformers import AutoTokenizer as _AutoTok

    device = next(model.parameters()).device
    model.eval()
    entropy_threshold = getattr(args, "entropy_threshold", 3.0)

    H = model.config.hidden_size
    base_input_emb = source_embeddings[:, :H]
    base_output_emb = source_embeddings[:, H:]

    bytes_to_chars = {v: k for k, v in CHARS_TO_BYTES.items()}

    def _to_bbpe(s):
        return "".join(bytes_to_chars[b] for b in s.encode("utf-8"))

    if args.use_original_emb_for_choices:
        original_tok = _AutoTok.from_pretrained("mistralai/Mistral-7B-v0.1")
        ids_4 = [_letter_token_id(original_tok, L) for L in ("A", "B", "C", "D")]
        choice_out_emb_4 = base_output_emb[torch.tensor(ids_4, device=device)]
        ids_2 = [_letter_token_id(original_tok, L) for L in ("A", "B")]
        choice_out_emb_2 = base_output_emb[torch.tensor(ids_2, device=device)]
    else:
        bbpe_4 = [_to_bbpe(s) for s in (" A", " B", " C", " D")]
        _, choice_out_emb_4 = get_hn_embeddings_for_tokens(
            bbpe_4, tokenizer, lang_index, hypernet,
            source_embeddings, device, base_input_emb, base_output_emb,
        )
        bbpe_2 = [_to_bbpe(s) for s in (" A", " B")]
        _, choice_out_emb_2 = get_hn_embeddings_for_tokens(
            bbpe_2, tokenizer, lang_index, hypernet,
            source_embeddings, device, base_input_emb, base_output_emb,
        )
    choice_out_emb_4 = choice_out_emb_4.to(torch.bfloat16)
    choice_out_emb_2 = choice_out_emb_2.to(torch.bfloat16)

    correct_per_task = defaultdict(int)
    total_per_task = defaultdict(int)
    total_correct = 0
    total_seen = 0

    latency = LatencyTracker()
    latency.start()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            prompts, choices_batch, gold_indices, _, _, tasks_in_batch = batch

            # Step 1: Dynamic BPE encode
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

            # Step 2: Entropy scan
            hidden = model.model(
                inputs_embeds=inputs_embeds, attention_mask=attention_mask
            ).last_hidden_state
            logits = model.lm_head(hidden).float()
            probs = torch.softmax(logits, dim=-1)
            entropy_matrix = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).cpu()

            # Step 3: Char-level split of high-entropy tokens
            new_batch_tokens = []
            for i, tokens_seq in enumerate(batch_tokens):
                attn = attention_mask[i].tolist()
                pad_offset = attn.index(1) if 1 in attn else 0
                new_seq = []
                for j, tok in enumerate(tokens_seq):
                    pos = pad_offset + j
                    ent = entropy_matrix[i, pos].item() if pos < entropy_matrix.shape[1] else 0.0
                    if ent > entropy_threshold and tok not in tokenizer.all_special_tokens:
                        new_seq.extend(_char_level_split(tok))
                    else:
                        new_seq.append(tok)
                new_batch_tokens.append(new_seq)

            # Step 4: Re-embed and forward
            t1 = time.perf_counter()
            new_inputs_embeds, new_attn_mask = _build_embeds_from_tokens(
                new_batch_tokens, datasetEncoder, tokenizer, args.max_len, device
            )
            outputs = model.model(inputs_embeds=new_inputs_embeds, attention_mask=new_attn_mask)
            last_hidden = outputs.last_hidden_state[:, -1, :]
            _cuda_sync(device)
            latency.forward_times.append(time.perf_counter() - t1)

            for i, (gold, task) in enumerate(zip(gold_indices, tasks_in_batch)):
                n = _NUM_CHOICES[task]
                choice_emb = choice_out_emb_4[:n] if n == 4 else choice_out_emb_2
                scores = last_hidden[i].float() @ choice_emb.float().T
                pred = scores.argmax().item()
                correct = int(pred == gold)
                correct_per_task[task] += correct
                total_per_task[task] += 1
                total_correct += correct
                total_seen += 1

            if batch_idx % 50 == 0:
                print(
                    f"[batch {batch_idx}] running acc = {total_correct/max(total_seen,1):.4f} "
                    f"({total_correct}/{total_seen})",
                    flush=True,
                )

    latency.stop()
    latency.report(args, total_seen, label="kobest_dynamic_bpe_entropy_split")
    return _print_results(total_correct, total_seen, correct_per_task, total_per_task)


def evaluate_model(dataloader, model, tokenizer, args,
                   base_model=None, hypernet=None, lang_index=None,
                   source_embeddings=None, datasetEncoder=None, **_):
    if args.exp_type == "plain":
        return _evaluate_plain(dataloader, model, tokenizer, args)
    if args.exp_type in ("original_tk_hypernet", "lp_tk_hypernet", "dynamic_bpe"):
        return _evaluate_hypernet(
            dataloader, model, tokenizer, args,
            hypernet, lang_index, source_embeddings, datasetEncoder,
        )
    if args.exp_type == "entropy_split":
        return _evaluate_entropy_split(dataloader, model, tokenizer, args)
    if args.exp_type == "dynamic_bpe_entropy_split":
        return _evaluate_dynamic_bpe_entropy_split(
            dataloader, model, tokenizer, args,
            hypernet, lang_index, source_embeddings, datasetEncoder,
        )
    raise NotImplementedError(f"exp_type={args.exp_type!r} is not supported.")
