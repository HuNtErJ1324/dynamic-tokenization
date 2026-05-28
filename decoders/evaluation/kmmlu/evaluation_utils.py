"""
Utility functions and classes for KMMLU (Korean MMLU) evaluation.

Mirrors decoders/evaluation/mmlu/evaluation_utils.py but adapts to the
HAERAE-HUB/KMMLU dataset schema (columns: question, A, B, C, D, answer
[1-indexed], Category, Human Accuracy) and uses the "dev" split for the
5-shot demonstrations (KMMLU's analogue of MMLU's "validation" split).
"""

import time
import statistics
import sys
import torch
from pathlib import Path
from typing import List, Tuple
import random
import numpy as np
import torch
from torch.utils.data import Dataset

from zett.utils import get_surface_form_matrix

# Reuse split helpers from the MMLU eval module to keep behavior identical.
_REPO_ROOT = str(Path(__file__).resolve().parents[3])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from decoders.evaluation.mmlu.split_utils import process_prompts_with_split, minimal_split


# Subjects available as configs in HAERAE-HUB/KMMLU.
KMMLU_SUBJECTS = [
    "Accounting", "Agricultural-Sciences", "Aviation-Engineering-and-Maintenance",
    "Biology", "Chemical-Engineering", "Chemistry", "Civil-Engineering",
    "Computer-Science", "Construction", "Criminal-Law", "Ecology", "Economics",
    "Education", "Electrical-Engineering", "Electronics-Engineering",
    "Energy-Management", "Environmental-Science", "Fashion", "Food-Processing",
    "Gas-Technology-and-Engineering", "Geomatics", "Health", "Industrial-Engineer",
    "Information-Technology", "Interior-Architecture-and-Design", "Korean-History",
    "Law", "Machine-Design-and-Manufacturing", "Management", "Maritime-Engineering",
    "Marketing", "Materials-Engineering", "Math", "Mechanical-Engineering",
    "Nondestructive-Testing", "Patent", "Political-Science-and-Sociology",
    "Psychology", "Public-Safety", "Railway-and-Automotive-Engineering",
    "Real-Estate", "Refrigerating-Machinery", "Social-Welfare", "Taxation",
    "Telecommunications-and-Wireless-Technology",
]


class LatencyTracker:
    """Collects per-batch timings so different exp_types can be compared apples-to-apples."""

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
    def _summarize(times: List[float]) -> dict:
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

    def report(self, args, total_examples: int, label: str = "kmmlu"):
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

        if not args.no_wandb:
            import wandb
            log_dict = {
                f"latency/{label}/total_wall_time_s": wall,
                f"latency/{label}/throughput_ex_per_s": thru,
                f"latency/{label}/per_example_ms": ms_per_ex,
            }
            for stage, stats in (("encode", encode), ("forward", forward)):
                for k, v in stats.items():
                    log_dict[f"latency/{label}/{stage}_{k}"] = v
            wandb.log(log_dict)


def _cuda_sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class _TokenLengthAccumulator:
    """Streams token-length stats across batches (mirrors MMLU version)."""

    def __init__(self, hn_maxlen: int = 7):
        self.hn_maxlen = hn_maxlen
        self.n = 0
        self.sum_len = 0
        self.max_len = 0
        self.n_gt_maxlen = 0
        self._lengths: list[int] = []

    def update(self, batch_tokens) -> None:
        skip = {"<pad>", "<s>", "</s>"}
        for seq in batch_tokens:
            for tok in seq:
                if tok in skip:
                    continue
                L = len(tok)
                self.n += 1
                self.sum_len += L
                if L > self.max_len:
                    self.max_len = L
                if L > self.hn_maxlen:
                    self.n_gt_maxlen += 1
                self._lengths.append(L)

    def summary(self) -> dict:
        if self.n == 0:
            return {"n": 0}
        s = sorted(self._lengths)
        p95_idx = max(0, int(round(0.95 * len(s))) - 1)
        return {
            "n": self.n,
            "mean": self.sum_len / self.n,
            "max": self.max_len,
            "p95": s[p95_idx],
            "frac_gt_hn_maxlen": self.n_gt_maxlen / self.n,
            "hn_maxlen": self.hn_maxlen,
        }


def _report_token_length_stats(stats: dict, label: str, args) -> None:
    if stats.get("n", 0) == 0:
        return
    print(
        f"[{label}] merged-token byte length: "
        f"n={stats['n']}, mean={stats['mean']:.2f}, p95={stats['p95']}, "
        f"max={stats['max']}, frac>{stats['hn_maxlen']}={stats['frac_gt_hn_maxlen']:.4f}",
        flush=True,
    )
    if not args.no_wandb:
        import wandb
        wandb.log({
            f"token_length/{label}/n": stats["n"],
            f"token_length/{label}/mean": stats["mean"],
            f"token_length/{label}/max": stats["max"],
            f"token_length/{label}/p95": stats["p95"],
            f"token_length/{label}/frac_gt_hn_maxlen": stats["frac_gt_hn_maxlen"],
        })


def get_hn_embeddings_for_tokens(
    tokens: List[str],
    tokenizer,
    lang_index: int,
    hypernet,
    source_embeddings: torch.Tensor,
    device: torch.device,
    base_input_embeddings: torch.Tensor,
    base_output_embeddings: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate hypernetwork embeddings for a list of tokens (see mmlu/evaluation_utils.py)."""
    with torch.no_grad():
        target_surface_forms = get_surface_form_matrix(
            tokens,
            maxlen=hypernet.config.hn_surface_maxlen,
            tokenizer_to_use=tokenizer,
        )[0]
        target_surface_forms = torch.from_numpy(target_surface_forms).to(device)

        special_tokens_mask = torch.isin(
            target_surface_forms[:, 0],
            torch.tensor(tokenizer.all_special_ids, device=device),
        )

        predicted_input_embeddings, predicted_output_embeddings, _ = hypernet(
            target_surface_forms,
            lang_index=lang_index,
            source_embeddings=source_embeddings,
        )

        predicted_input_embeddings[special_tokens_mask] = base_input_embeddings[
            target_surface_forms[special_tokens_mask, 0]
        ]
        predicted_output_embeddings[special_tokens_mask] = base_output_embeddings[
            target_surface_forms[special_tokens_mask, 0]
        ]

        return (
            predicted_input_embeddings.to(torch.bfloat16),
            predicted_output_embeddings.to(torch.bfloat16)
        )


def load_kmmlu_splits(ds_subject: str, max_retries: int = 5, base_backoff: float = 2.0):
    """
    Load test + per-subject dev datasets for KMMLU.

    Returns (test_dataset, validation_dataset, per_subject_validation_datasets, subjects_used).
    The returned datasets have a "subject" column added so downstream code can group accuracy
    by subject the same way MMLUDataset does.

    HF Hub occasionally returns 5xx; we retry each per-subject `load_dataset` call with
    exponential backoff so a single transient failure doesn't kill the whole 45-subject job.
    """
    from datasets import load_dataset, concatenate_datasets

    def _with_subject(ds, subject):
        if "subject" in ds.column_names:
            return ds
        return ds.add_column("subject", [subject] * len(ds))

    def _load_with_retry(subject):
        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                return load_dataset("HAERAE-HUB/KMMLU", subject)
            except Exception as e:
                last_err = e
                if attempt == max_retries:
                    break
                sleep_s = base_backoff ** attempt
                print(
                    f"  [retry] {subject}: attempt {attempt}/{max_retries} failed "
                    f"({type(e).__name__}: {e}); sleeping {sleep_s:.1f}s",
                    flush=True,
                )
                time.sleep(sleep_s)
        raise RuntimeError(
            f"Failed to load HAERAE-HUB/KMMLU config {subject!r} after {max_retries} attempts"
        ) from last_err

    if ds_subject == "all":
        subjects = list(KMMLU_SUBJECTS)
    else:
        if ds_subject not in KMMLU_SUBJECTS:
            raise ValueError(
                f"Unknown KMMLU subject {ds_subject!r}. Use 'all' or one of: {KMMLU_SUBJECTS}"
            )
        subjects = [ds_subject]

    test_parts = []
    dev_parts = []
    per_subject_dev = {}
    for subj in subjects:
        print(f"Downloading KMMLU subject {subj}", flush=True)
        ds = _load_with_retry(subj)
        test_parts.append(_with_subject(ds["test"], subj))
        dev = _with_subject(ds["dev"], subj)
        dev_parts.append(dev)
        per_subject_dev[subj] = dev

    test_dataset = concatenate_datasets(test_parts) if len(test_parts) > 1 else test_parts[0]
    validation_dataset = concatenate_datasets(dev_parts) if len(dev_parts) > 1 else dev_parts[0]
    return test_dataset, validation_dataset, per_subject_dev, subjects


class KMMLUDataset(Dataset):
    """
    KMMLU equivalent of MMLUDataset. Reads HAERAE-HUB/KMMLU rows
    (question, A, B, C, D, answer [1-indexed], Category) and produces the same
    (prompt, choices, gold_index, context, init_prompt, subject) tuple shape
    so the existing mmlu collate_fn / evaluator wiring works unchanged.
    """

    def __init__(self, dataset, validation_dataset, validation_datasets, num_shots=5):
        self.dataset = dataset
        self.validation_dataset = validation_dataset
        self.num_shots = num_shots
        self.validation_datasets = validation_datasets

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def _row_to_fields(row):
        question = row["question"]
        choices = [row["A"], row["B"], row["C"], row["D"]]
        answer_idx = int(row["answer"]) - 1  # KMMLU answer is 1-indexed
        subject = row.get("subject") or row.get("Category") or ""
        return question, choices, answer_idx, subject

    def format_prompt(
        self,
        question,
        choices,
        subject: str = "",
        is_context_question: bool = False,
        same_domain_shot: bool = True,
        answer: str = "",
        five_shot: bool = False,
    ):
        subject = subject.replace("_", " ").replace("-", " ")
        if is_context_question:
            assert answer != ""
            if same_domain_shot:
                return f"This question refers to the following information.\n{question.strip()}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer: {answer}\n\n"
            else:
                return f"This question is about {subject} and refers to the following information.\n{question.strip()}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer: {answer}\n\n"
        else:
            if five_shot and same_domain_shot:
                return f"This question refers to the following information.\n{question.strip()}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:"
            elif five_shot and not same_domain_shot:
                return f"This question is about {subject} and refers to the following information.\n{question.strip()}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:"
            return f"{question.strip()}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:"

    def __getitem__(self, idx):
        item = self.dataset[idx]
        question, choices, correct_answer_index, subject = self._row_to_fields(item)
        context = ""
        five_shot = getattr(self, 'five_shot', False)
        same_domain_shot = getattr(self, 'same_domain_shot', True)
        if five_shot:
            for _ in range(self.num_shots):
                if not same_domain_shot:
                    example = random.choice(self.validation_dataset)
                else:
                    example = random.choice(self.validation_datasets[subject])
                ex_q, ex_choices, ex_idx, ex_subj = self._row_to_fields(example)
                while ex_q == question and set(ex_choices) == set(choices):
                    if not same_domain_shot:
                        example = random.choice(self.validation_dataset)
                    else:
                        example = random.choice(self.validation_datasets[subject])
                    ex_q, ex_choices, ex_idx, ex_subj = self._row_to_fields(example)

                if ex_q == question and set(ex_choices) == set(choices):
                    raise Exception(
                        "Context question should be different than prompt question. Please check!"
                    )

                example_answer = chr(65 + ex_idx)
                if same_domain_shot:
                    assert ex_subj == subject
                example_prompt = self.format_prompt(
                    question=ex_q,
                    choices=ex_choices,
                    is_context_question=True,
                    answer=example_answer,
                    same_domain_shot=same_domain_shot,
                    subject=ex_subj,
                )

                context += example_prompt

        prompt = context + self.format_prompt(
            question=question,
            choices=choices,
            subject=subject,
            five_shot=five_shot,
            same_domain_shot=same_domain_shot,
        )
        if (five_shot and same_domain_shot) or (not five_shot):
            subject_pretty = subject.replace("_", " ").replace("-", " ")
            prompt = f"The following are multiple choice questions (with answers) about {subject_pretty}.\n\n{prompt}"
        elif (five_shot and not same_domain_shot):
            prompt = f"The following are multiple choice questions (with answers).\n\n{prompt}"
        init_prompt = self.format_prompt(
            question=question,
            choices=choices,
            subject=subject,
            five_shot=five_shot,
            same_domain_shot=same_domain_shot,
        )
        return prompt, choices, correct_answer_index, context, init_prompt, subject


def collate_fn(batch):
    prompts = [item[0] for item in batch]
    choices = [item[1] for item in batch]
    correct_answer_indices = [item[2] for item in batch]
    contexts = [item[3] for item in batch]
    init_prompts = [item[4] for item in batch]
    subjects = [item[5] for item in batch]
    return prompts, choices, correct_answer_indices, contexts, init_prompts, subjects


def setup_seed(seed):
    random.seed(0)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate_model(
    dataloader,
    model,
    tokenizer,
    args,
    base_model=None,
    hypernet=None,
    lang_index=None,
    source_embeddings=None,
    datasetEncoder=None,
    inout_1M_embeddings=None,
    subjects=None,
):
    """Same scoring strategy as MMLU: pick the choice (A/B/C/D) with the highest score
    at the last prompt token, either via plain logits or via hypernet output embeddings."""
    eval_type = args.eval_type.lower()
    if eval_type not in ("original", "origianl"):
        raise NotImplementedError(
            f"evaluate_model only supports eval_type='original'; got {args.eval_type!r}."
        )

    if args.exp_type == "plain":
        return _evaluate_plain(dataloader, model, tokenizer, args, do_split=False)
    if args.exp_type == "entropy_split":
        return _evaluate_plain(dataloader, model, tokenizer, args, do_split=True)
    if args.exp_type in ("original_tk_hypernet", "lp_tk_hypernet", "dynamic_bpe"):
        return _evaluate_hypernet(
            dataloader=dataloader,
            model=model,
            tokenizer=tokenizer,
            args=args,
            hypernet=hypernet,
            lang_index=lang_index,
            source_embeddings=source_embeddings,
            datasetEncoder=datasetEncoder,
        )
    if args.exp_type == "dynamic_bpe_entropy_split":
        return _evaluate_dynamic_bpe_entropy_split(
            dataloader=dataloader,
            model=model,
            tokenizer=tokenizer,
            args=args,
            hypernet=hypernet,
            lang_index=lang_index,
            source_embeddings=source_embeddings,
            datasetEncoder=datasetEncoder,
        )
    raise NotImplementedError(
        f"evaluate_model: exp_type={args.exp_type!r} is not implemented."
    )


def _letter_token_id(tokenizer, letter: str) -> int:
    for cand in (f" {letter}", letter):
        ids = tokenizer.encode(cand, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    return tokenizer.encode(f" {letter}", add_special_tokens=False)[0]


def _print_and_log(args, total_correct, total_seen, correct_per_subject, total_per_subject):
    import wandb
    overall_acc = total_correct / max(total_seen, 1)
    per_subject_acc = {
        s: correct_per_subject[s] / total_per_subject[s] for s in total_per_subject
    }
    print(f"\nOverall KMMLU accuracy: {overall_acc:.4f} ({total_correct}/{total_seen})")
    for s in sorted(per_subject_acc):
        print(
            f"  {s}: {per_subject_acc[s]:.4f} "
            f"({correct_per_subject[s]}/{total_per_subject[s]})"
        )
    if not args.no_wandb:
        wandb.log(
            {
                "kmmlu/overall_accuracy": overall_acc,
                "kmmlu/total_correct": total_correct,
                "kmmlu/total_seen": total_seen,
                **{f"kmmlu/per_subject/{s}": v for s, v in per_subject_acc.items()},
            }
        )
    return overall_acc, per_subject_acc


def _evaluate_plain(dataloader, model, tokenizer, args, do_split: bool = False):
    """
    Vanilla KMMLU scoring on Mistral's native tokenization.

    If do_split=True (i.e., exp_type='entropy_split'), runs entropy-driven prompt
    re-tokenization using args.entropy_threshold before scoring.
    """
    from collections import defaultdict

    device = next(model.parameters()).device
    model.eval()

    prev_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    do_split = bool(do_split or getattr(args, "split", False))
    entropy_threshold = float(getattr(args, "entropy_threshold", 3.0))

    choice_token_ids = torch.tensor(
        [_letter_token_id(tokenizer, L) for L in ("A", "B", "C", "D")], device=device
    )

    correct_per_subject = defaultdict(int)
    total_per_subject = defaultdict(int)
    total_correct = 0
    total_seen = 0

    latency = LatencyTracker()
    latency.start()

    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                prompts, _, gold_indices, _, _, subjects_in_batch = batch

                t0 = time.perf_counter()
                if do_split:
                    processed_input_ids_list = process_prompts_with_split(
                        model=model,
                        tokenizer=tokenizer,
                        prompts=prompts,
                        split_fn=minimal_split,
                        entropy_threshold=entropy_threshold,
                        device=device,
                    )
                    enc = tokenizer.pad(
                        {"input_ids": [torch.tensor(ids) for ids in processed_input_ids_list]},
                        padding=True,
                        return_tensors="pt",
                    ).to(device)
                else:
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
                choice_logits = last_logits[:, choice_token_ids]
                preds = choice_logits.argmax(dim=-1).tolist()

                for pred, gold, subj in zip(preds, gold_indices, subjects_in_batch):
                    correct = int(pred == gold)
                    correct_per_subject[subj] += correct
                    total_per_subject[subj] += 1
                    total_correct += correct
                    total_seen += 1

                if batch_idx % 50 == 0:
                    running = total_correct / max(total_seen, 1)
                    print(
                        f"[batch {batch_idx}] running acc = {running:.4f} "
                        f"({total_correct}/{total_seen})",
                        flush=True,
                    )
    finally:
        tokenizer.padding_side = prev_padding_side
        latency.stop()

    latency.report(args, total_seen, label="kmmlu")
    return _print_and_log(args, total_correct, total_seen, correct_per_subject, total_per_subject)


def _evaluate_hypernet(
    dataloader,
    model,
    tokenizer,
    args,
    hypernet,
    lang_index,
    source_embeddings,
    datasetEncoder,
):
    """Hypernet-aware KMMLU eval. Uses task='mmlu' for the encoder since the
    prompt structure (multiple-choice with last-token scoring) is identical to MMLU."""
    from collections import defaultdict
    from transformers import AutoTokenizer

    device = next(model.parameters()).device
    model.eval()

    H = model.config.hidden_size
    base_input_emb = source_embeddings[:, :H]
    base_output_emb = source_embeddings[:, H:]

    if args.use_original_emb_for_choices:
        original_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        ids = [_letter_token_id(original_tok, L) for L in ("A", "B", "C", "D")]
        choice_output_emb = base_output_emb[torch.tensor(ids, device=device)]
    else:
        from zett.utils import CHARS_TO_BYTES
        bytes_to_chars = {v: k for k, v in CHARS_TO_BYTES.items()}
        def _to_bbpe(s: str) -> str:
            return "".join(bytes_to_chars[b] for b in s.encode("utf-8"))
        choice_tokens_bbpe = [_to_bbpe(s) for s in (" A", " B", " C", " D")]

        _, choice_output_emb = get_hn_embeddings_for_tokens(
            tokens=choice_tokens_bbpe,
            tokenizer=tokenizer,
            lang_index=lang_index,
            hypernet=hypernet,
            source_embeddings=source_embeddings,
            device=device,
            base_input_embeddings=base_input_emb,
            base_output_embeddings=base_output_emb,
        )

    choice_output_emb = choice_output_emb.to(torch.bfloat16)

    correct_per_subject = defaultdict(int)
    total_per_subject = defaultdict(int)
    total_correct = 0
    total_seen = 0
    tok_len_acc = _TokenLengthAccumulator(hn_maxlen=getattr(args, "surface_form_maxlen", 7))

    latency = LatencyTracker()
    latency.start()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            prompts, _, gold_indices, _, _, subjects_in_batch = batch

            t0 = time.perf_counter()
            encoded = datasetEncoder.encode_examples_unique_tokens_lru(
                examples=list(prompts),
                max_length=args.max_len,
                merges=args.merges,
                task="mmlu",
            )
            inputs_embeds = encoded["inputs_embeds"].to(torch.bfloat16)
            attention_mask = encoded["attention_mask"]
            if "batch_tokens" in encoded:
                tok_len_acc.update(encoded["batch_tokens"])
            _cuda_sync(device)
            latency.encode_times.append(time.perf_counter() - t0)

            t1 = time.perf_counter()
            outputs = model.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )
            last_hidden = outputs.last_hidden_state[:, -1, :]
            scores = last_hidden.float() @ choice_output_emb.float().T
            _cuda_sync(device)
            latency.forward_times.append(time.perf_counter() - t1)

            preds = scores.argmax(dim=-1).tolist()

            for pred, gold, subj in zip(preds, gold_indices, subjects_in_batch):
                correct = int(pred == gold)
                correct_per_subject[subj] += correct
                total_per_subject[subj] += 1
                total_correct += correct
                total_seen += 1

            if batch_idx % 50 == 0:
                running = total_correct / max(total_seen, 1)
                print(
                    f"[batch {batch_idx}] running acc = {running:.4f} "
                    f"({total_correct}/{total_seen})",
                    flush=True,
                )

    latency.stop()
    latency.report(args, total_seen, label="kmmlu")
    _report_token_length_stats(tok_len_acc.summary(), label="kmmlu", args=args)
    return _print_and_log(args, total_correct, total_seen, correct_per_subject, total_per_subject)


# ─── dynamic_bpe + entropy split (multiple-choice variant) ────────────────────
# Helpers below mirror decoders/evaluation/mmlu/evaluation_utils.py. Keep them
# in sync; if you change one, change the other.

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


def _evaluate_dynamic_bpe_entropy_split(
    dataloader,
    model,
    tokenizer,
    args,
    hypernet,
    lang_index,
    source_embeddings,
    datasetEncoder,
):
    """KMMLU MC-scoring variant of dynamic_bpe + entropy_split. See MMLU version."""
    from collections import defaultdict
    from transformers import AutoTokenizer

    device = next(model.parameters()).device
    model.eval()

    H = model.config.hidden_size
    base_input_emb = source_embeddings[:, :H]
    base_output_emb = source_embeddings[:, H:]

    entropy_threshold = float(getattr(args, "entropy_threshold", 3.0))
    special_tokens = set(tokenizer.all_special_tokens)

    if args.use_original_emb_for_choices:
        original_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        ids = [_letter_token_id(original_tok, L) for L in ("A", "B", "C", "D")]
        choice_output_emb = base_output_emb[torch.tensor(ids, device=device)]
    else:
        from zett.utils import CHARS_TO_BYTES
        bytes_to_chars = {v: k for k, v in CHARS_TO_BYTES.items()}
        def _to_bbpe(s: str) -> str:
            return "".join(bytes_to_chars[b] for b in s.encode("utf-8"))
        choice_tokens_bbpe = [_to_bbpe(s) for s in (" A", " B", " C", " D")]
        _, choice_output_emb = get_hn_embeddings_for_tokens(
            tokens=choice_tokens_bbpe,
            tokenizer=tokenizer,
            lang_index=lang_index,
            hypernet=hypernet,
            source_embeddings=source_embeddings,
            device=device,
            base_input_embeddings=base_input_emb,
            base_output_embeddings=base_output_emb,
        )
    choice_output_emb = choice_output_emb.to(torch.bfloat16)

    correct_per_subject = defaultdict(int)
    total_per_subject = defaultdict(int)
    total_correct = 0
    total_seen = 0
    total_split = 0
    total_tokens_considered = 0
    tok_len_acc_pre = _TokenLengthAccumulator(hn_maxlen=getattr(args, "surface_form_maxlen", 7))
    tok_len_acc_post = _TokenLengthAccumulator(hn_maxlen=getattr(args, "surface_form_maxlen", 7))

    latency = LatencyTracker()
    latency.start()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            prompts, _, gold_indices, _, _, subjects_in_batch = batch

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
            tok_len_acc_pre.update(batch_tokens)
            _cuda_sync(device)
            latency.encode_times.append(time.perf_counter() - t0)

            hidden = model.model(
                inputs_embeds=inputs_embeds, attention_mask=attention_mask
            ).last_hidden_state
            logits = model.lm_head(hidden).float()
            probs = torch.softmax(logits, dim=-1)
            entropy_matrix = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).cpu()

            new_batch_tokens = []
            for i, tokens_seq in enumerate(batch_tokens):
                attn = attention_mask[i].tolist()
                pad_offset = attn.index(1) if 1 in attn else 0
                new_seq = []
                for j, tok in enumerate(tokens_seq):
                    pos = pad_offset + j
                    ent = (
                        entropy_matrix[i, pos].item()
                        if pos < entropy_matrix.shape[1]
                        else 0.0
                    )
                    total_tokens_considered += 1
                    if ent > entropy_threshold and tok not in special_tokens:
                        sub = _char_level_split(tok)
                        new_seq.extend(sub)
                        if len(sub) > 1:
                            total_split += 1
                    else:
                        new_seq.append(tok)
                new_batch_tokens.append(new_seq)
            tok_len_acc_post.update(new_batch_tokens)

            t1 = time.perf_counter()
            new_inputs_embeds, new_attn_mask = _build_embeds_from_tokens(
                new_batch_tokens, datasetEncoder, tokenizer, args.max_len, device
            )
            _cuda_sync(device)

            outputs = model.model(
                inputs_embeds=new_inputs_embeds,
                attention_mask=new_attn_mask,
            )
            last_hidden = outputs.last_hidden_state[:, -1, :]
            scores = last_hidden.float() @ choice_output_emb.float().T
            _cuda_sync(device)
            latency.forward_times.append(time.perf_counter() - t1)

            preds = scores.argmax(dim=-1).tolist()

            for pred, gold, subj in zip(preds, gold_indices, subjects_in_batch):
                correct = int(pred == gold)
                correct_per_subject[subj] += correct
                total_per_subject[subj] += 1
                total_correct += correct
                total_seen += 1

            if batch_idx % 50 == 0:
                running = total_correct / max(total_seen, 1)
                split_rate = total_split / max(total_tokens_considered, 1)
                print(
                    f"[batch {batch_idx}] running acc = {running:.4f} "
                    f"({total_correct}/{total_seen}), split_rate={split_rate:.4f}",
                    flush=True,
                )

    latency.stop()
    latency.report(args, total_seen, label="kmmlu_dyn_bpe_entropy_split")
    _report_token_length_stats(tok_len_acc_pre.summary(), label="kmmlu_pre_split", args=args)
    _report_token_length_stats(tok_len_acc_post.summary(), label="kmmlu_post_split", args=args)
    overall, per_subject = _print_and_log(
        args, total_correct, total_seen, correct_per_subject, total_per_subject
    )
    split_rate = total_split / max(total_tokens_considered, 1)
    print(
        f"\n[dyn_bpe_entropy_split] entropy_threshold={entropy_threshold:.2f}  "
        f"split_rate={split_rate:.4f}  "
        f"({total_split}/{total_tokens_considered} merged tokens re-split)",
        flush=True,
    )
    if not args.no_wandb:
        import wandb
        wandb.log({
            "kmmlu/dyn_bpe_entropy_split/entropy_threshold": entropy_threshold,
            "kmmlu/dyn_bpe_entropy_split/split_rate": split_rate,
            "kmmlu/dyn_bpe_entropy_split/total_split": total_split,
            "kmmlu/dyn_bpe_entropy_split/total_tokens_considered": total_tokens_considered,
        })
    return overall, per_subject
