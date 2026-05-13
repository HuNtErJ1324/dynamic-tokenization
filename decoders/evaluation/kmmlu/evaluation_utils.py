"""
Utility functions and classes for KMMLU (Korean MMLU) evaluation.

Mirrors decoders/evaluation/mmlu/evaluation_utils.py but adapts to the
HAERAE-HUB/KMMLU dataset schema (columns: question, A, B, C, D, answer
[1-indexed], Category, Human Accuracy) and uses the "dev" split for the
5-shot demonstrations (KMMLU's analogue of MMLU's "validation" split).
"""

import time
import statistics
import torch
from typing import List, Tuple
import random
import numpy as np
import torch
from torch.utils.data import Dataset

from zett.utils import get_surface_form_matrix


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
        return _evaluate_plain(dataloader, model, tokenizer, args)
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


def _evaluate_plain(dataloader, model, tokenizer, args):
    from collections import defaultdict

    device = next(model.parameters()).device
    model.eval()

    prev_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

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
    return _print_and_log(args, total_correct, total_seen, correct_per_subject, total_per_subject)
