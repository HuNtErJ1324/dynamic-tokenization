"""
Utility functions and classes for MMLU evaluation.

Includes:
- MMLUDataset: PyTorch Dataset for MMLU with 0-shot and 5-shot prompt formatting
- collate_fn: DataLoader collate function
- setup_seed: Seed setup for reproducibility
- Generate token embeddings on-the-fly. This is used to embed tokens during evaluation (i.e., inference time).
"""

import torch
from typing import List, Tuple
import random
import numpy as np
import torch
from torch.utils.data import Dataset

from zett.utils import get_surface_form_matrix


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
    """
    Generate hypernetwork embeddings for a list of tokens.
    
    This function takes a list of tokens and generates their corresponding
    hypernetwork embeddings. Special tokens are handled by using the base
    model's embeddings directly, while other tokens use the hypernetwork
    predictions. The resulting embeddings are converted to bfloat16 for
    memory efficiency.
    
    Args:
        tokens: List of token strings to generate embeddings for
        tokenizer: Hypernetwork tokenizer for surface form generation
        lang_index: Language index tensor for the hypernetwork
        hypernet: Hypernetwork model for embedding prediction
        source_embeddings: Source embeddings for the hypernetwork
        device: Target device for tensor operations
        base_input_embeddings: Base model input embeddings for special tokens
        base_output_embeddings: Base model output embeddings for special tokens
        
    Returns:
        Tuple containing:
        - predicted_input_embeddings: Generated input embeddings (bfloat16)
        - predicted_output_embeddings: Generated output embeddings (bfloat16)
    """
    with torch.no_grad():
        target_surface_forms = get_surface_form_matrix(
            tokens,  # byte representation of the tokens to predict
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

        # Replace special token embeddings with base model embeddings
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

class MMLUDataset(Dataset):
    def __init__(self, dataset, validation_dataset, validation_datasets, num_shots=5):
        self.dataset = dataset
        self.validation_dataset = validation_dataset
        self.num_shots = num_shots
        self.validation_datasets = validation_datasets

    def __len__(self):
        return len(self.dataset)

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
        subject = subject.replace("_", " ")
        if is_context_question:
            assert answer != ""
            if same_domain_shot:
                return f"This question refers to the following information.\n{question.strip()}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer: {answer}\n\n"
            else:  # random domain shots
                return f"This question is about {subject} and refers to the following information.\n{question.strip()}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer: {answer}\n\n"
        else:  # if main prompt question
            if five_shot and same_domain_shot:
                return f"This question refers to the following information.\n{question.strip()}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:"
            elif five_shot and not same_domain_shot:
                return f"This question is about {subject} and refers to the following information.\n{question.strip()}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:"
            return f"{question.strip()}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:"

    def __getitem__(self, idx):
        item = self.dataset[idx]
        question = item["question"]
        choices = item["choices"]
        correct_answer_index = item["answer"]
        subject = item["subject"]
        context = ""
        five_shot = getattr(self, 'five_shot', False)
        same_domain_shot = getattr(self, 'same_domain_shot', True)
        if five_shot:
            for _ in range(self.num_shots):
                if not same_domain_shot:
                    example = random.choice(self.validation_dataset)
                else:
                    example = random.choice(self.validation_datasets[subject])
                while example["question"] == question and set(
                    example["choices"]
                ) == set(choices):
                    if not same_domain_shot:
                        example = random.choice(self.validation_dataset)
                    else:
                        example = random.choice(self.validation_datasets[subject])

                if example["question"] == question and set(example["choices"]) == set(
                    choices
                ):
                    raise Exception(
                        "Context question should be different than prompt question. Please check!"
                    )

                example_question = example["question"]
                example_choices = example["choices"]
                example_answer_index = example["answer"]
                example_answer = chr(65 + example_answer_index)
                if same_domain_shot:
                    assert example["subject"] == subject
                example_prompt = self.format_prompt(
                    question=example_question,
                    choices=example_choices,
                    is_context_question=True,
                    answer=example_answer,
                    same_domain_shot=same_domain_shot,
                    subject=example["subject"],
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
            subject = subject.replace("_", " ")
            prompt = f"The following are multiple choice questions (with answers) about {subject}.\n\n{prompt}"
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
    """
    Score MMLU by picking the choice (A/B/C/D) with the highest score at the
    last prompt token. Two paths:
      - exp_type == "plain": tokenize with Mistral's tokenizer, take logits at
        the last position, argmax over A/B/C/D token IDs.
      - exp_type in {"original_tk_hypernet", "lp_tk_hypernet", "dynamic_bpe"}:
        use the hypernet to produce input embeddings for prompt tokens (via
        DatasetEncoder) and output embeddings for the four letter tokens. Take
        the model's last hidden state, dot it against each letter's output
        embedding, argmax.
    """
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
    """Single-token ID for ' <letter>' (preferred) or '<letter>'."""
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
    print(f"\nOverall MMLU accuracy: {overall_acc:.4f} ({total_correct}/{total_seen})")
    for s in sorted(per_subject_acc):
        print(
            f"  {s}: {per_subject_acc[s]:.4f} "
            f"({correct_per_subject[s]}/{total_per_subject[s]})"
        )
    if not args.no_wandb:
        wandb.log(
            {
                "mmlu/overall_accuracy": overall_acc,
                "mmlu/total_correct": total_correct,
                "mmlu/total_seen": total_seen,
                **{f"mmlu/per_subject/{s}": v for s, v in per_subject_acc.items()},
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

    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                prompts, _, gold_indices, _, _, subjects_in_batch = batch

                enc = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.max_len,
                ).to(device)

                logits = model(**enc).logits
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
    """
    Hypernet-aware MMLU eval. Works for original_tk_hypernet, lp_tk_hypernet,
    and dynamic_bpe — they only differ in how DatasetEncoder tokenizes the
    prompt. Output-side scoring is the same: dot the model's last hidden state
    against hypernet-predicted output embeddings for ' A', ' B', ' C', ' D'.
    """
    from collections import defaultdict
    from transformers import AutoTokenizer

    device = next(model.parameters()).device
    model.eval()

    # Use the live model's embeddings for special-token fallback (already on device, bf16).
    base_input_emb = model.get_input_embeddings().weight.data
    base_output_emb = model.get_output_embeddings().weight.data

    if args.use_original_emb_for_choices:
        # Mistral's native output embeddings for ' A', ' B', ' C', ' D'.
        # Use a fresh original tokenizer in case `tokenizer` was swapped (lp_tk_hypernet).
        original_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        ids = [_letter_token_id(original_tok, L) for L in ("A", "B", "C", "D")]
        choice_output_emb = base_output_emb[torch.tensor(ids, device=device)]
    else:
        # zett expects byte-level (BBPE) surface forms — space encodes to 'Ġ'.
        # Convert " A"/" B"/" C"/" D" into BBPE form before handing to the hypernet.
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

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            prompts, _, gold_indices, _, _, subjects_in_batch = batch

            encoded = datasetEncoder.encode_examples_unique_tokens_lru(
                examples=list(prompts),
                max_length=args.max_len,
                merges=args.merges,
                task="mmlu",
            )
            inputs_embeds = encoded["inputs_embeds"].to(torch.bfloat16)
            attention_mask = encoded["attention_mask"]

            outputs = model.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )
            last_hidden = outputs.last_hidden_state[:, -1, :]
            scores = last_hidden.float() @ choice_output_emb.float().T
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

    return _print_and_log(args, total_correct, total_seen, correct_per_subject, total_per_subject)
