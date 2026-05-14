"""
KMMLU (Korean MMLU) Evaluation Script for Mistral.

Mirrors decoders/evaluation/mmlu/mistral_mmlu.py but evaluates on the
HAERAE-HUB/KMMLU dataset. Defaults --lng to "ko" so the hypernetwork is
prompted with the Korean language index. See argparse for all options.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch
import argparse
import sys
from torch.utils.data import DataLoader
import wandb
from evaluation_utils import (
    KMMLUDataset,
    KMMLU_SUBJECTS,
    collate_fn,
    setup_seed,
    evaluate_model,
    load_kmmlu_splits,
)
from tokenizers.models import BPE, WordPiece

from pathlib import Path
HOME_PATH = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, HOME_PATH)
from tokenizations.tokenization_utils import DatasetEncoder
from tokenizations.hypernet_cache import LRU_Cache


def main():
    parser = argparse.ArgumentParser(description="Running KMMLU Evaluation")
    parser.add_argument("--ds_subject", type=str, default="all", help="KMMLU subject config to use, or 'all' to concatenate every subject.")
    parser.add_argument("--exp_type", type=str, default="plain", help="plain | original_tk_hypernet | lp_tk_hypernet | dynamic_bpe")
    parser.add_argument("--verbose", type=bool, default=False, help="Add extra loggings.")
    parser.add_argument("--eval_type", type=str, default="original", help="Original (compare with probs of A, B, C or D)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size to use during evaluation")
    parser.add_argument("--five_shot", action="store_true", help="Perform 5-shot evaluation")
    parser.add_argument("--no_wandb", action="store_true", help="Don't log data to wandb.")
    parser.add_argument("--exp_prefix", type=str, default="", help="Prefix to be added for the exp name in wandb")
    parser.add_argument("--same_domain_shot", action="store_true", help="Choose 5 shots from the same subject (different split)")
    parser.add_argument("--lng", type=str, default="ko", help="Language index for hypernetwork (KMMLU defaults to 'ko')")
    parser.add_argument("--max_len", type=int, default=4096, help="Max length to be used during tokenization")
    parser.add_argument("--vocab_1M", action="store_true", help="Use tokenizer with 1M vocab and HN embeddings")
    parser.add_argument("--multiple_merges_exp", action="store_true", help="Use HN embeddings with different sequence-reduction percentages")
    parser.add_argument("--use_original_emb_for_choices", action="store_true", help="Use original embeddings for A, B, C, D choices")
    parser.add_argument("--merges", type=int, default=1000, help="Number of BPE merges for dynamic_bpe (ignored for other exp_types)")

    args = parser.parse_args()
    setup_seed(1234)

    if not args.no_wandb:
        exp_prefix = args.exp_prefix + "_" if args.exp_prefix else ""
        wandb.init(
            project="dynamic-tokenization",
            config={"dataset": "KMMLU", "exp_type": args.exp_type, "eval_type": args.eval_type, "lng": args.lng},
            name=f"{exp_prefix}KMMLU_Mistral_{args.exp_type}_{args.eval_type}_five_shot_{args.five_shot}_subject_{args.ds_subject}_batch_size_{args.batch_size}_1M_Vocab_{args.vocab_1M}",
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "mistralai/Mistral-7B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)

    hypernet = None
    base_model = None
    lang_index = None
    source_embeddings = None
    datasetEncoder = None
    inout_1M_embeddings = None
    if args.exp_type in ["original_tk_hypernet", "lp_tk_hypernet", "dynamic_bpe", "word_tk_hypere"]:
        hypernet = AutoModel.from_pretrained("benjamin/zett-hypernetwork-Mistral-7B-v0.1", trust_remote_code=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained("benjamin/zett-hypernetwork-Mistral-7B-v0.1")
        if args.exp_type == "lp_tk_hypernet":
            if args.vocab_1M:
                inout_1M_embeddings = torch.load("decoders/data/1M_vocab_embeddings/large_HN_embeddings.pt")
                vocab_new_path = "decoders/data/large_tokenizer/vocab.json"
                merges_path = "decoders/data/large_tokenizer/merges.txt"
                vocab, merges = BPE.read_file(vocab_new_path, merges_path)
            else:
                vocab = tokenizer.get_vocab()
            unk_token = tokenizer.unk_token if tokenizer.unk_token is not None else tokenizer.eos_token
            tokenizer._tokenizer.model = WordPiece(vocab, unk_token=unk_token)
            tokenizer._tokenizer.model.continuing_subword_prefix = ""
        langs = [x.strip() for x in open(f"{HOME_PATH}/data/artifacts/26l.txt")]
        lang_index = torch.tensor(langs.index(args.lng), dtype=torch.int32).to(device)
        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        source_embeddings = torch.concatenate([
            base_model.get_input_embeddings().weight.data,
            base_model.get_output_embeddings().weight.data,
        ], axis=1).to(device)
        embeddings_cache = LRU_Cache(
            cache_size=5000,
            emb_size=base_model.get_input_embeddings().weight.data.shape[1],
            device=device,
        )
        datasetEncoder = DatasetEncoder(
            hypernet=hypernet,
            tokenizer=tokenizer,
            device=device,
            lang_index=lang_index,
            surface_form_maxlen=7,
            source_embeddings=source_embeddings,
            embeddings_cache=embeddings_cache,
            exp_type=args.exp_type,
            collect_extra_data=True,
            bpe_tokenizer_boundary="pretokens",
        )

    test_dataset, validation_dataset, per_subject_validation_datasets, subjects_used = load_kmmlu_splits(args.ds_subject)
    if not args.same_domain_shot:
        # match the MMLU script: only build per-subject dev sets when needed
        per_subject_validation_datasets_for_dataset = {}
    else:
        per_subject_validation_datasets_for_dataset = per_subject_validation_datasets

    kmmlu_dataset = KMMLUDataset(
        test_dataset,
        validation_dataset=validation_dataset,
        validation_datasets=per_subject_validation_datasets_for_dataset,
    )
    kmmlu_dataset.five_shot = args.five_shot
    kmmlu_dataset.same_domain_shot = args.same_domain_shot
    dataloader = DataLoader(kmmlu_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    evaluate_model(
        dataloader=dataloader,
        model=model,
        tokenizer=tokenizer,
        args=args,
        base_model=base_model,
        hypernet=hypernet,
        lang_index=lang_index,
        source_embeddings=source_embeddings,
        datasetEncoder=datasetEncoder,
        inout_1M_embeddings=inout_1M_embeddings,
        subjects=subjects_used,
    )


if __name__ == "__main__":
    main()
