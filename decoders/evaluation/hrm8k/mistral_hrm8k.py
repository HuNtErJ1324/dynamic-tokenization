"""
HRM8K Evaluation Script for Mistral-7B.

Evaluates the model on English and Korean versions of the same math problems to
measure the English-Korean reasoning gap. A widening gap under dynamic tokenization
suggests the problem is Korean input comprehension, not reasoning ability.

Dataset: HAERAE-HUB/HRM8K
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch
import argparse
import sys
from torch.utils.data import DataLoader
from evaluation_utils import (
    HRM8KDataset,
    collate_fn,
    setup_seed,
    evaluate_model,
    load_hrm8k,
)

from pathlib import Path
HOME_PATH = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, HOME_PATH)


def main():
    parser = argparse.ArgumentParser(description="HRM8K Evaluation for Mistral-7B")
    parser.add_argument(
        "--lang", type=str, default="both",
        help="Language to evaluate: en | ko | both",
    )
    parser.add_argument(
        "--exp_type", type=str, default="plain",
        help="plain | entropy_split | dynamic_bpe | dynamic_bpe_entropy_split",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--five_shot", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--exp_prefix", type=str, default="")
    parser.add_argument("--max_len", type=int, default=2048,
                        help="Max input length (shorter than MMLU since math prompts are long).")
    parser.add_argument("--max_new_tokens", type=int, default=64,
                        help="Max tokens to generate per answer.")
    parser.add_argument("--num_shots", type=int, default=3,
                        help="Number of few-shot examples (0 for zero-shot).")
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument(
        "--entropy_threshold", type=float, default=3.0,
        help="Entropy threshold for entropy_split and dynamic_bpe_entropy_split.",
    )
    parser.add_argument(
        "--lng", type=str, default="ko",
        help="Language index for hypernetwork (en | ko). Used for dynamic_bpe modes.",
    )
    parser.add_argument(
        "--merges", type=int, default=1000,
        help="Number of BPE merges for dynamic_bpe (ignored for plain/entropy_split).",
    )
    parser.add_argument(
        "--bpe_tokenizer_boundary", type=str, default="pretokens",
        help="Merge boundary for dynamic_bpe: pretokens, word, word_hyphen, sentence, superbpe.",
    )
    parser.add_argument(
        "--transition_point", type=int, default=0,
        help="SuperBPE two-stage schedule: first N merges use 'pretokens', rest use --bpe_tokenizer_boundary.",
    )

    args = parser.parse_args()
    setup_seed(1234)

    if not args.no_wandb:
        import wandb
        exp_prefix = args.exp_prefix + "_" if args.exp_prefix else ""
        wandb.init(
            project="dynamic-tokenization",
            config={"dataset": "HRM8K", "exp_type": args.exp_type, "lang": args.lang},
            name=f"{exp_prefix}HRM8K_Mistral_{args.lang}_five_shot_{args.five_shot}",
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "mistralai/Mistral-7B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)

    # Hypernetwork setup for dynamic_bpe modes
    langs_list = None
    datasetEncoder = None
    if args.exp_type in ["dynamic_bpe", "dynamic_bpe_entropy_split"]:
        from tokenizations.tokenization_utils import DatasetEncoder
        from tokenizations.hypernet_cache import LRU_Cache

        hypernet_model_id = "benjamin/zett-hypernetwork-Mistral-7B-v0.1"
        print(f"[hypernetwork] Loading {hypernet_model_id} ...", flush=True)
        hypernet = AutoModel.from_pretrained(
            hypernet_model_id, trust_remote_code=True
        ).to(device)
        hypernet_tokenizer = AutoTokenizer.from_pretrained(hypernet_model_id)

        langs_list = [x.strip() for x in open(f"{HOME_PATH}/data/artifacts/26l.txt")]
        lang_index = torch.tensor(langs_list.index(args.lng), dtype=torch.int32).to(device)

        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        source_embeddings = torch.concatenate([
            base_model.get_input_embeddings().weight.data,
            base_model.get_output_embeddings().weight.data,
        ], axis=1).to(device)
        del base_model

        embeddings_cache = LRU_Cache(
            cache_size=5000,
            emb_size=model.get_input_embeddings().weight.data.shape[1],
            device=device,
        )
        datasetEncoder = DatasetEncoder(
            hypernet=hypernet,
            tokenizer=hypernet_tokenizer,
            device=device,
            lang_index=lang_index,
            surface_form_maxlen=7,
            source_embeddings=source_embeddings,
            embeddings_cache=embeddings_cache,
            exp_type=args.exp_type,
            collect_extra_data=True,
            bpe_tokenizer_boundary=args.bpe_tokenizer_boundary,
            transition_point=args.transition_point,
        )
        print(
            f"[hypernetwork] Ready. lang_index={args.lng} ({lang_index.item()})",
            flush=True,
        )

    langs_to_run = ["en", "ko"] if args.lang == "both" else [args.lang]

    results = {}
    for lang in langs_to_run:
        # Per-language lang_index update for dynamic_bpe modes
        if datasetEncoder is not None and langs_list is not None:
            try:
                new_lang_idx = torch.tensor(
                    langs_list.index(lang), dtype=torch.int32
                ).to(device)
                datasetEncoder.lang_index = new_lang_idx
                print(
                    f"[hypernetwork] lang_index updated → {lang} ({new_lang_idx.item()})",
                    flush=True,
                )
            except ValueError:
                pass  # keep existing lang_index if lang not in list

        dataset = load_hrm8k(lang=lang)
        print(f"\n{'='*60}", flush=True)
        print(f"Evaluating HRM8K [{lang}]  exp_type={args.exp_type}", flush=True)
        print(f"{'='*60}", flush=True)

        hrm8k_dataset = HRM8KDataset(
            dataset, lang=lang,
            five_shot=(args.num_shots > 0),
            num_shots=args.num_shots,
        )
        dataloader = DataLoader(
            hrm8k_dataset, batch_size=args.batch_size,
            shuffle=False, collate_fn=collate_fn,
        )

        acc = evaluate_model(
            dataloader=dataloader,
            model=model,
            tokenizer=tokenizer,
            args=args,
            lang=lang,
            datasetEncoder=datasetEncoder,
        )
        results[lang] = acc

    print("\n" + "="*60, flush=True)
    print("HRM8K Summary", flush=True)
    print("="*60, flush=True)
    for lang, acc in results.items():
        print(f"  [{lang}] accuracy: {acc:.4f}", flush=True)
    if "en" in results and "ko" in results:
        gap = results["en"] - results["ko"]
        print(f"  English-Korean gap: {gap:+.4f} (positive = English higher)", flush=True)


if __name__ == "__main__":
    main()
