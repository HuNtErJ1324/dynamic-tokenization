"""
KoBEST Evaluation Script for Mistral-7B.

Evaluates plain / hypernetwork / dynamic-BPE tokenization on all 5 KoBEST tasks
(BoolQ, COPA, WiC, HellaSwag, SentiNeg) using the same logit-based approach as
mistral_kmmlu.py.

Dataset: skt/kobest_v1
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch
import argparse
import sys
from torch.utils.data import DataLoader
from evaluation_utils import (
    KoBESTDataset,
    KOBEST_TASKS,
    collate_fn,
    setup_seed,
    evaluate_model,
    load_kobest_splits,
)
from tokenizers.models import BPE, WordPiece

from pathlib import Path
HOME_PATH = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, HOME_PATH)


def main():
    parser = argparse.ArgumentParser(description="KoBEST Evaluation for Mistral-7B")
    parser.add_argument(
        "--task", type=str, default="all",
        help=f"KoBEST task to evaluate: all | {' | '.join(KOBEST_TASKS)}",
    )
    parser.add_argument("--exp_type", type=str, default="plain",
                        help="plain | original_tk_hypernet | dynamic_bpe")
    parser.add_argument("--eval_type", type=str, default="original")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--five_shot", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--exp_prefix", type=str, default="")
    parser.add_argument("--lng", type=str, default="ko")
    parser.add_argument("--max_len", type=int, default=4096)
    parser.add_argument("--merges", type=int, default=1000)
    parser.add_argument(
        "--bpe_tokenizer_boundary", type=str, default="pretokens",
        help="pretokens | word | word_hyphen | sentence | superbpe",
    )
    parser.add_argument(
        "--transition_point", type=int, default=0,
        help="SuperBPE warm-up: first N merges use pretokens boundary.",
    )
    parser.add_argument("--use_original_emb_for_choices", action="store_true")
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument(
        "--entropy_threshold", type=float, default=3.0,
        help="Entropy threshold for entropy_split exp_type. Tokens with entropy above this are split.",
    )
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Limit number of examples per task (for debugging).")

    args = parser.parse_args()
    setup_seed(1234)

    tasks_to_run = KOBEST_TASKS if args.task == "all" else [args.task]
    assert all(t in KOBEST_TASKS for t in tasks_to_run), \
        f"Unknown task. Choose from: {KOBEST_TASKS}"

    if not args.no_wandb:
        import wandb
        exp_prefix = args.exp_prefix + "_" if args.exp_prefix else ""
        wandb.init(
            project="dynamic-tokenization",
            config={"dataset": "KoBEST", "exp_type": args.exp_type, "task": args.task},
            name=f"{exp_prefix}KoBEST_{args.task}_Mistral_{args.exp_type}_five_shot_{args.five_shot}",
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
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
    if args.exp_type in ["original_tk_hypernet", "lp_tk_hypernet", "dynamic_bpe",
                          "dynamic_bpe_entropy_split"]:
        from tokenizations.tokenization_utils import DatasetEncoder
        from tokenizations.hypernet_cache import LRU_Cache
        hypernet = AutoModel.from_pretrained(
            "benjamin/zett-hypernetwork-Mistral-7B-v0.1", trust_remote_code=True
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained("benjamin/zett-hypernetwork-Mistral-7B-v0.1")
        langs = [x.strip() for x in open(f"{HOME_PATH}/data/artifacts/26l.txt")]
        lang_index = torch.tensor(langs.index(args.lng), dtype=torch.int32).to(device)
        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        source_embeddings = torch.concatenate([
            base_model.get_input_embeddings().weight.data,
            base_model.get_output_embeddings().weight.data,
        ], axis=1).to(device)
        emb_size = base_model.get_input_embeddings().weight.data.shape[1]
        del base_model
        base_model = None
        embeddings_cache = LRU_Cache(
            cache_size=5000,
            emb_size=emb_size,
            device=device,
        )
        exp_type_for_encoder = (
            "dynamic_bpe" if args.exp_type == "dynamic_bpe_entropy_split"
            else args.exp_type
        )
        datasetEncoder = DatasetEncoder(
            hypernet=hypernet,
            tokenizer=tokenizer,
            device=device,
            lang_index=lang_index,
            surface_form_maxlen=7,
            source_embeddings=source_embeddings,
            embeddings_cache=embeddings_cache,
            exp_type=exp_type_for_encoder,
            collect_extra_data=True,
            bpe_tokenizer_boundary=args.bpe_tokenizer_boundary,
            transition_point=args.transition_point,
        )

    all_results = {}
    for task in tasks_to_run:
        print(f"\n{'='*60}", flush=True)
        print(f"Evaluating KoBEST task: {task}", flush=True)
        print(f"{'='*60}", flush=True)

        test_dataset, validation_dataset = load_kobest_splits(task)
        dataset = KoBESTDataset(
            test_dataset, validation_dataset, task=task, num_shots=5
        )
        dataset.five_shot = args.five_shot

        if args.max_examples is not None:
            from torch.utils.data import Subset
            dataset = Subset(dataset, range(min(args.max_examples, len(dataset))))
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
        )

        acc = evaluate_model(
            dataloader=dataloader,
            model=model,
            tokenizer=tokenizer,
            args=args,
            base_model=base_model,
            hypernet=hypernet,
            lang_index=lang_index,
            source_embeddings=source_embeddings,
            datasetEncoder=datasetEncoder,
        )
        all_results[task] = acc

    print("\n" + "="*60, flush=True)
    print("KoBEST Summary", flush=True)
    print("="*60, flush=True)
    for task, acc in all_results.items():
        print(f"  {task}: {acc:.4f}", flush=True)
    if all_results:
        avg = sum(all_results.values()) / len(all_results)
        print(f"  Average: {avg:.4f}", flush=True)


if __name__ == "__main__":
    main()
