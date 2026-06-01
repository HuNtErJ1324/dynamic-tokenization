"""
Pre-download datasets (and optionally models) ONCE so sweep jobs can run with
HF_HUB_OFFLINE=1 and never hit the HuggingFace API.

Why: every KMMLU job calls load_dataset("HAERAE-HUB/KMMLU", subject) for all ~45
subjects — that's ~45 API calls per job. Firing many KMMLU cells at once gets the
cluster IP rate-limited (HTTP 429). Caching once + running offline removes the API
calls entirely (jobs read local disk).

Usage (on a node WITH internet — a login node — authenticated):
    export HF_TOKEN=hf_xxx                 # https://huggingface.co/settings/tokens
    python scripts/precache_data.py            # datasets + tokenizers
    python scripts/precache_data.py --models   # also the Mistral + zett models (~14GB)

Then submit sweeps so the jobs use the cache and skip the API:
    export HF_TOKEN=hf_xxx
    export HF_HUB_OFFLINE=1
    bash scripts/run_sweep.sh sweeps/S2_threshold.tsv --submit

Note: the cache lives under $HF_HOME (default ~/.cache/huggingface). Make sure that
path is on shared storage so compute nodes see what the login node downloaded.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _with_retry(fn, label, attempts=5, base=5):
    for i in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - want to retry on any HF/network error
            if i == attempts:
                print(f"  [FAIL] {label}: {type(e).__name__}: {e}", flush=True)
                return None
            wait = base * i
            print(f"  [retry {i}/{attempts}] {label}: {type(e).__name__}; sleeping {wait}s", flush=True)
            time.sleep(wait)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", action="store_true",
                        help="Also cache the Mistral-7B + zett hypernetwork weights (~14GB).")
    args = parser.parse_args()

    # Precaching MUST run online — refuse if offline mode is on (the classic gotcha:
    # exporting HF_HUB_OFFLINE=1 for the job submit, then running precache in the same
    # shell, which downloads nothing).
    if os.environ.get("HF_HUB_OFFLINE", "0") not in ("0", "", "false", "False") or \
       os.environ.get("HF_DATASETS_OFFLINE", "0") not in ("0", "", "false", "False"):
        print("ERROR: HF_HUB_OFFLINE / HF_DATASETS_OFFLINE is set, but precaching needs the "
              "network. Run:  unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE  then retry.", flush=True)
        return 1

    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")):
        print("[warn] No HF_TOKEN in env — anonymous requests are aggressively rate-limited. "
              "Create one at https://huggingface.co/settings/tokens and `export HF_TOKEN=...` "
              "before running this.", flush=True)

    # Print the exact cache dir so you can confirm the SLURM jobs read the same one
    # (HF_HOME must be identical, and on shared /gpfs, for login-node caching to reach
    # compute nodes).
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        print(f"[cache] HF_HOME={os.environ.get('HF_HOME', '(default)')}  hub_cache={HF_HUB_CACHE}", flush=True)
    except Exception:
        pass

    from datasets import DatasetDict, get_dataset_config_names, load_dataset

    local = REPO_ROOT / "data" / "hf_local"
    print(f"== datasets -> {local} (eval reads these via load_from_disk: no Hub, fully offline) ==", flush=True)

    print("== cais/mmlu (config 'all') ==", flush=True)
    mmlu = _with_retry(lambda: load_dataset("cais/mmlu", "all"), "cais/mmlu")
    if mmlu is not None:
        DatasetDict({"test": mmlu["test"], "validation": mmlu["validation"]}).save_to_disk(str(local / "mmlu_all"))
        print(f"  saved -> {local / 'mmlu_all'}", flush=True)

    print("== HAERAE-HUB/KMMLU (all subjects) ==", flush=True)
    subjects = _with_retry(lambda: get_dataset_config_names("HAERAE-HUB/KMMLU"),
                           "KMMLU config list")
    if subjects:
        print(f"  {len(subjects)} subjects", flush=True)
        for i, s in enumerate(subjects, 1):
            ds = _with_retry(lambda s=s: load_dataset("HAERAE-HUB/KMMLU", s), f"KMMLU/{s}")
            if ds is not None:
                DatasetDict({"test": ds["test"], "dev": ds["dev"]}).save_to_disk(str(local / "kmmlu" / s))
            print(f"  [{i}/{len(subjects)}] {s}: {'ok' if ds is not None else 'FAILED'}", flush=True)
    else:
        print("  [FAIL] could not list KMMLU subjects (still throttled?). "
              "Wait ~15 min for the IP limit to clear, then rerun with HF_TOKEN set.", flush=True)

    print("== caching tokenizers ==", flush=True)
    from transformers import AutoTokenizer
    _with_retry(lambda: AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1"), "mistral tokenizer")
    _with_retry(lambda: AutoTokenizer.from_pretrained("benjamin/zett-hypernetwork-Mistral-7B-v0.1"),
                "zett tokenizer")

    if args.models:
        print("== caching models (~14GB) ==", flush=True)
        from transformers import AutoModel, AutoModelForCausalLM
        _with_retry(lambda: AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1"),
                    "Mistral-7B weights")
        _with_retry(lambda: AutoModel.from_pretrained(
            "benjamin/zett-hypernetwork-Mistral-7B-v0.1", trust_remote_code=True), "zett hypernet")
    else:
        print("(skipping model weights — pass --models if offline jobs can't find them; "
              "prior decoder runs usually already cached Mistral-7B + zett.)", flush=True)

    print(f"\nDone. Datasets saved under {REPO_ROOT / 'data' / 'hf_local'} (read via load_from_disk — "
          "no Hub access). Submit with:  export HF_HUB_OFFLINE=1 (for the cached model/zett); "
          "bash scripts/run_sweep.sh <spec> --submit", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
