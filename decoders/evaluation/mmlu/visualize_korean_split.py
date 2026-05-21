"""
한국어 문장에서 entropy 기반 token splitting 시각화.

Usage:
    python visualize_korean_split.py
    python visualize_korean_split.py --threshold 4.0
"""
import torch
import argparse
import sys
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from split_utils import process_prompts_with_split, minimal_split

KOREAN_SENTENCES = [
    "심혈관 질환의 예방을 위해 규칙적인 운동이 필요합니다.",
    "인공지능 기반의 진단 시스템이 빠르게 발전하고 있습니다.",
    "자동차가 시속 60km로 2.5시간 동안 달리면 이동한 거리는 얼마입니까?",
    "직사각형의 가로가 8cm이고 세로가 5cm일 때, 직사각형의 넓이는 얼마입니까?",
    "한 가게에서 사과는 개당 2,000원, 오렌지는 개당 3,000원에 팔고 있습니다.",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=3.0)
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading {args.model}...")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()

    print(f"\nEntropy threshold: {args.threshold}")
    process_prompts_with_split(
        model, tokenizer, KOREAN_SENTENCES, minimal_split,
        entropy_threshold=args.threshold,
        device=device,
        verbose=True,
    )


if __name__ == "__main__":
    main()
