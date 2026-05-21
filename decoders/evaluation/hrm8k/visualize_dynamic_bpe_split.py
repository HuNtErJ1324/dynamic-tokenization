"""
Dynamic BPE + Entropy Split 시각화 스크립트.

각 파이프라인 단계에서 토큰이 어떻게 변하는지 보여줍니다:
  Stage 0: 원본 Mistral 토크나이저 (plain)
  Stage 1: Dynamic BPE 적용 후 (merged tokens)
  Stage 2: 엔트로피 계산 (각 merged token의 entropy 값)
  Stage 3: 고엔트로피 토큰 splitting 후 (final tokens)

Usage:
    python visualize_dynamic_bpe_split.py
    python visualize_dynamic_bpe_split.py --merges 500 --threshold 3.0
    python visualize_dynamic_bpe_split.py --lang en
"""

import torch
import argparse
import sys
from pathlib import Path

HOME_PATH = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, HOME_PATH)

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

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


def _decode(tok):
    t = tok.replace('Ġ', ' ').replace('▁', ' ')
    try:
        bts = bytes([_GPT2_BYTE_DECODER.get(c, ord(c) & 0xFF) for c in t])
        return bts.decode('utf-8')
    except (UnicodeDecodeError, KeyError):
        return tok


def _gpt2_encode_text(text: str) -> str:
    return ''.join(_GPT2_BYTE_ENCODER[b] for b in text.encode('utf-8'))


def _char_level_split(tok: str):
    """GPT-2 merged token → 글자 단위 GPT-2 토큰 리스트. tokenize() 미사용."""
    text = _decode(tok)
    if text == tok:
        return [tok]
    chars = list(text)
    if len(chars) <= 1:
        return [tok]
    result = []
    i = 0
    if chars[0] == ' ' and len(chars) > 1:
        result.append(_gpt2_encode_text(' ' + chars[1]))
        i = 2
    while i < len(chars):
        result.append(_gpt2_encode_text(chars[i]))
        i += 1
    return result if result else [tok]


KOREAN_SENTENCES = [
    "심혈관 질환의 예방을 위해 규칙적인 운동이 필요합니다.",
    "인공지능 기반의 진단 시스템이 빠르게 발전하고 있습니다.",
    "자동차가 시속 60km로 2.5시간 동안 달리면 이동한 거리는 얼마입니까?",
    "직사각형의 가로가 8cm이고 세로가 5cm일 때, 직사각형의 넓이는 얼마입니까?",
    "한 가게에서 사과는 개당 2,000원, 오렌지는 개당 3,000원에 팔고 있습니다.",
]

ENGLISH_SENTENCES = [
    "Cardiovascular disease prevention requires regular physical exercise.",
    "AI-based diagnostic systems are advancing rapidly in medical fields.",
    "If a car travels at 60km/h for 2.5 hours, how far does it travel?",
    "A rectangle has length 8cm and width 5cm. What is its area?",
    "A store sells apples for $2 each and oranges for $3 each.",
]


@torch.no_grad()
def visualize_pipeline(
    model,
    tokenizer,
    hypernet_tokenizer,
    datasetEncoder,
    sentences,
    merges=1000,
    entropy_threshold=3.0,
    device="cuda",
):
    print(f"\n{'='*80}")
    print(f"Dynamic BPE + Entropy Split Visualization")
    print(f"  merges={merges}  threshold={entropy_threshold}")
    print(f"{'='*80}")

    for sent_idx, sentence in enumerate(sentences):
        print(f"\n{'─'*80}")
        print(f"[{sent_idx}] {sentence}")
        print(f"{'─'*80}")

        # Stage 0: Plain Mistral tokenization
        plain_tokens = tokenizer.tokenize(sentence)
        print(f"\nStage 0 — Plain Mistral  ({len(plain_tokens)} tokens):")
        print(f"  {plain_tokens}")

        # Stage 1: Dynamic BPE encoding (single example)
        encoded = datasetEncoder.encode_examples_unique_tokens_lru(
            examples=[sentence],
            max_length=2048,
            merges=merges,
            task="mmlu",
        )
        inputs_embeds = encoded["inputs_embeds"].to(torch.bfloat16)   # (1, seq, emb)
        attention_mask = encoded["attention_mask"]                     # (1, seq)
        batch_tokens = encoded["batch_tokens"]                         # list of token strings

        # batch_tokens[0] is the merged token sequence (no padding)
        merged_tokens = batch_tokens[0]
        readable = [_decode(t) for t in merged_tokens]
        print(f"\nStage 1 — Dynamic BPE    ({len(merged_tokens)} tokens, {len(plain_tokens)-len(merged_tokens):+d}):")
        print(f"  raw:      {merged_tokens}")
        print(f"  readable: {readable}")

        # Stage 2: Entropy scan over merged embeddings
        hidden = model.model(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask
        ).last_hidden_state
        logits = model.lm_head(hidden).float()
        probs = torch.softmax(logits, dim=-1)
        entropy_row = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)[0].cpu()  # (seq,)

        # Align batch_tokens with attention_mask positions
        attn = attention_mask[0].tolist()
        pad_offset = attn.index(1) if 1 in attn else 0

        print(f"\nStage 2 — Entropy per merged token (threshold={entropy_threshold}):")
        print(f"  {'token':<30} {'entropy':>8}  action")
        print(f"  {'-'*55}")

        # Stage 3: Split high-entropy tokens
        final_tokens = []
        for j, tok in enumerate(merged_tokens):
            pos = pad_offset + j
            ent = entropy_row[pos].item() if pos < len(entropy_row) else 0.0
            if ent > entropy_threshold and tok not in hypernet_tokenizer.all_special_tokens:
                sub = _char_level_split(tok)
                final_tokens.extend(sub)
                print(f"  {repr(tok):<30} {ent:>8.3f}  SPLIT → {[_decode(s) for s in sub]}")
            else:
                final_tokens.append(tok)
                marker = "keep" if ent <= entropy_threshold else "keep (special)"
                print(f"  {repr(tok):<30} {ent:>8.3f}  {marker}")

        delta_bpe = len(plain_tokens) - len(merged_tokens)
        delta_split = len(final_tokens) - len(merged_tokens)
        readable_final = [_decode(t) for t in final_tokens]
        print(f"\nStage 3 — After split    ({len(final_tokens)} tokens, {delta_split:+d} from Stage 1):")
        print(f"  raw:      {final_tokens}")
        print(f"  readable: {readable_final}")
        print(f"\n  Summary: plain={len(plain_tokens)} → dynamic_bpe={len(merged_tokens)} ({delta_bpe:+d}) "
              f"→ after_split={len(final_tokens)} ({delta_split:+d} from BPE)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merges", type=int, default=1000)
    parser.add_argument("--threshold", type=float, default=3.0)
    parser.add_argument("--lang", type=str, default="ko", choices=["ko", "en"])
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--hypernet", type=str, default="benjamin/zett-hypernetwork-Mistral-7B-v0.1")
    parser.add_argument("--lng", type=str, default="ko",
                        help="Language index for hypernetwork.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading base model: {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()

    print(f"Loading hypernetwork: {args.hypernet} ...")
    hypernet = AutoModel.from_pretrained(args.hypernet, trust_remote_code=True).to(device)
    hypernet_tokenizer = AutoTokenizer.from_pretrained(args.hypernet)

    langs_list = [x.strip() for x in open(f"{HOME_PATH}/data/artifacts/26l.txt")]
    lang_index = torch.tensor(langs_list.index(args.lng), dtype=torch.int32).to(device)

    base_model = AutoModelForCausalLM.from_pretrained(args.model)
    source_embeddings = torch.concatenate([
        base_model.get_input_embeddings().weight.data,
        base_model.get_output_embeddings().weight.data,
    ], axis=1).to(device)
    del base_model

    from tokenizations.tokenization_utils import DatasetEncoder
    from tokenizations.hypernet_cache import LRU_Cache

    embeddings_cache = LRU_Cache(
        cache_size=2000,
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
        exp_type="dynamic_bpe",
        collect_extra_data=True,
        bpe_tokenizer_boundary="pretokens",
        transition_point=0,
    )

    sentences = KOREAN_SENTENCES if args.lang == "ko" else ENGLISH_SENTENCES

    visualize_pipeline(
        model=model,
        tokenizer=tokenizer,
        hypernet_tokenizer=hypernet_tokenizer,
        datasetEncoder=datasetEncoder,
        sentences=sentences,
        merges=args.merges,
        entropy_threshold=args.threshold,
        device=device,
    )


if __name__ == "__main__":
    main()
