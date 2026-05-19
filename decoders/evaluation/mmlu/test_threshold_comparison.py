"""
Function to test a single prompt with different entropy thresholds.
Demonstrates token splitting behavior and LLM response generation at each threshold.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from split_utils import process_prompts_with_split, minimal_split

def _letter_token_id(tokenizer, letter: str) -> int:
    """Single-token ID for ' <letter>' (preferred) or '<letter>'."""
    for cand in (f" {letter}", letter):
        ids = tokenizer.encode(cand, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    return tokenizer.encode(f" {letter}", add_special_tokens=False)[0]

def display_token_sequence(tokens, token_ids, max_width=100):
    """
    Clean token visualization using slashes.
    No indices, no whitespace markers, minimal noise.
    """

    output = []
    current_line = ""
    current_length = 0

    for token_str in tokens:

        # Minimal cleanup of special tokens only
        if token_str in {"[CLS]", "[SEP]"}:
            display_str = token_str.strip("[]")
        elif token_str in {"<s>", "</s>", "<unk>", "[PAD]"}:
            display_str = token_str.replace("<s>", "BOS") \
                                  .replace("</s>", "EOS") \
                                  .replace("<unk>", "UNK") \
                                  .replace("[PAD]", "PAD")
        else:
            display_str = token_str

        token_display = display_str
        token_length = len(token_display) + 3  # " / "

        # wrap logic
        if current_length + token_length > max_width and current_line:
            output.append(current_line.rstrip(" /"))
            current_line = token_display + " / "
            current_length = token_length
        else:
            current_line += token_display + " / "
            current_length += token_length

    if current_line:
        output.append(current_line.rstrip(" /"))

    return "\n".join(output)

def compute_entropy(logits):
    # logits: (batch, seq, vocab)
    last_logits = logits[:, -1, :]  # (batch, vocab)

    probs = torch.softmax(last_logits, dim=-1)
    log_probs = torch.log_softmax(last_logits, dim=-1)

    entropy = -(probs * log_probs).sum(dim=-1)

    return entropy.item()

@torch.no_grad()
def test_prompt_with_different_thresholds(
    prompt,
    thresholds=None,
    split_fn=minimal_split,
    model_id="mistralai/Mistral-7B-v0.1",
    device=None
):
    """
    Test a single prompt with different entropy thresholds for token splitting.
    
    Args:
        prompt: The input prompt string
        thresholds: List of entropy thresholds to test (default: [2.0, 3.0, 4.0, 5.0])
        split_fn: Function to split high-entropy tokens (default: minimal_split)
        model_id: HuggingFace model ID (default: Mistral-7B)
        device: Compute device (default: auto-detect CUDA)
    
    Returns:
        Dictionary with results for each threshold
    """
    if thresholds is None:
        thresholds = [1.0, 2.0, 3.0, 4.0, 7.0, 10.0]
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Device: {device}\n")
    print(f"Loading model {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    
    print(f"Model loaded.\n")
    print("=" * 80)
    print("PROMPT")
    print("=" * 80)
    print(f"{prompt}\n")
    
    # Get original tokenization (no splitting)
    original_ids = tokenizer.encode(prompt, add_special_tokens=True)
    original_tokens = tokenizer.convert_ids_to_tokens(original_ids)
    print("=" * 80)
    print("ORIGINAL TOKENIZATION (no splitting)")
    print("=" * 80)
    print(f"Token count: {len(original_ids)}\n")
    print(display_token_sequence(original_tokens, original_ids))
    print()
    
    results = {"original": {
        "token_count": len(original_ids),
        "tokens": original_tokens,
        "token_ids": original_ids,
    }}
    
    choice_token_ids = torch.tensor(
        [_letter_token_id(tokenizer, L) for L in ("A", "B", "C", "D")], device=device
    )

    # Test each threshold
    for threshold in thresholds:
        print("=" * 80)
        print(f"THRESHOLD: {threshold}")
        print("=" * 80)
        
        # Apply splitting based on entropy
        split_ids = process_prompts_with_split(
            model=model,
            tokenizer=tokenizer,
            prompts=[prompt],
            split_fn=split_fn,
            entropy_threshold=threshold,
            device=device
        )
        
        split_ids_list = split_ids[0]  # Get first (and only) prompt
        split_tokens = tokenizer.convert_ids_to_tokens(split_ids_list)
        
        print(f"Token count after split: {len(split_ids_list)}")
        inflation_pct = 100 * (len(split_ids_list) - len(original_ids)) / len(original_ids)
        print(f"Token count: {len(original_ids)} → {len(split_ids_list)} " +
              f"({inflation_pct:+.1f}% inflation)\n")
        print(display_token_sequence(split_tokens, split_ids_list))
        print()
        
        # Generate response using split token IDs
        print("Generating answer...")
        with torch.no_grad():
            # Tokenize and pad
            enc = tokenizer.pad(
                {"input_ids": [torch.tensor(ids) for ids in split_ids]},
                padding=True,
                return_tensors="pt"
            ).to(device)
            
            logits = model(**enc).logits
            last_logits = logits[:, -1, :]
            choice_logits = last_logits[0, choice_token_ids]
            pred = choice_logits.argmax(dim=-1).item()

            answer = ["A", "B", "C", "D"][pred]
            # probabilities over full vocab for last token
            probs = torch.softmax(last_logits, dim=-1)

            # extract A/B/C/D probabilities
            option_probs = probs[0, choice_token_ids]

            pA, pB, pC, pD = option_probs.tolist()
            print(f"Answer: {answer} (p(A)={pA:.4f}, p(B)={pB:.4f}, p(C)={pC:.4f}, p(D)={pD:.4f})")
        
        results[f"threshold_{threshold}"] = {
            "threshold": threshold,
            "token_count": len(split_ids_list),
            "tokens": split_tokens,
            "token_ids": split_ids_list,
            "answer": answer,
            "inflation_percent": 100 * (len(split_ids_list) - len(original_ids)) / len(original_ids)
        }
    
    # Print summary table
    print("=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Threshold':<15} {'Token Count':<15} {'Inflation %':<15}")
    print("-" * 45)
    print(f"{'Original':<15} {len(original_ids):<15} {'N/A':<15}")
    
    for threshold in thresholds:
        key = f"threshold_{threshold}"
        if key in results:
            r = results[key]
            print(f"{threshold:<15.1f} {r['token_count']:<15} {r['inflation_percent']:+14.1f}%")
    
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    # Example usage
    test_prompt = """The following are multiple choice questions (with answers) about high school biology.

In plants, the tendency of climbing vines to twine their tendrils around a trellis is called
A. thigmotropism
B. hydrotropism
C. phototropism
D. geotropism
Answer:"""
    
    results = test_prompt_with_different_thresholds(
        prompt=test_prompt
    )