import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

@torch.no_grad()
def process_prompts_with_split(model, tokenizer, prompts, split_fn, entropy_threshold=3.0, device="cuda"):
    """
    Takes raw strings, calculates entropy in a single batch, 
    and applies a splitting strategy to high-entropy tokens.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- PASS 1: SCAN ---
    # Encode the full strings for the entropy check
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    
    # We also need the un-padded version to iterate through later 
    # (so we don't process padding tokens)
    encoded_prompts = [tokenizer.encode(p, add_special_tokens=True) for p in prompts]

    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    # Move to CPU immediately to free up GPU VRAM and perform string ops
    entropy_matrix = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).cpu()
    print(f'finished forward pass')
    # --- PASS 2: APPLY SPLIT ---
    processed_prompts = []
    
    for i, original_ids in enumerate(encoded_prompts):
        # Initialize with the first token (usually BOS)
        new_prompt = [original_ids[0]]
        
        # Check logit at position j, split token at position j if needed
        for j in range(1, len(original_ids)):
            next_token_id = original_ids[j]
            token_entropy = entropy_matrix[i, j].item()
            if token_entropy > entropy_threshold:
                # Apply your modular splitting strategy
                fragments = split_fn(tokenizer, next_token_id)
                new_prompt.extend(fragments)
            else:
                new_prompt.append(next_token_id)
                
        processed_prompts.append(new_prompt)
        
    return processed_prompts

def mask_last_char_split(tokenizer, token_id):
    # 1. Get the string (e.g., "Paris")
    text = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
    
    # 2. Safety check: Don't split single characters
    if len(text.strip()) <= 3:
        return [token_id]
        
    # 3. Force the split by encoding two separate strings
    # This prevents the greedy algorithm from finding "Paris"
    prefix_ids = tokenizer.encode(text[:-1], add_special_tokens=False)
    suffix_ids = tokenizer.encode(text[-1], add_special_tokens=False)
    
    return prefix_ids + suffix_ids

def middle_split(tokenizer, token_id):
    text = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
    
    # Safety: Don't split very short tokens
    if len(text.strip()) <= 3:
        return [token_id]
    
    mid = len(text) // 2
    
    # We encode halves separately to force the tokenizer to use sub-optimal (smaller) 
    # tokens that exist in the vocab, rather than the original single token.
    prefix_ids = tokenizer.encode(text[:mid], add_special_tokens=False)
    suffix_ids = tokenizer.encode(text[mid:], add_special_tokens=False)
    
    return prefix_ids + suffix_ids

def minimal_split(tokenizer, token_id):
    # 1. Decode to string
    text = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
    
    # 2. Length Constraint: Skip words shorter than 3 characters
    if len(text.strip()) <= 3:
        return [token_id]

    best_ids = [token_id]
    min_token_count = float('inf')
    found_split = False

    # 3. Exhaustive search for the most efficient split point
    # We try every possible split point from n=1 to len-1
    for n in range(1, len(text)):
        prefix_text = text[:-n]
        suffix_text = text[-n:]
        
        # Encode separately to force the boundary
        p_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
        s_ids = tokenizer.encode(suffix_text, add_special_tokens=False)
        
        combined = p_ids + s_ids
        total_tokens = len(combined)

        # 4. We want a split (total > 1) that has the smallest possible count
        if total_tokens > 1:
            # If this split is more efficient than our previous best split
            if total_tokens < min_token_count:
                min_token_count = total_tokens
                best_ids = combined
                found_split = True
            
            # Optimization: If we find a 2-token split, that's the absolute minimum 
            # for a split, so we can return immediately.
            if total_tokens == 2:
                return best_ids

    return best_ids

# --- Testing ---
if __name__ == "__main__":
    model_id = "mistralai/Mistral-7B-v0.1"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    print(f"Loading model {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # Load in 16-bit to save memory; use device_map="auto" for multi-GPU
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    model.eval()

    test_prompts = [
        "The capital of France is",
        "Explain quantum computing in simple terms:",
        "The middle name of the 14th President of the United States was"
    ]

    encoded_prompts = [tokenizer.encode(p, add_special_tokens=True) for p in test_prompts]
    for p in encoded_prompts:
        # This decodes each ID separately so you can see the 'cuts'
        token_list = [tokenizer.decode([token_id]) for token_id in p]
        print(token_list)

    print("Starting entropy-split prefill...")
    final_input_ids = process_prompts_with_split(
        model, 
        tokenizer, 
        test_prompts, 
        minimal_split,
        entropy_threshold=4.0,
        device=device
    )

    print("\nPrefill Complete.")
    
    # Decode one to see the result
    for i, output in enumerate(final_input_ids):
        print(f"\nPrompt {i} decoded path:\n{[tokenizer.decode([token_id]) for token_id in output]}")