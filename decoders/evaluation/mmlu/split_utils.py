import torch
import torch.nn.functional as F
import random
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
    # --- PASS 2: APPLY SPLIT ---
    processed_prompts = []
    
    for i, original_ids in enumerate(encoded_prompts):
        # Initialize with the first token (usually BOS)
        new_prompt = [original_ids[0]]
        
        # Check logit at position j, split token at position j if needed
        for j in range(1, len(original_ids)):
            next_token_id = original_ids[j]
            token_entropy = entropy_matrix[i, j].item()  # this looks at the entropy of predicting the next token
            if token_entropy > entropy_threshold:
                # Apply your modular splitting strategy
                fragments = split_fn(tokenizer, next_token_id)
                new_prompt.extend(fragments)
            else:
                new_prompt.append(next_token_id)
                
        processed_prompts.append(new_prompt)
        
    return processed_prompts

def process_prompts_with_random_split(tokenizer, prompts, split_fn, split_prob=0.2):
    encoded_prompts = [tokenizer.encode(p, add_special_tokens=True) for p in prompts]
    processed_prompts = []
    
    for i, original_ids in enumerate(encoded_prompts):
        # Initialize with the first token (usually BOS)
        new_prompt = [original_ids[0]]
        for j in range(1, len(original_ids)):
            next_token_id = original_ids[j]
            if (random.random() < split_prob):
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

def is_korean(char):
    # Hangul syllables range
    return '\uac00' <= char <= '\ud7a3'

def decompose_hangul(char):
    """
    Decompose a single Hangul syllable into jamo (초성, 중성, 종성).
    
    Returns:
        list of jamo characters (e.g., ['ㅎ','ㅏ','ㄴ'])
        or [char] if not a Hangul syllable
    """
    code = ord(char)

    # Hangul syllable range
    if not (0xAC00 <= code <= 0xD7A3):
        return [char]

    base = code - 0xAC00

    choseong_idx = base // 588
    jungseong_idx = (base % 588) // 28
    jongseong_idx = base % 28

    CHOSEONG = [
        "ㄱ","ㄲ","ㄴ","ㄷ","ㄸ","ㄹ","ㅁ","ㅂ","ㅃ","ㅅ",
        "ㅆ","ㅇ","ㅈ","ㅉ","ㅊ","ㅋ","ㅌ","ㅍ","ㅎ"
    ]

    JUNGSEONG = [
        "ㅏ","ㅐ","ㅑ","ㅒ","ㅓ","ㅔ","ㅕ","ㅖ","ㅗ","ㅘ",
        "ㅙ","ㅚ","ㅛ","ㅜ","ㅝ","ㅞ","ㅟ","ㅠ","ㅡ","ㅢ","ㅣ"
    ]

    JONGSEONG = [
        "", "ㄱ","ㄲ","ㄳ","ㄴ","ㄵ","ㄶ","ㄷ","ㄹ","ㄺ",
        "ㄻ","ㄼ","ㄽ","ㄾ","ㄿ","ㅀ","ㅁ","ㅂ","ㅄ","ㅅ",
        "ㅆ","ㅇ","ㅈ","ㅊ","ㅋ","ㅌ","ㅍ","ㅎ"
    ]

    result = [
        CHOSEONG[choseong_idx],
        JUNGSEONG[jungseong_idx]
    ]

    if JONGSEONG[jongseong_idx]:
        result.append(JONGSEONG[jongseong_idx])

    return result

def minimal_split(tokenizer, token_id, jamo=True):
    # 1. Decode to string
    text = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
    
    stripped = text.strip()

    # Handle single Korean unicode character → split into UTF-8 bytes
    if len(stripped) == 1 and is_korean(stripped):
        if (jamo):
            # print(f"jamo detected {stripped}")
            jamos = decompose_hangul(stripped)
            # print(f"{stripped} -> {jamos}")

            jamo_ids = []
            for j in jamos:
                ids = tokenizer.encode(j, add_special_tokens=False)
                jamo_ids.extend(ids)
                # print(jamo_ids)
            
            if len(jamo_ids) > 1:
                return jamo_ids
        else:
            byte_values = stripped.encode("utf-8")
            
            byte_ids = []
            for b in byte_values:
                byte_token = bytes([b]).decode("latin-1")
                ids = tokenizer.encode(byte_token, add_special_tokens=False)
                byte_ids.extend(ids)
            
            if len(byte_ids) > 1:
                return byte_ids

    # 2. Length Constraint: Skip words shorter than 3 characters
    if len(stripped) <= 3:
        return [token_id]

    best_ids = [token_id]
    min_token_count = float('inf')

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