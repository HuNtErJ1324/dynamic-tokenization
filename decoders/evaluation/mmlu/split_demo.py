import torch
import torch.nn.functional as F
import ftfy
from tokenizations.dynamic_bpe import Dynamic_BPE
from transformers import AutoModelForCausalLM, AutoTokenizer
from decoders.evaluation.mmlu.split_utils import process_prompts_with_split, minimal_split

if __name__ == "__main__":
    model_id = "mistralai/Mistral-7B-v0.1"
    hypernet_id = "benjamin/zett-hypernetwork-Mistral-7B-v0.1"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    print(f"Loading model {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    print(f"Loading model {hypernet_id}...")
    hypernet_tokenizer = AutoTokenizer.from_pretrained(hypernet_id)

    dynamic_bpe = Dynamic_BPE(tokenizer=hypernet_tokenizer, tokenizer_boundary='pretokens')
    
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
        "Could you provide a code snippet for tuning a transformer's hyperparameters?",
        "Explain quantum computing in simple terms:",
        "The institutionalization of uncharacteristically dysfunctional compartmentalization",
        "숙소 위치는 찾기 쉽고 일반적인 한국의 반지하 숙소입니다."
    ]

    encoded_prompts = [tokenizer.encode(p, add_special_tokens=True) for p in test_prompts]
    for p in encoded_prompts:
        print(p)
        # This decodes each ID separately so you can see the 'cuts'
        token_list = [tokenizer.decode([token_id]) for token_id in p]
        print(token_list)

    print("Starting entropy-split prefill...")
    final_input_ids = process_prompts_with_split(
        model, 
        tokenizer, 
        test_prompts, 
        minimal_split,
        entropy_threshold=3.0,
        device=device
    )

    print("\nPrefill Complete.")
    
    # Decode one to see the result
    for i, output in enumerate(final_input_ids):
        print(f"\nPrompt {i} decoded path:\n{[tokenizer.decode([token_id]) for token_id in output]}")

    examples = [{"text": p} for p in test_prompts]
    examples = [
        {"text": "Ribonucleoprotein particles regulate mRNA stability."},
        {"text": "mRNA stability is modulated by ribonucleoprotein factors."},
    ]
    dynamic_tokens, _, _, _ = dynamic_bpe.tokenize_batch(
        batch_examples=examples,
        max_nr_merges=10,
        mlm=False
    )

    for i, tokens in enumerate(dynamic_tokens):
        # We strip the <s> and </s> special tokens for a cleaner view of the splits
        clean_tokens = [t for t in tokens if t not in ['<s>', '</s>']]
        
        # Display the prompt (truncated if too long) and the token list
        prompt_preview = test_prompts[i][:37] + "..." if len(test_prompts[i]) > 40 else test_prompts[i]
        print(f"{prompt_preview:<40} | {clean_tokens}")
        print(f"{'Count: ' + str(len(clean_tokens)):<40} |")
        print("-" * 80)

    merging_batch = [
        {"text": "The transcriptomic analysis of ribonucleoprotein revealed mRNA patterns."},
        {"text": "Ribonucleoprotein complexes are essential for mRNA stability and regulation."},
        {"text": "We analyzed the ribonucleoprotein interactions in the transcriptomic data."},
        {"text": "mRNA localization is driven by specific ribonucleoprotein particles."},
        {"text": "Transcriptomic profiling helps identify ribonucleoprotein binding sites."},
        {"text": "Hyperparameter tuning for transformer models involves the learning_rate."},
        {"text": "Adjusting the learning_rate is a key hyperparameter optimization step."},
    ]

    korean_merging_batch = [
        {"text": "심혈관 질환의 예방을 위해 규칙적인 운동이 필요합니다."},
        {"text": "심혈관 건강은 식습관과 밀접한 관련이 있습니다."},
        {"text": "최근 연구에서 심혈관 세포의 재생 가능성이 확인되었습니다."},
        {"text": "인공지능 기반의 진단 시스템이 빠르게 발전하고 있습니다."},
        {"text": "인공지능 기술은 의료 분야에서 혁신적인 변화를 일으키고 있습니다."}
    ]

    dynamic_bpe = Dynamic_BPE(tokenizer=hypernet_tokenizer, tokenizer_boundary='pretokens')

    # Assuming your dynamic_bpe and hypernet_tokenizer are already initialized
    dynamic_tokens, _, _, _ = dynamic_bpe.tokenize_batch(
        batch_examples=korean_merging_batch,
        max_nr_merges=15, # Slightly higher merges to capture longer words
        mlm=True
    )

    standard_results = [tokenizer.tokenize(ex["text"]) for ex in korean_merging_batch]

    def decode_tokenizer_jargon(token_list):
        # This cleans up the "mojibake" (scrambled text) 
        # and turns the byte-strings back into readable Korean
        return [ftfy.fix_encoding(t.replace('Ġ', ' ')) for t in token_list]
    
    for i, tokens in enumerate(dynamic_tokens):
        # Remove special tokens for readability
        dyn_clean = [t for t in dynamic_tokens[i] if t not in ['<s>', '</s>']]
        std_clean = standard_results[i]
        print(f"{'Original':<15} | {len(std_clean):<5} | {std_clean}")
        print(f"{'Dynamic BPE':<15} | {len(dyn_clean):<5} | {dyn_clean}")
        decoded_dyn = decode_tokenizer_jargon(dyn_clean)
        print(f"Dynamic (Readable): {decoded_dyn}")
        print("-" * 100)