# Korean Tokenizer Survey for Dynamic Tokenization

**Goal:** Find tokenizers trained on richer Korean data than Mistral-7B, evaluate entropy split (and merge+split where applicable), compare accuracy on KoBEST.

---

## 1. Survey Summary

| # | Model | Korean Training Data | Tokens (sent1) | Tokens (sent2) | Merge Support |
|---|-------|---------------------|----------------|----------------|---------------|
| 1 | Mistral-7B-v0.1 (baseline) | Minimal (~few %) | 35 | 43 | ✅ zett HN |
| 2 | Polyglot-ko-5.8B | ~100% Korean (863B tokens) | 15 | 20 | ❌ |
| 3 | ko-gpt-trinity-1.2B | Korean web corpus (SKT) | 13 | 16 | ❌ (vocab-based possible) |
| 4 | EXAONE-3.5-2.4B | Extensive Korean+EN (LG AI) | 15 | 23 | ❌ |
| 5 | open-llama-ko-7B | LLaMA-2 + Korean extension | 16 | 25 | ❌ |
| 6 | Qwen2.5-7B | Large multilingual (incl. Korean) | 19 | 29 | ❌ |
| 7 | XLM-RoBERTa-base | CC-100 multilingual | 17 | 20 | ✅ zett HN (encoder-only) |
| 8 | TinyLlama-1.1B | English-dominant | 53 | 55 | ✅ zett HN (poor Korean) |

**Test sentence 1:** 심혈관 질환의 예방을 위해 규칙적인 운동이 필요합니다.
**Test sentence 2:** 자동차가 시속 60km로 2.5시간 동안 달리면 이동한 거리는 얼마입니까?

---

## 2. Tokenization Visualization

### Sentence 1: "심혈관 질환의 예방을 위해 규칙적인 운동이 필요합니다."

#### Mistral-7B (35 tokens)
```
['', '심', '혈(3 bytes)', '관', '', '질', '환', '의', '', '예', '방', '을', '', '위', '해', '', '규', '칙(3b)', '적', '인', '', '운', '동', '이', '', '필', '요', '합', '니', '다', '.']
→ 한국어 음절 상당수가 UTF-8 byte 3개로 분리됨 (byte fallback)
```

#### Polyglot-ko-5.8B (15 tokens)
```
['심', '혈관', ' 질환', '의', ' 예방', '을', ' 위해', ' 규칙', '적', '인', ' 운동', '이', ' 필요', '합니다', '.']
→ 어절 단위 근접. 혈관, 질환, 예방, 위해, 운동, 합니다 등 단어 단위 토큰
```

#### ko-gpt-trinity-1.2B (13 tokens) ★ 최소
```
['심', '혈', '관', '질환', '의', '예', '방을', '위해', '규칙', '적인', '운동이', '필요', '합니다.']
→ 조사 포함 어절 단위 토큰 (방을, 운동이, 합니다. 등)
```

#### EXAONE-3.5-2.4B (15 tokens)
```
['심', '혈관', ' 질환', '의', ' 예방', '을', ' 위해', ' 규칙', '적', '인', ' 운동', '이', ' 필요', '합니다', '.']
→ Polyglot-ko와 동일한 패턴, 102k vocab 활용
```

#### open-llama-ko-7B (16 tokens)
```
['심', '혈', '관', '질환', '의', '예방', '을', '위해', '규', '칙', '적인', '운동', '이', '필요', '합니다', '.']
→ 규칙 → 규+칙 분리. 전반적으로 음절~단어 사이
```

#### Qwen2.5-7B (19 tokens)
```
['심', '혈', '관', ' 질', '환', '의', ' 예', '방', '을', ' 위해', ' 규', '칙', '적인', ' 운', '동', '이', ' 필요', '합니다', '.']
→ 151k vocab이지만 한국어 byte-merge 위주. Mistral보다는 나음
```

#### XLM-RoBERTa-base (17 tokens, zett HN available)
```
['심', '혈', '관', '', '질환', '의', '예방', '을', '위해', '', '규칙', '적인', '운동', '이', '필요', '합니다', '.']
→ 5,373개 한국어 token 직접 보유. 질환/예방/위해/규칙/적인/운동/필요/합니다 단어 단위
```

---

## 3. Korean Training Data Detail

| Model | Organization | Korean Data | Notes |
|-------|-------------|-------------|-------|
| Polyglot-ko-5.8B | EleutherAI | ~863B Korean tokens (나무위키, 신문, 웹 등) | 한국어 전용 |
| ko-gpt-trinity-1.2B | SKT | Korean web + 뉴스 + 위키 | 한국어 전용 |
| EXAONE-3.5-2.4B | LG AI | 비공개, 한국어+영어 (EXAONE 시리즈) | 대규모 한국어 |
| open-llama-ko-7B | kfkas | LLaMA-2 + Korean alpaca/wiki | 한국어 추가학습 |
| XLM-RoBERTa-base | Meta/Fairseq | CC-100 Korean subset | 다국어 |
| Qwen2.5-7B | Alibaba | Multilingual (한국어 포함) | 한국어 비율 미공개 |
| Mistral-7B | Mistral AI | 영어 중심 (한국어 극소량) | baseline |
| TinyLlama-1.1B | TinyLlama | 영어 중심 | zett HN 있으나 한국어 취약 |

---

## 4. Experiment Plan

### Conditions per model
- **plain**: original tokenization → forward → predict
- **entropy_split**: plain + entropy scan → split high-entropy tokens → re-forward
- **merge** (Mistral only, zett HN): dynamic BPE merge → forward
- **merge+split** (Mistral only): merge → entropy scan → split → re-forward

### Benchmark
- **KoBEST COPA** (500 examples, 5-shot) — fastest classification task
- max_examples=100 for quick results; full run on Tillicum

---

## 5. Results (TBD — populated after experiments)

| Model | plain | entropy_split | merge | merge+split |
|-------|-------|--------------|-------|-------------|
| Mistral-7B | TBD | TBD | TBD | TBD |
| Polyglot-ko-5.8B | TBD | TBD | N/A | N/A |
| ko-gpt-trinity-1.2B | TBD | TBD | N/A | N/A |
| EXAONE-3.5-2.4B | TBD | TBD | N/A | N/A |
| open-llama-ko-7B | TBD | TBD | N/A | N/A |
| Qwen2.5-7B | TBD | TBD | N/A | N/A |
| XLM-RoBERTa-base | TBD | TBD | TBD (enc) | TBD (enc) |
