# HRM8K Evaluation Results

**Model:** mistralai/Mistral-7B-v0.1  
**Dataset:** HAERAE-HUB/HRM8K (KSM for Korean, GSM8K for English)  
**Date:** 2026-05-18 ~ 2026-05-19  
**Setup:** 3-shot, max_new_tokens=64, batch_size=4

---

## Accuracy

| Method | English | Korean | EN-KO Gap |
|--------|--------:|-------:|----------:|
| Plain | 2.50% (33/1319) | 3.15% (45/1428) | -0.65pp (KO higher) |
| Entropy Split (threshold=3.0) | 2.96% (39/1319) | 3.15% (45/1428) | -0.19pp (KO higher) |
| Dynamic BPE (merges=1000) | 1.59% (21/1319) | 4.76% (68/1428) | -3.17pp (KO higher) |
| Dynamic BPE + Entropy Split (byte-level, broken) | 1.52% (20/1319) | 1.47% (21/1428) | +0.05pp (EN higher) |
| Dynamic BPE + Entropy Split (char-level, thr=3.0) | 0.61% (8/1319) | 1.96% (28/1428) | -1.35pp (KO higher) |
| Dynamic BPE + Entropy Split (char-level, thr=6.0) | 1.29% (17/1319) | 4.69% (67/1428) | -3.40pp (KO higher) |

---

## Latency

### Plain (job 118905)
- Total wall time: 797s (13m 17s)
- Throughput: ~3.49 ex/s
- Per example: ~286ms

### Entropy Split (job 118907)
- Total wall time: 855s (14m 15s)
- Throughput: ~3.24 ex/s
- Per example: ~309ms
- Encode overhead (entropy scan): mean 75.35ms/batch vs plain 1.51ms/batch

### Dynamic BPE (job 118951)
- Total wall time: 877s (14m 37s)
- Throughput: ~3.31 ex/s
- Per example: ~302ms
- Encode overhead (BPE+hypernet): mean 77.34ms/batch
- Note: Early exit common (~250/1000 merges avg) — Korean tokens already fine-grained

### Dynamic BPE + Entropy Split (job 118962)
- Total wall time: 936s (15m 36s)
- Throughput: ~3.10 ex/s
- Per example: ~323ms
- Encode: mean 8.77ms/batch (LRU cache warm from EN pass)
- Generate: mean 1234ms/batch (includes entropy scan + re-embed)
- **No-answer rate: 84.03% (1200/1428 KO)** ← generation broken

---

## Observations

### Plain
- Low accuracy expected for math generation without CoT and max_new_tokens=64
- Korean slightly higher than English (negative gap) — consistent across methods

### Entropy Split
- English improves slightly (+0.46pp), Korean unchanged
- Korean tokens already at syllable/byte level → minimal_split can't split further → no effect on Korean
- ~8% latency overhead from entropy scan forward pass

### Dynamic BPE
- English drops from 2.50% → 1.59% (-0.91pp)
- Korean: 3.15% → 4.76% (+1.61pp) — modest gain, likely noise given overall low accuracy
- "Early exit" common: only ~250 merges out of 1000 possible (Korean vocab already byte-level)
- Token count significantly reduced: e.g., 35 tokens → 9 tokens for one Korean sentence (word-level merging)

### Dynamic BPE + Entropy Split (byte-level, broken)
- **No-answer rate 84%**: model generates text without any numbers
- Accuracy collapses to near-zero (1.52% EN, 1.47% KO)
- Root cause: `tokenize(char)` adds spurious Ġ prefix → byte-level explosion → incoherent generation

### Dynamic BPE + Entropy Split (char-level fix, job 119014/119015)
- **No-answer rate 0.56% (KO)** — 84% → 0.56%, 모델이 숫자 생성 정상화
- EN: 0.61% (8/1319), KO: 1.96% (28/1428)
- EN-KO gap: -1.35pp (KO가 EN보다 높음, 다른 방법들과 일관된 패턴)
- 정확도는 plain보다 낮음: char-level 토큰 시퀀스가 Mistral pretraining과 다른 분포
- 레이턴시: ~297ms/ex (이전 byte-level 323ms보다 빠름, encode overhead 8ms로 감소)
- threshold=3.0에서 대부분 어절 split → plain 대비 이점 없음; threshold 상향 필요

### Dynamic BPE + Entropy Split (char-level, thr=6.0, job 119021)
- EN: 1.29% (17/1319), KO: **4.69% (67/1428)**
- **KO가 plain(3.15%) 및 dynamic_bpe(4.76%)에 근접**
- No-answer rate: 2.45% (KO) — 적정 수준
- EN-KO gap: -3.40pp (KO가 EN보다 훨씬 높음)
- threshold=6.0: 매우 불확실한 merged token만 split → 의미 있는 어절 보존
- EN은 plain 대비 여전히 낮음 (pretraining token 분포 shift 영향)

---

## Comparison with MMLU/KMMLU (from Reports 6–8)

| Method | MMLU (EN) | KMMLU (KO) | HRM8K EN | HRM8K KO |
|--------|----------:|----------:|---------:|---------:|
| Plain | 60.25% | 36.39% | 2.50% | 3.15% |
| Entropy Split (thr=4.0) | 51.44% | — | 2.96% | 3.15% |
| Original TK + HN Embed | 56.92% | — | — | — |
| Dynamic BPE | 54.06% | 19.18% | 1.59% | 4.76% |
| Dynamic BPE + Entropy Split | — | — | 1.52%* | 1.47%* |

*84% no-answer rate — not meaningful

---

## Key Findings

1. **Entropy split is ineffective for Korean**: Mistral tokenizes Korean to syllable/byte level → nothing meaningful to split
2. **Dynamic BPE hurts English** consistently (MMLU: -6pp, HRM8K: -0.9pp) due to token boundary distribution shift from pretraining
3. **Dynamic BPE + Entropy Split is broken**: High entropy on BPE-merged tokens causes over-splitting → incoherent generation. Needs much higher threshold (e.g., 7.0+) or different entropy measurement strategy
4. **All methods below 5% on HRM8K**: math generation requires CoT; these results reflect tokenization effects, not absolute model capability
5. **Threshold sensitivity** (from MMLU): thr=10.0 ≈ plain (59.92%), thr=3.0 → 49.00%; entropy split consistently hurts at any practically useful threshold
