# Dynamic BPE + Entropy Split 시각화 (job 119003, char-fix 이전)

**설정:** merges=1000, entropy_threshold=3.0, 한국어 5문장  
**상태:** char-level split 수정 전 — split 결과가 byte 조각으로 폭발

---

## 문장 0: 심혈관 질환의 예방을 위해 규칙적인 운동이 필요합니다.

**Stage 0 — Plain Mistral (35 tokens)**
```
['▁', '심', '<0xED>', '<0x98>', '<0x88>', '관', '▁', '질', '환', '의', '▁', '예', '방', '을',
 '▁', '위', '해', '▁', '규', '<0xEC>', '<0xB9>', '<0x99>', '적', '인', '▁', '운', '동', '이',
 '▁', '필', '요', '합', '니', '다', '.']
```
Early exit: 25/1000 merges

**Stage 1 — Dynamic BPE (9 tokens, +26)**
```
['<s>', ' 심혈관', ' 질환의', ' 예방을', ' 위해', ' 규칙적인', ' 운동이', ' 필요합니다', '.']
```

**Stage 2 — Entropy (threshold=3.0)**

| 토큰 | entropy | 결정 |
|------|--------:|------|
| `<s>` | 4.447 | keep (special) |
| ` 심혈관` | 4.181 | SPLIT |
| ` 질환의` | 1.608 | keep |
| ` 예방을` | 1.719 | keep |
| ` 위해` | 3.396 | SPLIT |
| ` 규칙적인` | 3.729 | SPLIT |
| ` 운동이` | 4.631 | SPLIT |
| ` 필요합니다` | 2.574 | keep |
| `.` | 2.509 | keep |

**Stage 3 — After split (59 tokens, +50 from BPE)**
- ` 심혈관` → 14개 byte 조각 (char-fix 이전 버그)
- ` 위해` → 9개 byte 조각
- ` 규칙적인` → 17개 byte 조각
- ` 운동이` → 14개 byte 조각
- plain=35 → dynamic_bpe=9 → after_split=59 (**원래보다 24개 더 많음**)

---

## 문장 1: 인공지능 기반의 진단 시스템이 빠르게 발전하고 있습니다.

**Stage 0 — Plain Mistral (34 tokens)**
```
['▁', '인', '공', '지', '능', '▁', '기', '반', '의', '▁', '진', '단', '▁', '시', '스', '템', '이',
 '▁', '<0xEB>', '<0xB9>', '<0xA0>', '르', '게', '▁', '발', '전', '하', '고', '▁', '있', '습', '니', '다', '.']
```
Early exit: 25/1000 merges

**Stage 1 — Dynamic BPE (9 tokens, +25)**
```
['<s>', ' 인공지능', ' 기반의', ' 진단', ' 시스템이', ' 빠르게', ' 발전하고', ' 있습니다', '.']
```

**Stage 2 — Entropy (threshold=3.0)**

| 토큰 | entropy | 결정 |
|------|--------:|------|
| `<s>` | 4.447 | keep (special) |
| ` 인공지능` | 4.775 | SPLIT → 18개 byte 조각 |
| ` 기반의` | 1.444 | keep |
| ` 진단` | 3.866 | SPLIT → 9개 byte 조각 |
| ` 시스템이` | 6.589 | SPLIT → 17개 byte 조각 |

---

## 핵심 문제 (char-fix 이전)

split 결과가 Unicode 글자 단위가 아니라 UTF-8 byte 단위로 분해됨:
- ` 심혈관` (1 token) → 14 byte 조각 (기대값: 3글자 토큰)
- ` 인공지능` (1 token) → 18 byte 조각 (기대값: 4글자 토큰)

**원인:** `hypernet_tokenizer.tokenize(char)` 가 각 글자에 Ġ prefix를 추가하고 byte 분해  
**수정:** `_gpt2_encode_text(char)` 직접 인코딩으로 교체

수정 후 기대값:
- ` 심혈관` → `['Ġìĭ¬', 'íĺĪ', 'ê´Ģ']` = `[' 심', '혈', '관']` (3 tokens)
- ` 인공지능` → `['ĠìĿ¸', 'ê³µ', 'ì§Ģ', 'ëĬ¥']` = `[' 인', '공', '지', '능']` (4 tokens)
