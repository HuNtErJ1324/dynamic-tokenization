"""
viz-*.out 파일의 GPT-2 스타일 인코딩된 토큰을 사람이 읽을 수 있게 디코딩하여 MD로 저장.

Usage:
    python decode_viz_output.py logs/viz-119003.out
"""

import sys
import re


def build_byte_decoder():
    """GPT-2 bytes_to_unicode 역방향 매핑 (unicode char → byte value)."""
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


BYTE_DECODER = build_byte_decoder()


def decode_token(tok):
    """GPT-2 스타일 토큰 → 읽을 수 있는 문자열."""
    if tok in ('<s>', '</s>', '<unk>', '<pad>') or tok.startswith('<0x'):
        return tok
    t = tok.replace('Ġ', ' ').replace('▁', ' ')
    try:
        bts = bytes([BYTE_DECODER.get(c, ord(c) & 0xFF) for c in t])
        return bts.decode('utf-8')
    except (UnicodeDecodeError, KeyError):
        return tok


def decode_token_list(line):
    """'['tok1', 'tok2', ...]' 형태의 줄에서 토큰 리스트를 디코딩."""
    tokens = re.findall(r"'([^']*)'", line)
    if not tokens:
        return line
    decoded = [decode_token(t) for t in tokens]
    return "  " + str(decoded)


def process(path):
    with open(path, encoding='utf-8') as f:
        lines = f.readlines()

    out_lines = []
    for line in lines:
        stripped = line.rstrip('\n')
        # 토큰 리스트가 있는 줄 디코딩
        if ("raw:" in stripped or "readable:" in stripped
                or "Stage 0" in stripped or "Stage 1" in stripped
                or "Stage 3" in stripped or "before" in stripped
                or "after" in stripped):
            # raw/readable 줄은 토큰 리스트 부분만 디코딩
            if stripped.strip().startswith('[') or ("  ['" in stripped) or "  ['" in stripped:
                out_lines.append(decode_token_list(stripped))
                continue
        out_lines.append(stripped)

    return '\n'.join(out_lines)


def to_md(text, out_path):
    md_lines = ["# Dynamic BPE + Entropy Split Visualization\n"]
    in_sentence = False

    for line in text.split('\n'):
        # 구분선
        if set(line.strip()) <= {'─', '=', '-'} and len(line.strip()) > 5:
            md_lines.append("---")
        # 문장 헤더
        elif re.match(r'\[\d+\]', line.strip()):
            md_lines.append(f"\n## {line.strip()}\n")
        # Stage 헤더
        elif line.strip().startswith('Stage'):
            md_lines.append(f"\n**{line.strip()}**")
        # 엔트로피 테이블 헤더
        elif 'token' in line and 'entropy' in line and 'action' in line:
            md_lines.append("\n| token | entropy | action |")
            md_lines.append("|-------|---------|--------|")
        # 엔트로피 행
        elif re.match(r"\s+'.*'\s+[\d.]+\s+(keep|SPLIT|keep \()", line):
            parts = line.strip().split()
            tok = parts[0]
            ent = parts[1]
            action = ' '.join(parts[2:])
            tok_decoded = decode_token(tok.strip("'"))
            md_lines.append(f"| `{tok_decoded}` | {ent} | {action} |")
        elif line.strip():
            md_lines.append(line.rstrip())
        else:
            md_lines.append("")

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    inp = sys.argv[1] if len(sys.argv) > 1 else "logs/viz-119003.out"
    out = inp.replace('.out', '_decoded.md').replace('logs/', 'results/')
    import os
    os.makedirs(os.path.dirname(out) if os.path.dirname(out) else '.', exist_ok=True)

    processed = process(inp)
    to_md(processed, out)
    print("\n--- Preview (first 60 lines) ---")
    for l in processed.split('\n')[:60]:
        print(l)
