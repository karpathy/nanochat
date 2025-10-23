import sys
import ctypes
from pathlib import Path

from nanochat.tokenizer import SPLIT_PATTERN
from rustbpe import split_text as rust_split_text
from fregex.cload import *
from fregex.py_tokenizer import tokenize_py as py_tokenize_str

def escape_bytes(b: bytes) -> str:
    buf = []
    for code in b:
        if code == 0x5C:
            buf.append('\\')
        elif code == 0x0A:
            buf.append('\\n')
        elif code == 0x0D:
            buf.append('\\r')
        elif code == 0x09:
            buf.append('\\t')
        elif code == 0x0C:
            buf.append('\\f')
        elif code == 0x0B:
            buf.append('\\v')
        elif code == 0x22:
            buf.append('\\"')
        elif code < 32 or code >= 127:
            buf.append(f"\\x{code:02X}")
        else:
            buf.append(chr(code))
    return ''.join(buf)

def dump_tokens(tokens: list[bytes]) -> str:
    return "\n".join(f"{len(b)}\t{escape_bytes(b)}" for b in tokens)

def tokenize_c_bytes(data: bytes) -> list[bytes]:
    tl = TokenList()
    c_lib.tokenlist_init(ctypes.byref(tl))
    try:
        c_lib.tokenize_fast(data, len(data), ctypes.byref(tl))
        out: list[bytes] = []
        count = int(tl.count)
        for i in range(count):
            ptr = tl.tokens[i]
            ln = int(tl.lengths[i])
            out.append(ctypes.string_at(ptr, ln))
        return out
    finally:
        c_lib.tokenlist_free(ctypes.byref(tl))

def tokenize_py_bytes(data: bytes) -> list[bytes]:
    text = data.decode('utf-8', errors='surrogatepass')
    toks = py_tokenize_str(text)
    return [t.encode('utf-8', errors='surrogatepass') for t in toks]

def tokenize_rs_bytes(data: bytes) -> list[bytes]:
    text = data.decode('utf-8', errors='surrogatepass')
    parts = rust_split_text(SPLIT_PATTERN, text)
    return [t.encode('utf-8', errors='surrogatepass') for t in parts]

def compare_one(path: Path) -> int:
    data_bytes = Path(path).read_bytes()
    try:
        c_toks = tokenize_c_bytes(data_bytes)
    except Exception as e:
        print(f"C tokenizer failed on {path}:\n{e}", file=sys.stderr)
        return 1
    try:
        py_toks = tokenize_py_bytes(data_bytes)
    except Exception as e:
        print(f"Python tokenizer failed on {path}:\n{e}", file=sys.stderr)
        return 1
    try:
        rs_toks = tokenize_rs_bytes(data_bytes)
    except Exception as e:
        print(f"Rust split failed on {path}:\n{e}", file=sys.stderr)
        return 1

    out_c = dump_tokens(c_toks)
    out_py = dump_tokens(py_toks)
    out_rs = dump_tokens(rs_toks)

    if out_c == out_py == out_rs:
        print(f"OK {path.name}")
        return 0
    else:
        print(f"DIFF {path.name}")
        # Show a small 3-way diff at first differing line, with byte offsets
        c_lines = out_c.splitlines()
        p_lines = out_py.splitlines()
        r_lines = out_rs.splitlines()

        def parse_lines(lines):
            parsed = []
            for ln in lines:
                # Format is: "<len>\t<escaped>"
                try:
                    left, right = ln.split('\t', 1)
                    blen = int(left)
                except Exception:
                    blen = 0
                    right = ln
                parsed.append((blen, right))
            return parsed

        c_parsed = parse_lines(c_lines)
        p_parsed = parse_lines(p_lines)
        r_parsed = parse_lines(r_lines)

        def byte_offsets(parsed):
            offs = []
            pos = 0
            for blen, _ in parsed:
                offs.append((pos, pos + blen))
                pos += blen
            return offs

        c_offs = byte_offsets(c_parsed)
        p_offs = byte_offsets(p_parsed)
        r_offs = byte_offsets(r_parsed)

        data_bytes = Path(path).read_bytes()

        def print_unicode_debug(label, offs_list, idx):
            if idx >= len(offs_list):
                print(f"    {label} piece: [n/a]")
                return
            start, end = offs_list[idx]
            piece_bytes = data_bytes[start:end]
            piece_text = piece_bytes.decode('utf-8', errors='replace')
            if not piece_bytes:
                print(f"    {label} piece: [EMPTY]")
                return
            cp_parts = []
            for ch in piece_text:
                cp_parts.append(f"U+{ord(ch):04X}")
            bytes_hex = ' '.join(f"{b:02X}" for b in piece_bytes)
            print(f"    {label} chars: {' | '.join(cp_parts)}")
            print(f"    {label} bytes: {bytes_hex}  ({len(piece_bytes)}B, {len(piece_text)} chars)")

        max_len = max(len(c_lines), len(p_lines), len(r_lines))
        for i in range(max_len):
            cl = c_lines[i] if i < len(c_lines) else "<eof>"
            pl = p_lines[i] if i < len(p_lines) else "<eof>"
            rl = r_lines[i] if i < len(r_lines) else "<eof>"
            if not (cl == pl == rl):
                # Collect byte positions if available
                c_pos = f"[{c_offs[i][0]}:{c_offs[i][1]}]" if i < len(c_offs) else "[n/a]"
                p_pos = f"[{p_offs[i][0]}:{p_offs[i][1]}]" if i < len(p_offs) else "[n/a]"
                r_pos = f"[{r_offs[i][0]}:{r_offs[i][1]}]" if i < len(r_offs) else "[n/a]"
                print(
                    f"  line {i+1}:\n"
                    f"    C:  {cl}  @ bytes {c_pos}\n"
                    f"    Py: {pl}  @ bytes {p_pos}\n"
                    f"    Rs: {rl}  @ bytes {r_pos}"
                )
                print("    === Unicode split detail ===")
                print_unicode_debug("C", c_offs, i)
                print_unicode_debug("Py", p_offs, i)
                print_unicode_debug("Rs", r_offs, i)
                break
        return 2

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <tests-dir>")
        sys.exit(2)
    paths = sorted(Path(sys.argv[1]).glob('*.txt'))
    bad = 0
    for p in paths:
        bad += compare_one(p)
    print(f"Completed. Failures: {bad}")
    
if __name__ == '__main__':
    main()


