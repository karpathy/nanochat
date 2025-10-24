import sys
import time
import random
import argparse
import unicodedata as u
import ctypes
from pathlib import Path

from fregex.cload import *

HERE = Path(__file__).resolve().parent
TESTS_DIR = HERE / "tests"

from fregex.py_tokenizer import tokenize_py as py_tokenize_str

def escape_bytes(b: bytes) -> str:
    buf = []
    for code in b:
        if code == 0x5C:
            buf.append('\\\\')
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

def gen_valid_unicode_string(rng: random.Random, max_len: int) -> str:
    target_len = rng.randint(0, max_len)

    ws_cps = [
        0x20, 0x09, 0x0A, 0x0B, 0x0C, 0x0D,  # space, \t, \n, \v, \f, \r
        0x00A0,  # NO-BREAK SPACE
        0x1680,  # OGHAM SPACE MARK
        0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006,
        0x2007, 0x2008, 0x2009, 0x200A,  # EN/EM/THIN/HAIR SPACES etc.
        0x2028, 0x2029,  # LINE SEPARATOR, PARAGRAPH SEPARATOR
        0x202F,  # NARROW NO-BREAK SPACE
        0x205F,  # MEDIUM MATHEMATICAL SPACE
        0x3000,  # IDEOGRAPHIC SPACE
        0x200B,  # ZERO WIDTH SPACE (not WS in Python, but hits tokenizer class)
        0xFEFF,  # ZERO WIDTH NO-BREAK SPACE
    ]

    ascii_punct = [
        ord(c)
        for c in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    ]

    def rand_scalar_excluding_surrogates(lo: int, hi: int) -> int:
        while True:
            cp = rng.randint(lo, hi)
            if 0xD800 <= cp <= 0xDFFF:
                continue
            return cp

    def is_ws_char(ch: str) -> bool:
        cp = ord(ch)
        return ch.isspace() or (cp in ws_cps)

    def gen_ws_segment(max_run: int) -> str:
        # Mix of various spaces, often multi-length; sometimes explicit CRLFs
        if rng.random() < 0.35:
            # Build CR, LF, CRLF, or repeated newlines
            seqs = ["\n", "\r", "\r\n"]
            unit = rng.choice(seqs)
            unit_len = len(unit)
            max_reps = max(1, max_run // unit_len)
            seg = unit * rng.randint(1, max_reps)
            return seg
        run = rng.randint(1, max(1, max_run))
        buf = []
        for _ in range(run):
            cp = rng.choice(ws_cps)
            buf.append(chr(cp))
        return ''.join(buf)

    def gen_letter_run(max_run: int) -> str:
        run = rng.randint(1, max(1, max_run))
        buf = []
        for _ in range(run):
            if rng.random() < 0.6:
                # ASCII letters
                base = ord('A') if rng.random() < 0.5 else ord('a')
                buf.append(chr(base + rng.randint(0, 25)))
            else:
                # Any Unicode letter
                while True:
                    cp = rand_scalar_excluding_surrogates(0x00A0, 0x10FFFF)
                    if u.category(chr(cp)).startswith('L'):
                        buf.append(chr(cp))
                        break
        # optional prefix of single non-WS, non-letter, non-number to stress
        # the leading [^\r\n\p{L}\p{N}]?+ in the regex
        if rng.random() < 0.3:
            buf.insert(0, gen_punc_run(1, allow_space=False))
        return ''.join(buf)

    def gen_number_run(max_run: int) -> str:
        # Bias to lengths 1..2 per \p{N}{1,2}, but sometimes longer
        if rng.random() < 0.7:
            run = rng.randint(1, min(2, max_run))
        else:
            run = rng.randint(3, max(3, max_run))
        buf = []
        for _ in range(run):
            if rng.random() < 0.75:
                buf.append(chr(ord('0') + rng.randint(0, 9)))
            else:
                # Other numeric categories (Nd/Nl/No)
                while True:
                    cp = rand_scalar_excluding_surrogates(0x00A0, 0x10FFFF)
                    if u.category(chr(cp)).startswith('N'):
                        buf.append(chr(cp))
                        break
        return ''.join(buf)

    def gen_punc_run(max_run: int, allow_space: bool = True) -> str:
        run = rng.randint(1, max(1, max_run))
        buf = []
        # optional leading single space before punc block
        if allow_space and rng.random() < 0.5:
            buf.append(' ')
        for _ in range(run):
            if rng.random() < 0.6:
                cp = rng.choice(ascii_punct)
            else:
                while True:
                    cp = rand_scalar_excluding_surrogates(0, 0x10FFFF)
                    ch = chr(cp)
                    if (
                        not u.category(ch).startswith('L') and
                        not u.category(ch).startswith('N') and
                        cp not in ws_cps and
                        not ch.isspace()
                    ):
                        break
                # ensure we don't accidentally add null
            buf.append(chr(cp))
        # optional trailing newlines to stress [\r\n]*
        if rng.random() < 0.35:
            tail = gen_ws_segment(3)
            # Keep only CR/LF components in the tail for this case
            tail = tail.replace('\t', '').replace('\v', '').replace('\f', '').replace(' ', '')
            buf.append(tail)
        return ''.join(buf)

    def gen_contraction() -> str:
        # e.g., we're, he'll, I'd, I'm, can't, they've
        prefixes = [gen_letter_run( rng.randint(1, 6) )]
        suffix = rng.choice(["s", "d", "m", "t", "ll", "ve", "re"])
        return prefixes[0] + "'" + suffix

    def gen_random_unicode(max_run: int) -> str:
        run = rng.randint(1, max(1, max_run))
        buf = []
        for _ in range(run):
            cp = rand_scalar_excluding_surrogates(0, 0x10FFFF)
            try:
                buf.append(chr(cp))
            except ValueError:
                continue
        return ''.join(buf)

    buf: list[str] = []
    curr_len = 0
    # Build by segments until target_len
    while curr_len < target_len:
        remain = target_len - curr_len
        r = rng.random()
        if r < 0.40:
            seg = gen_ws_segment(remain)
        elif r < 0.45:
            # Explicit newline-focused segment
            seg = ("\r\n" if rng.random() < 0.5 else ("\n" if rng.random() < 0.5 else "\r")) * rng.randint(1, max(1, remain))
        elif r < 0.65:
            seg = gen_letter_run(remain)
        elif r < 0.75:
            seg = gen_number_run(remain)
        elif r < 0.90:
            seg = gen_punc_run(remain)
        elif r < 0.95:
            seg = gen_contraction()
        else:
            seg = gen_random_unicode(remain)

        if not seg:
            continue
        # Trim if needed
        # Append
        for ch in seg:
            if curr_len >= target_len:
                break
            if is_ws_char(ch):
                buf.append(ch)
                curr_len += 1
            else:
                buf.append(ch)
                curr_len += 1

    # Occasionally end with trailing spaces to stress \s+(?!\S)
    if curr_len < max_len and rng.random() < 0.3:
        trail = gen_ws_segment(max_len - curr_len)
        if rng.random() < 0.7:
            trail = (' ' if rng.random() < 0.6 else '\t') * rng.randint(1, min(8, max_len - curr_len))
        # Append trailing
        for ch in trail:
            if curr_len >= max_len:
                break
            if is_ws_char(ch):
                buf.append(ch)
                curr_len += 1
            else:
                buf.append(ch)
                curr_len += 1

    return ''.join(buf)

def write_temp_case(text: str, tag: str = "RUN") -> Path:
    TESTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = int(time.time() * 1000)
    fname = f"in_fuzz_{tag}_{ts}.txt"
    path = TESTS_DIR / fname
    with open(path, 'wb') as f:
        f.write(text.encode('utf-8', errors='surrogatepass'))
    return path

def _format_tokens_dump(tokens: list[bytes]) -> str:
    lines = []
    for b in tokens:
        lines.append(f"{len(b)}\t{escape_bytes(b)}")
    return "\n".join(lines)

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
    if py_tokenize_str is None:
        raise RuntimeError("py_tokenizer not available")
    text = data.decode('utf-8', errors='surrogatepass')
    toks = py_tokenize_str(text)
    return [t.encode('utf-8', errors='surrogatepass') for t in toks]

def compare_pair_text(text: str):
    data = text.encode('utf-8', errors='surrogatepass')
    try:
        toks_c = tokenize_c_bytes(data)
    except Exception as e:
        return False, f"C failed: {e}", None, None
    try:
        toks_py = tokenize_py_bytes(data)
    except Exception as e:
        return False, f"Py failed: {e}", None, None
    ok = toks_c == toks_py
    return ok, None, _format_tokens_dump(toks_c), _format_tokens_dump(toks_py)

def run_fuzz(iters: int, max_len: int, seed: int, stop_on_first: bool):
    rng = random.Random(seed)
    total = 0
    mismatches = 0
    last_save = None

    for i in range(iters if iters > 0 else 1_000_000_000):
        s = gen_valid_unicode_string(rng, max_len)
        ok, err, out_c, out_py = compare_pair_text(s)
        total += 1
        if not ok:
            mismatches += 1
            fail_path = write_temp_case(s, tag="FAIL")
            last_save = fail_path
            print(f"Mismatch at iter {i}, saved to {fail_path}")
            print(f"Seed: {seed}")

            cps = [f"U+{ord(ch):04X}" for ch in s]
            cats = [u.category(ch) for ch in s]
            print(f"Text bytes len: {len(s.encode('utf-8','surrogatepass'))}, chars: {len(s)}")
            print(f"Codepoints: {' '.join(cps)}")
            print(f"Categories: {' '.join(cats)}")
            if err:
                print(err)
            if out_c is not None:
                print("--- C tokens ---")
                print(out_c)
            if out_py is not None:
                print("--- Py tokens ---")
                print(out_py)

            if stop_on_first:
                break

        if (i + 1) % 100 == 0:
            print(f"[fuzz] {i+1} cases, mismatches={mismatches}")
            # print(out_c, out_py, sep="\n")

    return total, mismatches, last_save


def main():
    ap = argparse.ArgumentParser(description="Fuzz C vs Python tokenizers on random valid UTF-8 inputs")
    ap.add_argument("--iters", type=int, default=0, help="Number of iterations (0 = very large run)")
    ap.add_argument("--max-len", type=int, default=256, help="Maximum number of Unicode scalars per case")
    ap.add_argument("--seed", type=int, default=12345, help="PRNG seed for reproducibility")
    ap.add_argument("--stop-on-first", action="store_true", help="Stop at first mismatch (default: run all)")
    args = ap.parse_args()

    total, mismatches, last = run_fuzz(args.iters, args.max_len, args.seed, args.stop_on_first)
    print(f"Completed {total} cases, mismatches={mismatches}")
    if last:
        print(f"Last failing case saved at: {last}")
    if mismatches:
        sys.exit(1)


if __name__ == "__main__":
    main()


