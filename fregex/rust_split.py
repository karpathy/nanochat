import sys

from nanochat.tokenizer import SPLIT_PATTERN
from rustbpe import split_text as rust_split_text

def escape_bytes(b: bytes) -> str:
	buf = []
	for code in b:
		if code == 0x5C: buf.append('\\\\') 
		elif code == 0x0A: buf.append('\\n')
		elif code == 0x0D: buf.append('\\r')
		elif code == 0x09: buf.append('\\t')
		elif code == 0x0C: buf.append('\\f')
		elif code == 0x0B: buf.append('\\v')
		elif code == 0x22: buf.append('\\"')
		elif code < 32 or code >= 127:
			buf.append(f"\\x{code:02X}")
		else:
			buf.append(chr(code))
	return ''.join(buf)

def main():
	if len(sys.argv) < 2:
		print(f"Usage: {sys.argv[0]} <input-file>", file=sys.stderr)
		sys.exit(2)
	with open(sys.argv[1], 'rb') as f:
		data = f.read()
	text = data.decode('utf-8', errors='surrogatepass')
	parts = rust_split_text(SPLIT_PATTERN, text)
	for tok in parts:
		b = tok.encode('utf-8', errors='surrogatepass')
		esc = escape_bytes(b)
		print(f"{len(b)}\t{esc}")

if __name__ == "__main__":
	main()


