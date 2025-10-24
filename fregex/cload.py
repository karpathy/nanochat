import ctypes

c_lib = ctypes.CDLL("fregex/libfregex.dylib")


class TokenPos(ctypes.Structure):
    _fields_ = [
        ("start", ctypes.c_size_t),
        ("end", ctypes.c_size_t),
    ]


class TokenList(ctypes.Structure):
    _fields_ = [
        ("splits", ctypes.POINTER(TokenPos)),
        ("count", ctypes.c_size_t),
        ("capacity", ctypes.c_size_t),
    ]


c_lib.tokenlist_init.argtypes = [ctypes.POINTER(TokenList)]
c_lib.tokenlist_init.restype = None
c_lib.tokenlist_free.argtypes = [ctypes.POINTER(TokenList)]
c_lib.tokenlist_free.restype = None
# Accept a raw pointer to the input buffer rather than a Python bytes object
c_lib.tokenize_fast.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(TokenList)]
c_lib.tokenize_fast.restype = None

def tokenize_c_bytes(data: bytes) -> list[bytes]:
    # Use a C char* view of the original bytes; offsets computed from this base
    c_data = ctypes.c_char_p(data)
    tl = TokenList()
    c_lib.tokenlist_init(ctypes.byref(tl))
    try:
        base_addr = ctypes.cast(c_data, ctypes.c_void_p).value
        # Pass the same pointer to C
        c_lib.tokenize_fast(ctypes.cast(c_data, ctypes.c_void_p), len(data), ctypes.byref(tl))
        out: list[bytes] = []
        count = int(tl.count)
        for i in range(count):
            start_addr = int(tl.splits[i].start)
            end_addr = int(tl.splits[i].end)
            # Compute offsets into our local buffer
            off_start = start_addr - base_addr
            off_end = end_addr - base_addr
            if off_start < 0 or off_end < off_start or off_end > len(data):
                raise RuntimeError(f"Invalid span [{start_addr}:{end_addr}] for buffer base {base_addr}")
            out.append(data[off_start:off_end])
        return out
    finally:
        c_lib.tokenlist_free(ctypes.byref(tl))