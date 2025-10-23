import ctypes

c_lib = ctypes.CDLL("fregex/libfregex.dylib")

class TokenList(ctypes.Structure):
    pass

TokenList._fields_ = [
    ("tokens", ctypes.POINTER(ctypes.POINTER(ctypes.c_char))),
    ("lengths", ctypes.POINTER(ctypes.c_size_t)),
    ("count", ctypes.c_size_t),
    ("capacity", ctypes.c_size_t),
]

c_lib.tokenlist_init.argtypes = [ctypes.POINTER(TokenList)]
c_lib.tokenlist_init.restype = None
c_lib.tokenlist_free.argtypes = [ctypes.POINTER(TokenList)]
c_lib.tokenlist_free.restype = None
c_lib.tokenize_fast.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(TokenList)]
c_lib.tokenize_fast.restype = None