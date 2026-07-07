SEQUENCE_LEN = 64
MAX_TOKENS = 10
MAX_CTX = SEQUENCE_LEN - MAX_TOKENS  # 54


def apply_sliding_window(conversation_tokens, sequence_len, max_tokens):
    # Local mirror of the guard added to scripts/chat_cli.py.
    # TODO: once Task 3 lands, import the real function instead of duplicating.
    max_ctx = sequence_len - max_tokens
    if len(conversation_tokens) > max_ctx:
        return conversation_tokens[-max_ctx:]
    return conversation_tokens


def test_truncation_removes_oldest_tokens_when_over_limit():
    long_tokens = list(range(80))

    result = apply_sliding_window(long_tokens, SEQUENCE_LEN, MAX_TOKENS)

    assert len(result) == MAX_CTX
    assert result == long_tokens[-MAX_CTX:]


def test_truncation_leaves_short_conversation_unchanged():
    short_tokens = list(range(20))

    result = apply_sliding_window(short_tokens, SEQUENCE_LEN, MAX_TOKENS)

    assert result == short_tokens


def test_truncation_at_exact_boundary_leaves_unchanged():
    boundary_tokens = list(range(MAX_CTX))

    result = apply_sliding_window(boundary_tokens, SEQUENCE_LEN, MAX_TOKENS)

    assert result == boundary_tokens


def test_truncation_one_over_boundary_drops_oldest_token():
    tokens = list(range(MAX_CTX + 1))

    result = apply_sliding_window(tokens, SEQUENCE_LEN, MAX_TOKENS)

    assert len(result) == MAX_CTX
    assert result[0] == 1
    assert result[-1] == MAX_CTX
