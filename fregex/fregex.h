#ifndef FAST_TOKENIZER_H
#define FAST_TOKENIZER_H

#include <stddef.h>
#include <stdio.h>

typedef struct {
	char **tokens;
	size_t *lengths;  // Store length of each token to handle null bytes
	size_t count;
	size_t capacity;
} TokenList;

void tokenlist_init(TokenList *list);
void tokenlist_free(TokenList *list);

// Tokenize input according to the GPT-like regex split semantics
// r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
void tokenize_fast(const char *input, size_t input_len, TokenList *out);

// Utility to print tokens with C-like escaping (one per line):
// <length>\t<escaped-bytes>\n
void print_token_escaped(const char *s, size_t len, FILE *out);
void print_tokens_escaped(const TokenList *list, FILE *out);

#endif // FAST_TOKENIZER_H


