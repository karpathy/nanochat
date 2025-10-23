#ifndef FAST_REGEX_H
#define FAST_REGEX_H

#include <stddef.h>
#include <stdio.h>

typedef struct {
	char **tokens;
	size_t *lengths;
	size_t count;
	size_t capacity;
} TokenList;

void tokenlist_init(TokenList *list);
void tokenlist_free(TokenList *list);

void tokenize_fast(const char *input, size_t input_len, TokenList *out);
void print_token_escaped(const char *s, size_t len, FILE *out);
void print_tokens_escaped(const TokenList *list, FILE *out);

#endif // FAST_REGEX_H


