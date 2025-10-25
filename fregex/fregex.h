#ifndef FAST_REGEX_H
#define FAST_REGEX_H

#include <stddef.h>
#include <stdio.h>

typedef struct {
	size_t start, end;
} TokenPos;

typedef struct {
    TokenPos *splits;
	size_t count;
	size_t capacity;
} TokenList;

void tokenlist_init(TokenList *list);
void tokenlist_free(TokenList *list);
void tokenize_fast(const char *input, size_t input_len, TokenList *out);

#endif // FAST_REGEX_H
