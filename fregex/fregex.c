#include "fregex-2.h"

#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>
#include <stdbool.h>

#include "utf8proc/utf8proc.h"

/*
Regex pattern we care about from nanochat/tokenizer.py SPLIT_PATTERN

Break it down:
A) '(?i:[sdmt]|ll|ve|re)
B) [^\r\n\p{L}\p{N}]?+\p{L}+
C) \p{N}{1,2}
D)  ?[^\s\p{L}\p{N}]++[\r\n]*
E) \s*[\r\n]
F) \s+(?!\S)
G) \s+
*/
  
#define UNICODE_LF    0x000A  // Line Feed
#define UNICODE_CR    0x000D  // Carriage Return

static inline size_t utf8_decode_cp(
    const char *s, 
    const char *end, 
    unsigned int *out_cp
) {
    if (s >= end) { 
        *out_cp = 0; 
        return 0; 
    }
    utf8proc_int32_t ch = 0;
    ssize_t ret = utf8proc_iterate(
        (const utf8proc_uint8_t*)s, 
        (ssize_t)(end - s), 
        &ch
    );
    if (ret < 0) {
        // invalid sequence: treat as single byte
        *out_cp = (unsigned char)*s;
        return 1;
    }
    *out_cp = (unsigned int)ch;
    return (size_t)ret;
}

static inline bool is_utf8_cont_byte(unsigned char b) { 
    return (b & 0xC0) == 0x80; 
}

static inline bool is_cr_or_lf(unsigned int cp) { 
    return cp == UNICODE_LF || cp == UNICODE_CR; 
}

static inline bool is_letter(unsigned int cp) {
    utf8proc_category_t cat = utf8proc_category((utf8proc_int32_t)cp);
    
    switch (cat) {
        case UTF8PROC_CATEGORY_LU: // Letter, Uppercase
        case UTF8PROC_CATEGORY_LL: // Letter, Lowercase
        case UTF8PROC_CATEGORY_LT: // Letter, Titlecase
        case UTF8PROC_CATEGORY_LM: // Letter, Modifier
        case UTF8PROC_CATEGORY_LO: // Letter, Other
            return true;
        default:
            return false;
    }
}

static inline bool is_number(unsigned int cp) {
    utf8proc_category_t cat = utf8proc_category((utf8proc_int32_t)cp);
    switch (cat) {
        case UTF8PROC_CATEGORY_ND: // Number, Decimal Digit
        case UTF8PROC_CATEGORY_NL: // Number, Letter
        case UTF8PROC_CATEGORY_NO: // Number, Other
            return true;
        default:
            return false;
    }
}

static inline bool is_space(unsigned int cp) {
    utf8proc_category_t cat = utf8proc_category((utf8proc_int32_t)cp);

    if (
        cat == UTF8PROC_CATEGORY_ZS ||
        cat == UTF8PROC_CATEGORY_ZL ||
        cat == UTF8PROC_CATEGORY_ZP
    ) {
        return true;
    }

    switch (cp) {
        case 0x0009: // TAB
        case 0x000A: // LF
        case 0x000B: // VT
        case 0x000C: // FF
        case 0x000D: // CR
        case 0x0085: // NEL
            return true;
        default:
            return false;
    }
}

static inline bool is_alnum(unsigned int cp) {
    return is_letter(cp) || is_number(cp);
}

static inline bool is_non_space_letter_number(unsigned int cp) {
    return !is_space(cp) && !is_alnum(cp);
}

static void *xrealloc(void *ptr, size_t new_size) {
	void *p = realloc(ptr, new_size);
	if (!p) {
		fprintf(stderr, "Out of memory while reallocating %zu bytes\n", new_size);
		exit(1);
	}
	return p;
}

static char *xmemdup(const char *src, size_t len) {
	char *s = (char*)malloc(len + 1);
	if (!s) {
		fprintf(stderr, "Out of memory while allocating %zu bytes\n", len + 1);
		exit(1);
	}
	memcpy(s, src, len);
	s[len] = '\0';
	return s;
}

void tokenlist_init(TokenList *list) {
	list->splits = NULL;
	list->count = 0;
	list->capacity = 0;
}

void tokenlist_free(TokenList *list) {
	if (!list)
        return;
    free(list->splits);
    list->splits = NULL;
    list->count = 0;
	list->capacity = 0;
}

static void tokenlist_push(TokenList *list, const char *start, size_t len) {
	if (list->count == list->capacity) {
        const size_t new_cap = list->capacity ? (list->capacity * 2) : 128;
		list->splits = (TokenPos*)xrealloc(list->splits, new_cap * sizeof(TokenPos));
		list->capacity = new_cap;
	}
    /* Write the start / end position of string */
	list->splits[list->count].start = (size_t)start;
    list->splits[list->count].end = (size_t)(start + len); // len - 1 ?
	list->count++;
}

static void fput_escaped_char(unsigned char c, FILE *out) {
	switch (c) {
		case '\\': fputs("\\\\", out); break;
		case '\n': fputs("\\n", out); break;
		case '\r': fputs("\\r", out); break;
		case '\t': fputs("\\t", out); break;
		case '\f': fputs("\\f", out); break;
		case '\v': fputs("\\v", out); break;
		case '\"': fputs("\\\"", out); break;
		default:
			if (c < 32 || c >= 127) {
				fprintf(out, "\\x%02X", c);
			} else {
				fputc(c, out);
			}
	}
}

/* A) '(?i:[sdmt]|ll|ve|re) */
static size_t match_contraction(const char *p, const char *end) {
    if (p >= end || *p != '\'' || (p + 1) >= end) 
        return 0;

    unsigned char a = (unsigned char)p[1];

    // locale-independent lowercase for ASCII letters:
    // map A–Z to a–z; leaves others unchanged
    if (a >= 'A' && a <= 'Z') 
        a = (unsigned char)(a + ('a' - 'A'));

    // 's 'd 'm 't
    if (a == 's' || a == 'd' || a == 'm' || a == 't')
        return 2; 

    // Need a second following byte for 'll 've 're
    if (p + 2 >= end) 
        return 0;

    unsigned char b = (unsigned char)p[2];
    if (b >= 'A' && b <= 'Z') 
        b = (unsigned char)(b + ('a' - 'A'));

    if ((a == 'l' && b == 'l') ||
        (a == 'v' && b == 'e') ||
        (a == 'r' && b == 'e')) {
        return 3;
    }

    return 0;
}

/* B) [^\r\n\p{L}\p{N}]?+\p{L}+ */
static size_t match_word_with_optional_prefix(const char *p, const char *end) {
    if (p >= end) 
        return 0;

    const char *q = p;
    unsigned int cp0; 
    size_t n0 = utf8_decode_cp(q, end, &cp0);
    if (n0 == 0) 
        return 0;

    const char *letters_start = q;   // will point to first letter of \p{L}+
    size_t prefix_bytes = 0;

    // Consider optional one-codepoint prefix if first cp is NOT (CR/LF or alnum)
    if (!is_cr_or_lf(cp0) && !is_alnum(cp0)) {
        // Look ahead: must have at least one letter right after to commit the prefix
        const char *after_prefix = q + n0;
        if (after_prefix >= end) {
            return 0; // no room for required \p{L}+
        }
        unsigned int cp1;
        size_t n1 = utf8_decode_cp(after_prefix, end, &cp1);
        if (n1 > 0 && is_letter(cp1)) {
            // Commit the prefix (possessive) and start letters after it
            prefix_bytes = n0;
            letters_start = after_prefix;
            q = after_prefix;
        } else {
            // Can't commit the prefix (no letter follows), so this branch fails
            return 0;
        }
    } else if (is_letter(cp0)) {
        // No prefix; we already sit on the first letter
        letters_start = q;
    } else {
        // First cp is CR/LF or a number; this branch doesn't match
        return 0;
    }

    // Consume \p{L}+  (require at least one letter)
    size_t letter_count = 0;
    const char *scan = letters_start;
    while (scan < end) {
        unsigned int cp;
        size_t n = utf8_decode_cp(scan, end, &cp);
        if (n == 0 || !is_letter(cp)) 
            break;
        scan += n;
        letter_count++;
    }

    if (letter_count == 0) {
        // Shouldn't happen given the look-ahead, but keep the guard
        return 0;
    }

    return (size_t)(scan - p); // includes prefix (if any) + letters
}

/* C) \p{N}{1,2} */
static size_t match_short_number(const char *p, const char *end) {
    if (p >= end) 
        return 0;

    /* First number required */
    unsigned int cp1; 
    size_t n1 = utf8_decode_cp(p, end, &cp1);
    if (n1 == 0 || !is_number(cp1)) 
        return 0;

    const char *q = p + n1;

    // Optional second number cp
    if (q < end) {
        unsigned int cp2;
        size_t n2 = utf8_decode_cp(q, end, &cp2);
        if (n2 > 0 && is_number(cp2))
            return (size_t)((q + n2) - p); // 2 numbers
    }
    return (size_t)(q - p); // 1 number
}

/* D) ?[^\s\p{L}\p{N}]++[\r\n]* */
static size_t match_punct_run(const char *p, const char *end) {
    const char *q = p;

    /* Optional single ASCII space */
    if (q < end && *q == ' ') {
        const char *r = q + 1;
        if (r >= end) 
            return 0;

        unsigned int cp;
        size_t n = utf8_decode_cp(r, end, &cp);
        if (n == 0) 
            return 0;

        // Eligible b/c not whitespace and not alnum
        if (!is_space(cp) && !is_alnum(cp)) {
            q = r; // commit the space 
        } else {
            // Not followed by eligible punct
            return 0;
        }
    }

    // Now require at least one eligible (not whitespace, not alnum)
    size_t took = 0;
    while (q < end) {
        unsigned int cp;
        size_t n = utf8_decode_cp(q, end, &cp);
        if (n == 0) break;
        if (is_space(cp) || is_alnum(cp)) 
            break; // stop on any whitespace or letter/number
        q += n;
        took++;
    }
    if (took == 0) 
        return 0; // must have at least one punctuation/symbol

    // Finally, optionally absorb CR/LF sequence(s)
    while (q < end) {
        unsigned int cp;
        size_t n = utf8_decode_cp(q, end, &cp);
        if (n == 0 || !is_cr_or_lf(cp)) 
            break;
        q += n;
    }

    return (size_t)(q - p);
}

/* E) \s*[\r\n] */
static size_t match_ws_then_linebreak(const char *p, const char *end) {
    const char *q = p;
    const char *best = NULL;

    // Check boundary before consuming any whitespace, too (zero-length \s*)
    if (q < end) {
        unsigned int nx; 
        size_t nn = utf8_decode_cp(q, end, &nx);
        if (nn > 0 && is_cr_or_lf(nx)) {
            best = q;  // \s* = 0, [\r\n] = this char
        }
    }

    // Scan whitespace; at each boundary, test the next cp
    while (q < end) {
        unsigned int cp; 
        size_t n = utf8_decode_cp(q, end, &cp);
        if (n == 0 || !is_space(cp)) 
            break;
        q += n; // we consumed one whitespace cp; boundary is at q now

        if (q < end) {
            unsigned int nx; 
            size_t nn = utf8_decode_cp(q, end, &nx);
            if (nn > 0 && is_cr_or_lf(nx)) {
                best = q;  // prefer the rightmost usable boundary
            }
        }
    }

    if (!best) return 0;

    // At 'best' the next cp is the CR/LF to include
    unsigned int br; 
    size_t nb = utf8_decode_cp(best, end, &br);
    return (size_t)((best + nb) - p);
}

/* F) \s+(?!\S) */
static size_t match_trailing_ws(const char *p, const char *end) {
    if (p >= end) 
        return 0;

    // First cp must be whitespace
    unsigned int cp; 
    size_t n = utf8_decode_cp(p, end, &cp);
    if (n == 0 || !is_space(cp)) 
        return 0;

    // Consume full whitespace run [p, r)
    const char *r = p + n;
    while (r < end) {
        size_t m = utf8_decode_cp(r, end, &cp);
        if (m == 0 || !is_space(cp)) 
            break;
        r += m;
    }

    if (r == end) {
        // Only whitespace to EOF -> take all of it
        return (size_t)(r - p);
    }

    // Backtrack by exactly one whitespace cp 
    // If the run length is only 1 cp, F must fail.
    // Find the start of the last whitespace cp in [p, r)
    const char *t = r;
    // step back to beginning of previous UTF-8 cp
    do { 
        --t; 
    } while (t > p && is_utf8_cont_byte(*t));

    if (t == p) {
        // run had length 1 cp -> cannot backtrack to keep \s+ >= 1
        return 0;
    }
    // Now [p, t) is k-1 whitespace cps
    return (size_t)(t - p);
}

/* G) \s+ */
static size_t match_ws_run(const char *p, const char *end) {
    if (p >= end) 
        return 0;

    const char *q = p;
    unsigned int cp;
    size_t n = utf8_decode_cp(q, end, &cp);
    if (n == 0 || !is_space(cp)) 
        return 0;

    /* We have at least one whitespace, consume the run */
    q += n;
    while (q < end) {
        size_t m = utf8_decode_cp(q, end, &cp);
        if (m == 0 || !is_space(cp)) 
            break;
        q += m;
    }
    return (size_t)(q - p);
}

void tokenize_fast(const char *input, size_t input_len, TokenList *out) {
	if (!input) {
        out->splits = NULL;
		out->count = 0;
		out->capacity = 0;
		return;
	}
	const char *p = input;
	const char *end = input + input_len;

	while (p < end) {
		/* Special tokens take precedence */
        // TODO LATER
        int captured = 0;
		
        /* Evaluate case A */
        captured = match_contraction(p, end);
        /* Evaluate case B */
        if (!captured) captured = match_word_with_optional_prefix(p, end);
        /* Evaluate case C */
        if (!captured) captured = match_short_number(p, end);
        /* Evaluate case D */
        if (!captured) captured = match_punct_run(p, end);
        /* Evaluate case E */
        if (!captured) captured = match_ws_then_linebreak(p, end); 
        /* Evaluate case F */
        if (!captured) captured = match_trailing_ws(p, end);
        /* Evaluate case G */
        if (!captured) captured = match_ws_run(p, end);

        if (captured) {
            tokenlist_push(out, p, captured);
            p += captured;
        }
		else {
			/* Fallback take a single CP */
            unsigned int cp;
            size_t adv = utf8_decode_cp(p, end, &cp);
            if (adv == 0) 
                adv = 1;
            tokenlist_push(out, p, adv);
            p += adv;
		}
	}
}


