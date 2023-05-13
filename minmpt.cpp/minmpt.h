#pragma once

#include <stdint.h>
#include <stddef.h>

// errors
#define MINMPT_OK 0
#define MINMPT_INVALID 1
#define MINMPT_FAILURE 2
#define MINMPT_CTX_LIMIT 3

#ifdef __cplusplus
extern "C" {
#endif
typedef void * minmpt_handle;
typedef int minmpt_error;
minmpt_error minmpt_load(minmpt_handle* handle, const char* filename, size_t fnlen);
size_t minmpt_n_vocab(minmpt_handle handle);
size_t minmpt_n_past(minmpt_handle handle);
void minmpt_rewind(minmpt_handle handle, size_t n);
size_t minmpt_n_ctx(minmpt_handle handle);
void minmpt_reset_ctx(minmpt_handle handle);
minmpt_error minmpt_eval_logits(minmpt_handle handle,
    const uint32_t *tokens, size_t n_tokens, float* logits);
void minmpt_free(minmpt_handle handle);
#ifdef __cplusplus
}
#endif