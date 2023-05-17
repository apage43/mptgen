#include "mpt.h"
#include "minmpt.h"

#include <unistd.h>
#include <stdlib.h>
#include <string.h>

struct minmpt_session
{
    mpt_model model;
    size_t mem_per_token = 0;
    int n_threads = 4;
    size_t n_past = 0;
};

static minmpt_session *from_handle(minmpt_handle h)
{
    return reinterpret_cast<minmpt_session *>(h);
}

minmpt_error minmpt_load(minmpt_handle *handle, const char *filename, size_t fnlen, size_t n_ctx_override)
{
    auto modelp = new minmpt_session;
    std::string fn(filename, fnlen);
    try
    {
        if (mpt_model_load(fn, modelp->model, n_ctx_override))
        {
            *handle = reinterpret_cast<minmpt_handle>(modelp);
            return MINMPT_OK;
        }
    }
    catch (...)
    {
        delete modelp;
    }
    return MINMPT_FAILURE;
}

size_t minmpt_n_vocab(minmpt_handle handle)
{
    auto modelp = from_handle(handle);
    return modelp->model.hparams.n_vocab;
}

size_t minmpt_n_ctx(minmpt_handle handle)
{
    auto modelp = from_handle(handle);
    return modelp->model.hparams.n_ctx;
}

size_t minmpt_n_past(minmpt_handle handle)
{
    auto modelp = from_handle(handle);
    return modelp->n_past;
}

void minmpt_rewind(minmpt_handle handle, size_t n)
{
    auto modelp = from_handle(handle);
    if (modelp->n_past > n)
    {
        modelp->n_past -= n;
    }
    else
    {
        modelp->n_past = 0;
    }
}

void minmpt_reset_ctx(minmpt_handle handle)
{
    auto modelp = from_handle(handle);
    modelp->n_past = 0;
}

void minmpt_set_n_threads(minmpt_handle handle, unsigned int n_threads) {
    auto modelp = from_handle(handle);
    modelp->n_threads = n_threads > 0 ? n_threads : 4;
}

minmpt_error minmpt_eval_logits(minmpt_handle handle,
                                const uint32_t *tokens,
                                size_t n_tokens,
                                float *logits)
{
    auto modelp = from_handle(handle);
    if (modelp->n_past + n_tokens > modelp->model.hparams.n_ctx) {
        return MINMPT_CTX_LIMIT;
    }
    if (modelp->mem_per_token == 0)
    {
        std::vector<float> dummy_logits;
        mpt_eval_cpp(modelp->model, 8, 0, {1, 2, 3, 4}, dummy_logits, modelp->mem_per_token);
    }
    if (!mpt_eval(modelp->model, modelp->n_threads, modelp->n_past, tokens, n_tokens, logits, modelp->mem_per_token))
    {
        printf("Failed to predict\n");
        return MINMPT_FAILURE;
    }
    modelp->n_past += n_tokens;
    return MINMPT_OK;
}
void minmpt_free(minmpt_handle handle)
{
    auto modelp = from_handle(handle);
    if (modelp && modelp->model.ctx)
    {
        ggml_free(modelp->model.ctx);
    }
    delete modelp;
}