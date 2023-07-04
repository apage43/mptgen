#include "minmpt.h"
#include "mpt.h"

#include <memory>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

struct minmpt_session {
  std::shared_ptr<mpt_model> model;
  std::unique_ptr<mpt_kvcache> kvcache;
  size_t mem_per_token = 0;
  int n_threads = 4;
  size_t n_past = 0;
};

static minmpt_session *from_handle(minmpt_handle h) {
  return reinterpret_cast<minmpt_session *>(h);
}

static const minmpt_session *from_handle_const(const minmpt_handle h) {
  return reinterpret_cast<const minmpt_session *>(h);
}

minmpt_error minmpt_load(minmpt_handle *handle, const char *filename,
                         size_t fnlen, size_t n_ctx_override) {
  auto modelp = new minmpt_session;
  std::string fn(filename, fnlen);
  try {
    modelp->model = std::make_shared<mpt_model>();
    if (mpt_model_load(fn, *modelp->model, n_ctx_override)) {
      modelp->kvcache = std::make_unique<mpt_kvcache>(*modelp->model);
      *handle = reinterpret_cast<minmpt_handle>(modelp);
      return MINMPT_OK;
    }
  } catch (...) {
    delete modelp;
  }
  return MINMPT_FAILURE;
}

void minmpt_fork(minmpt_handle handle, minmpt_handle *child) {
  auto modelp = from_handle_const(handle);
  auto newp = new minmpt_session;
  newp->mem_per_token = modelp->mem_per_token;
  newp->model = modelp->model;
  newp->kvcache = std::make_unique<mpt_kvcache>(*modelp->model);
  memcpy(newp->kvcache->memory_k->data, modelp->kvcache->memory_k->data,
         ggml_nbytes(newp->kvcache->memory_k));
  memcpy(newp->kvcache->memory_v->data, modelp->kvcache->memory_v->data,
         ggml_nbytes(newp->kvcache->memory_v));
  newp->n_past = modelp->n_past;
  *child = reinterpret_cast<minmpt_handle>(newp);
}

size_t minmpt_n_vocab(minmpt_handle handle) {
  auto modelp = from_handle(handle);
  return modelp->model->hparams.n_vocab;
  ;
}

size_t minmpt_n_ctx(minmpt_handle handle) {
  auto modelp = from_handle(handle);
  return modelp->model->hparams.n_ctx;
}

size_t minmpt_n_past(minmpt_handle handle) {
  auto modelp = from_handle(handle);
  return modelp->n_past;
}

void minmpt_rewind(minmpt_handle handle, size_t n) {
  auto modelp = from_handle(handle);
  if (modelp->n_past > n) {
    modelp->n_past -= n;
  } else {
    modelp->n_past = 0;
  }
}

void minmpt_reset_ctx(minmpt_handle handle) {
  auto modelp = from_handle(handle);
  modelp->n_past = 0;
}

void minmpt_set_n_threads(minmpt_handle handle, unsigned int n_threads) {
  auto modelp = from_handle(handle);
  modelp->n_threads = n_threads > 0 ? n_threads : 4;
}

minmpt_error minmpt_eval_logits(minmpt_handle handle, const uint32_t *tokens,
                                size_t n_tokens, float *logits) {
  auto modelp = from_handle(handle);
  if (modelp->n_past + n_tokens > (size_t)modelp->model->hparams.n_ctx) {
    return MINMPT_CTX_LIMIT;
  }
  if (modelp->mem_per_token == 0) {
    std::vector<float> dummy_logits;
    mpt_eval_cpp(*modelp->model, *modelp->kvcache, modelp->n_threads, 0,
                 {1, 2, 3, 4}, dummy_logits, modelp->mem_per_token);
  }
  if (!mpt_eval(*modelp->model, *modelp->kvcache, modelp->n_threads,
                modelp->n_past, tokens, n_tokens, logits,
                modelp->mem_per_token)) {
    printf("Failed to predict\n");
    return MINMPT_FAILURE;
  }
  modelp->n_past += n_tokens;
  return MINMPT_OK;
}
void minmpt_free(minmpt_handle handle) {
  auto modelp = from_handle(handle);
  delete modelp;
}
