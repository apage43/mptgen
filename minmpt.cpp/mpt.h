#pragma once
#include "ggml.h"

#include <map>
#include <string>
#include <vector>

// default hparams (MPT 7B)
struct mpt_hparams {
  int32_t n_vocab = 50432;
  int32_t n_ctx = 2048;
  int32_t n_embd = 4096;
  int32_t n_head = 32;
  int32_t n_layer = 32;
  float alibi_bias_max = 8;
  float clip_qkv = 0;
  int32_t expand = 4;
  int32_t ftype = 1;
};

struct mpt_layer {
  // normalization
  struct ggml_tensor *norm_1_w;
  struct ggml_tensor *norm_2_w;

  // attention
  struct ggml_tensor *attn_Wqkv_w;
  struct ggml_tensor *attn_out_proj_w;

  // ff
  struct ggml_tensor *ffn_up_proj_w;
  struct ggml_tensor *ffn_down_proj_w;
};

struct mpt_model {
  mpt_hparams hparams;

  // token embeddings
  struct ggml_tensor *wte;

  // final norm
  struct ggml_tensor *norm_f_w;

  std::vector<mpt_layer> layers;

  struct ggml_context *ctx;
  std::map<std::string, struct ggml_tensor *> tensors;

  ~mpt_model() {
    if (ctx) {
      ggml_free(ctx);
    }
  }
};

// key + value memory
struct mpt_kvcache {
  mpt_kvcache(mpt_model &model) {
    const auto &hparams = model.hparams;
    const int n_embd = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_ctx = hparams.n_ctx;

    size_t ctx_size = 0;
    ctx_size += (size_t)n_ctx * n_layer * n_embd *
                ggml_type_size(GGML_TYPE_F16); // memory_k
    ctx_size += (size_t)n_ctx * n_layer * n_embd *
                ggml_type_size(GGML_TYPE_F16); // memory_v
    ctx_size += ggml_tensor_overhead() * 2;
    ggml_init_params params{ctx_size, nullptr, 0};
    ctx = ggml_init(params);

    const int n_mem = n_layer * n_ctx;
    const size_t n_elements = (size_t)n_embd * n_mem;
    memory_k = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, n_elements);
    memory_v = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, n_elements);
    const size_t memory_size = ggml_nbytes(memory_k) + ggml_nbytes(memory_v);
    printf("%s: memory_size = %8.2f MB, n_mem = %d\n", __func__,
           memory_size / 1024.0 / 1024.0, n_mem);
  }

  ~mpt_kvcache() {
    if (ctx) {
      ggml_free(ctx);
    }
  }

  struct ggml_tensor *memory_k;
  struct ggml_tensor *memory_v;
  struct ggml_context *ctx;
};

bool mpt_model_load(const std::string &fname, mpt_model &model,
                    size_t n_ctx_override = 0);
bool mpt_eval(const mpt_model &model, mpt_kvcache &kvcache, const int n_threads,
              const int n_past, const uint32_t *embd_inp,
              const size_t n_embd_inp, float *embd_w, size_t &mem_per_token);
bool mpt_eval_cpp(const mpt_model &model, mpt_kvcache &kvcache,
                  const int n_threads, const int n_past,
                  const std::vector<uint32_t> &embd_inp,
                  std::vector<float> &embd_w, size_t &mem_per_token);
