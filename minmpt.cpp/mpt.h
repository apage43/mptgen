#pragma once
#include "ggml.h"

#include <vector>
#include <map>
#include <string>

// default hparams (MPT 7B)
struct mpt_hparams {
    int32_t n_vocab      = 50432;
    int32_t n_ctx        = 2048;
    int32_t n_embd       = 4096;
    int32_t n_head       = 32;
    int32_t n_layer      = 32;
    float alibi_bias_max = 8;
    float clip_qkv       = 0;
    int32_t expand       = 4;
    int32_t ftype        = 1;
};

struct mpt_layer {
    // normalization
    struct ggml_tensor * norm_1_w;
    struct ggml_tensor * norm_2_w;

    // attention
    struct ggml_tensor * attn_Wqkv_w;
    struct ggml_tensor * attn_out_proj_w;
    
    // ff
    struct ggml_tensor * ffn_up_proj_w;
    struct ggml_tensor * ffn_down_proj_w;
};

struct mpt_model {
    mpt_hparams hparams;

    // token embeddings
    struct ggml_tensor * wte;

    // final norm
    struct ggml_tensor * norm_f_w;

    std::vector<mpt_layer> layers;

    // key + value memory
    struct ggml_tensor * memory_k;
    struct ggml_tensor * memory_v;

    struct ggml_context * ctx;
    std::map<std::string, struct ggml_tensor *> tensors;
};

bool mpt_model_load(const std::string & fname, mpt_model & model, size_t n_ctx_override = 0);
bool mpt_eval(
        const mpt_model & model,
        const int n_threads,
        const int n_past,
        const uint32_t* embd_inp,
        const size_t n_embd_inp,
        float* embd_w,
        size_t& mem_per_token);
bool mpt_eval_cpp(
        const mpt_model & model,
        const int n_threads,
        const int n_past,
        const std::vector<uint32_t> & embd_inp,
        std::vector<float>    & embd_w,
        size_t                & mem_per_token); 