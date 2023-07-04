#include "mpt.h"
#include "mpt-util.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <string>
#include <unistd.h>
#include <vector>

enum minmpt_format_version { minmpt_format_v1_no_vocab = 0 };

// load the model's weights from a file
bool mpt_model_load(const std::string &fname, mpt_model &model,
                    size_t n_ctx_override) {
  printf("%s: loading model from '%s' - please wait ...\n", __func__,
         fname.c_str());

  auto mptf = mpt_file(fname.c_str(), "rb");

  // verify magic
  {
    uint32_t magic = mptf.read_u32();
    uint32_t version = mptf.read_u32();
    if (magic != 0x67676d64) { // GGMD
      fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__,
              fname.c_str());
      return false;
    }
    if (version != minmpt_format_v1_no_vocab) {
      fprintf(stderr, "%s: invalid file format version %d, expected %d\n",
              __func__, version, minmpt_format_v1_no_vocab);
      return false;
    }
  }

  // load hparams
  {
    auto &hparams = model.hparams;

    mptf.read_raw(&hparams.n_vocab, sizeof(hparams.n_vocab));
    mptf.read_raw(&hparams.n_ctx, sizeof(hparams.n_ctx));
    mptf.read_raw(&hparams.n_layer, sizeof(hparams.n_layer));
    mptf.read_raw(&hparams.n_head, sizeof(hparams.n_head));
    mptf.read_raw(&hparams.n_embd, sizeof(hparams.n_embd));
    mptf.read_raw(&hparams.alibi_bias_max, sizeof(hparams.alibi_bias_max));
    mptf.read_raw(&hparams.clip_qkv, sizeof(hparams.clip_qkv));
    mptf.read_raw(&hparams.ftype, sizeof(hparams.ftype));

    printf("%s: n_vocab        = %d\n", __func__, hparams.n_vocab);
    if (n_ctx_override != 0) {
      hparams.n_ctx = n_ctx_override;
      printf("%s: n_ctx (forced) = %d\n", __func__, hparams.n_ctx);
    } else {
      printf("%s: n_ctx          = %d\n", __func__, hparams.n_ctx);
    }
    printf("%s: n_embd         = %d\n", __func__, hparams.n_embd);
    printf("%s: n_head         = %d\n", __func__, hparams.n_head);
    printf("%s: n_layer        = %d\n", __func__, hparams.n_layer);
    printf("%s: alibi_bias_max = %f\n", __func__, hparams.alibi_bias_max);
    printf("%s: clip_qkv       = %f\n", __func__, hparams.clip_qkv);
    printf("%s: ftype          = %d\n", __func__, hparams.ftype);
  }

  // for the big tensors, we have the option to store the data in 16-bit floats
  // or quantized in order to save memory and also to speed up the computation
  ggml_type wtype = ggml_ftype_to_ggml_type((ggml_ftype)(model.hparams.ftype));
  if (wtype == GGML_TYPE_COUNT) {
    fprintf(stderr, "%s: invalid model file '%s' (bad ftype value %d)\n",
            __func__, fname.c_str(), model.hparams.ftype);
    return false;
  }

  auto &ctx = model.ctx;

  size_t ctx_size = 0;

  {
    const auto &hparams = model.hparams;

    const int n_embd = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_ctx = hparams.n_ctx;
    const int n_vocab = hparams.n_vocab;
    const int expand = hparams.expand;

    ctx_size += n_embd * ggml_type_size(GGML_TYPE_F32); // ln_f_w

    ctx_size += n_embd * n_vocab * ggml_type_size(GGML_TYPE_F32); // wte

    ctx_size += n_layer * (n_embd * ggml_type_size(GGML_TYPE_F32)); // norm_1_w
    ctx_size += n_layer * (n_embd * ggml_type_size(GGML_TYPE_F32)); // norm_2_w

    ctx_size +=
        n_layer * (3 * n_embd * n_embd * ggml_type_sizef(wtype)); // attn_Wqkv_w
    ctx_size +=
        n_layer * (n_embd * n_embd * ggml_type_sizef(wtype)); // attn_out_proj_w

    ctx_size += n_layer * (expand * n_embd * n_embd *
                           ggml_type_sizef(wtype)); // ffn_up_proj_w
    ctx_size += n_layer * (expand * n_embd * n_embd *
                           ggml_type_sizef(wtype)); // ffn_down_proj_w

    ctx_size += (size_t)n_ctx * n_layer * n_embd *
                ggml_type_size(GGML_TYPE_F16); // memory_k
    ctx_size += (size_t)n_ctx * n_layer * n_embd *
                ggml_type_size(GGML_TYPE_F16); // memory_v

    // TODO probably less now?
    ctx_size += (5 + 10 * n_layer) * 256; // object overhead

    printf("%s: ggml ctx size = %6.2f MB\n", __func__,
           ctx_size / (1024.0 * 1024.0));
  }

  // create the ggml context
  {
    struct ggml_init_params params = {
        /* .mem_size   = */ ctx_size,
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ false,
    };

    model.ctx = ggml_init(params);
    if (!model.ctx) {
      fprintf(stderr, "%s: ggml_init() failed\n", __func__);
      return false;
    }
  }

  // prepare memory for the weights
  {
    const auto &hparams = model.hparams;

    const int n_embd = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    // const int n_ctx = hparams.n_ctx;
    const int n_vocab = hparams.n_vocab;
    const int expand = hparams.expand;

    model.layers.resize(n_layer);

    model.wte = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_vocab);
    model.norm_f_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

    // map by name
    model.tensors["transformer.wte.weight"] = model.wte;
    model.tensors["transformer.norm_f.weight"] = model.norm_f_w;

    for (int i = 0; i < n_layer; ++i) {
      auto &layer = model.layers[i];

      layer.norm_1_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
      layer.norm_2_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

      layer.attn_Wqkv_w = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd * 3);
      layer.attn_out_proj_w = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);
      layer.ffn_up_proj_w =
          ggml_new_tensor_2d(ctx, wtype, n_embd, expand * n_embd);
      layer.ffn_down_proj_w =
          ggml_new_tensor_2d(ctx, wtype, expand * n_embd, n_embd);

      // map by name
      model.tensors["transformer.blocks." + std::to_string(i) +
                    ".norm_1.weight"] = layer.norm_1_w;
      model.tensors["transformer.blocks." + std::to_string(i) +
                    ".norm_2.weight"] = layer.norm_2_w;
      model.tensors["transformer.blocks." + std::to_string(i) +
                    ".attn.Wqkv.weight"] = layer.attn_Wqkv_w;
      model.tensors["transformer.blocks." + std::to_string(i) +
                    ".attn.out_proj.weight"] = layer.attn_out_proj_w;

      model.tensors["transformer.blocks." + std::to_string(i) +
                    ".ffn.up_proj.weight"] = layer.ffn_up_proj_w;
      model.tensors["transformer.blocks." + std::to_string(i) +
                    ".ffn.down_proj.weight"] = layer.ffn_down_proj_w;
    }
  }

  // key + value memory
  {
    const auto &hparams = model.hparams;

    const int n_embd = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_ctx = hparams.n_ctx;

    const int n_mem = n_layer * n_ctx;
    const size_t n_elements = (size_t)n_embd * n_mem;

    model.memory_k = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, n_elements);
    model.memory_v = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, n_elements);

    const size_t memory_size =
        ggml_nbytes(model.memory_k) + ggml_nbytes(model.memory_v);

    printf("%s: memory_size = %8.2f MB, n_mem = %d\n", __func__,
           memory_size / 1024.0 / 1024.0, n_mem);
  }

  // load weights
  {
    int n_tensors = 0;
    size_t total_size = 0;

    printf("%s: ", __func__);

    while (mptf.tell() < mptf.size) {
      int32_t n_dims;
      int32_t length;
      int32_t ttype;

      mptf.read_raw(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
      mptf.read_raw(reinterpret_cast<char *>(&length), sizeof(length));
      mptf.read_raw(reinterpret_cast<char *>(&ttype), sizeof(ttype));

      int32_t nelements = 1;
      std::vector<uint32_t> ne;
      ne.resize(n_dims);
      mptf.read_raw(ne.data(), sizeof(ne[0]) * n_dims);
      for (auto &dim : ne) {
        nelements *= dim;
      }

      std::string name = mptf.read_string(length);

      if (model.tensors.find(name.data()) == model.tensors.end()) {
        fprintf(stderr, "%s: unknown tensor '%s' in model file\n", __func__,
                name.data());
        return false;
      }

      auto tensor = model.tensors[name.data()];
      if (ggml_nelements(tensor) != nelements) {
        fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n",
                __func__, name.data());
        return false;
      }

      if (tensor->ne[0] != ne[0] || (n_dims > 1 && tensor->ne[1] != ne[1])) {
        fprintf(stderr,
                "%s: tensor '%s' has wrong shape in model file: got [%d, %d], "
                "expected [%d, %d]\n",
                __func__, name.data(), (int)tensor->ne[0], (int)tensor->ne[1],
                ne[0], ne[1]);
        return false;
      }

      // for debugging
      if (0) {
        printf("%24s - [%5d, %5d], type = %6s, %6.2f MB, %9zu bytes\n",
               name.data(), ne[0], ne[1], ggml_type_name(ggml_type(ttype)),
               ggml_nbytes(tensor) / 1024.0 / 1024.0, ggml_nbytes(tensor));
      }

      const size_t bpe = ggml_type_size(ggml_type(ttype));

      if ((nelements * bpe) / ggml_blck_size(tensor->type) !=
          ggml_nbytes(tensor)) {
        fprintf(stderr,
                "%s: tensor '%s' has wrong size in model file: got %zu, "
                "expected %zu\n",
                __func__, name.data(), ggml_nbytes(tensor), nelements * bpe);
        return false;
      }

      mptf.read_raw(reinterpret_cast<char *>(tensor->data),
                    ggml_nbytes(tensor));

      // printf("%42s - [%5d, %5d], type = %6s, %6.2f MB\n", name.data(), ne[0],
      // ne[1], ttype == 0 ? "float" : "f16",
      // ggml_nbytes(tensor)/1024.0/1024.0);
      total_size += ggml_nbytes(tensor);
      if (++n_tensors % 8 == 0) {
        printf(".");
        fflush(stdout);
      }
    }

    printf(" done\n");

    printf("%s: model size = %8.2f MB / num tensors = %d\n", __func__,
           total_size / 1024.0 / 1024.0, n_tensors);
  }

  return true;
}

// evaluate the transformer
//
//   - model:     the model
//   - n_threads: number of threads to use
//   - n_past:    the context size so far
//   - embd_inp:  the embeddings of the tokens in the context
//   - embd_w:    the predicted logits for the next token
//
// The GPT-J model requires about 16MB of memory per input token.
//

bool mpt_eval(const mpt_model &model, const int n_threads, const int n_past,
              const uint32_t *embd_inp, const size_t n_embd_inp, float *embd_w,
              size_t &mem_per_token) {
  const int N = n_embd_inp;

  const auto &hparams = model.hparams;

  const int n_embd = hparams.n_embd;
  const int n_layer = hparams.n_layer;
  const int n_ctx = hparams.n_ctx;
  const int n_head = hparams.n_head;
  const int n_vocab = hparams.n_vocab;

  size_t buf_size = 256u * 1024 * 1024;

  // TODO - better approach to guess required mem.
  // mem_per_token is kind of hacky, N is too small at some point, N + n_past
  // seems to work up to 2048 tokens at least.  llama.cpp now just hardcodes
  // some known good values for given model sizes
  if (mem_per_token > 0) {
    const size_t buf_size_new =
        1.1 * (mem_per_token * 1.3 * N) +
        (mem_per_token * n_past); // add 10% to account for ggml object overhead
    if (buf_size_new > buf_size) {
      buf_size = buf_size_new;
    }
  }

  struct ggml_init_params params = {
      .mem_size = buf_size,
      .mem_buffer = nullptr,
      .no_alloc = false,
  };

  struct ggml_context *ctx0 = ggml_init(params);
  struct ggml_cgraph gf = {.n_threads = n_threads};

  struct ggml_tensor *embd = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
  memcpy(embd->data, embd_inp, N * ggml_element_size(embd));

  // wte
  struct ggml_tensor *inpL = ggml_get_rows(ctx0, model.wte, embd);

  for (int il = 0; il < n_layer; ++il) {

    struct ggml_tensor *inpSA = inpL;
    struct ggml_tensor *cur = inpSA;
    // self-attention
    {

      // norm1
      cur = ggml_norm(ctx0, cur);
      cur = ggml_mul(ctx0, ggml_repeat(ctx0, model.layers[il].norm_1_w, cur),
                     cur);
      // compute QKV
      cur = ggml_mul_mat(ctx0, model.layers[il].attn_Wqkv_w, cur);

      if (model.hparams.clip_qkv > 0.0f) {
        cur = ggml_clamp(ctx0, cur, -model.hparams.clip_qkv,
                         model.hparams.clip_qkv);
      }
      struct ggml_tensor *Qcur =
          ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd, N, cur->nb[1],
                                       0 * ggml_element_size(cur) * n_embd));
      struct ggml_tensor *Kcur =
          ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd, N, cur->nb[1],
                                       1 * ggml_element_size(cur) * n_embd));
      struct ggml_tensor *Vcur =
          ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd, N, cur->nb[1],
                                       2 * ggml_element_size(cur) * n_embd));

      // TODO: qk_ln? (seems to be False in MPT-7B configs)
      {
        struct ggml_tensor *k =
            ggml_view_1d(ctx0, model.memory_k, N * n_embd,
                         (ggml_element_size(model.memory_k) * n_embd) *
                             (il * n_ctx + n_past));
        struct ggml_tensor *v =
            ggml_view_1d(ctx0, model.memory_v, N * n_embd,
                         (ggml_element_size(model.memory_v) * n_embd) *
                             (il * n_ctx + n_past));

        ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Kcur, k));
        ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Vcur, v));
      }
      // Q = Qcur.contiguous().view(n_embd/n_head, n_head, N).permute(0, 2, 1,
      // 3)
      struct ggml_tensor *Q = ggml_permute(
          ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd / n_head, n_head, N), 0, 2,
          1, 3);

      struct ggml_tensor *K = ggml_permute(
          ctx0,
          ggml_reshape_3d(
              ctx0,
              ggml_view_1d(ctx0, model.memory_k, (n_past + N) * n_embd,
                           il * n_ctx * ggml_element_size(model.memory_k) *
                               n_embd),
              n_embd / n_head, n_head, n_past + N),
          0, 2, 1, 3);

      // K * Q
      struct ggml_tensor *KQ = ggml_mul_mat(ctx0, K, Q);

      // KQ_scaled = KQ / sqrt(n_embd/n_head)
      struct ggml_tensor *KQ_scaled = ggml_scale(
          ctx0, KQ, ggml_new_f32(ctx0, 1.0f / sqrt(float(n_embd) / n_head)));

      // Alibi
      struct ggml_tensor *KQ_scaled_biased =
          ggml_alibi(ctx0, ggml_cont(ctx0, KQ_scaled), n_past, n_head,
                     model.hparams.alibi_bias_max);
      ggml_set_name(KQ_scaled_biased, "alibi");

      // KQ_masked = mask_past(KQ_scaled)
      struct ggml_tensor *KQ_masked =
          ggml_diag_mask_inf(ctx0, KQ_scaled_biased, n_past);

      // KQ = soft_max(KQ_masked)
      struct ggml_tensor *KQ_soft_max = ggml_soft_max(ctx0, KQ_masked);

      // V_trans = Vmem.view(n_embd/n_head, n_head, n_past + N).permute(1, 2, 0,
      // 3).contiguous()
      struct ggml_tensor *V_trans = ggml_cpy(
          ctx0,
          ggml_permute(
              ctx0,
              ggml_reshape_3d(
                  ctx0,
                  ggml_view_1d(ctx0, model.memory_v, (n_past + N) * n_embd,
                               il * n_ctx * ggml_element_size(model.memory_v) *
                                   n_embd),
                  n_embd / n_head, n_head, n_past + N),
              1, 2, 0, 3),
          ggml_new_tensor_3d(ctx0, model.memory_v->type, n_past + N,
                             n_embd / n_head, n_head));

      // KQV = transpose(V) * KQ_soft_max
      struct ggml_tensor *KQV = ggml_mul_mat(ctx0, V_trans, KQ_soft_max);

      // KQV_merged = KQV.permute(0, 2, 1, 3)
      struct ggml_tensor *KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);

      // cur = KQV_merged.contiguous().view(n_embd, N)
      cur = ggml_cpy(ctx0, KQV_merged,
                     ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N));

      // projection (no bias)
      cur = ggml_mul_mat(ctx0, model.layers[il].attn_out_proj_w, cur);
    }

    // residual
    struct ggml_tensor *resSA = ggml_add(ctx0, cur, inpSA);
    // feed-forward network
    {
      cur = resSA;
      // norm2
      cur = ggml_norm(ctx0, cur);
      cur = ggml_mul(ctx0, ggml_repeat(ctx0, model.layers[il].norm_2_w, cur),
                     cur);
      // ffn
      cur = ggml_mul_mat(ctx0, model.layers[il].ffn_up_proj_w, cur);
      cur = ggml_gelu(ctx0, cur);
      cur = ggml_mul_mat(ctx0, model.layers[il].ffn_down_proj_w, cur);
    }

    // self-attention + FF
    inpL = ggml_add(ctx0, cur, resSA);
  }

  struct ggml_tensor *out = inpL;
  // -> logits
  {
    out = ggml_norm(ctx0, out);
    out = ggml_mul(ctx0, ggml_repeat(ctx0, model.norm_f_w, out), out);
    out = ggml_mul_mat(ctx0, model.wte, out);
  }

  // run the computation
  ggml_build_forward_expand(&gf, out);
  ggml_graph_compute(ctx0, &gf);

  // return result for just the last token
  memcpy(embd_w, (float *)ggml_get_data(out) + (n_vocab * (N - 1)),
         sizeof(float) * n_vocab);

  if (mem_per_token == 0) {
    mem_per_token = ggml_used_mem(ctx0) / N;
  }
  // printf("used_mem = %zu\n", ggml_used_mem(ctx0));

  ggml_free(ctx0);

  return true;
}

bool mpt_eval_cpp(const mpt_model &model, const int n_threads, const int n_past,
                  const std::vector<uint32_t> &embd_inp,
                  std::vector<float> &embd_w, size_t &mem_per_token) {
  embd_w.resize(model.hparams.n_vocab);
  return mpt_eval(model, n_threads, n_past, embd_inp.data(), embd_inp.size(),
                  embd_w.data(), mem_per_token);
}
