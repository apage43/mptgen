#include "ggml.h"
extern "C" {
#include "k_quants.h"
}

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <regex>
#include <string>
#include <vector>

// begin ggml/examples/common-ggml.cpp
#include <map>
#include <regex>

static const std::map<std::string, enum ggml_ftype> GGML_FTYPE_MAP = {
    {"q4_0", GGML_FTYPE_MOSTLY_Q4_0}, {"q4_1", GGML_FTYPE_MOSTLY_Q4_1},
    {"q5_0", GGML_FTYPE_MOSTLY_Q5_0}, {"q5_1", GGML_FTYPE_MOSTLY_Q5_1},
    {"q8_0", GGML_FTYPE_MOSTLY_Q8_0},
    // k-quants don't seem to work on mpt for me - maybe bad sizes?
    // {"q3_k", GGML_FTYPE_MOSTLY_Q3_K},
    // {"q4_k", GGML_FTYPE_MOSTLY_Q4_K},
    // {"q6_k", GGML_FTYPE_MOSTLY_Q6_K},
};

void ggml_print_ftypes(FILE *fp) {
  for (auto it = GGML_FTYPE_MAP.begin(); it != GGML_FTYPE_MAP.end(); it++) {
    fprintf(fp, "  type = \"%s\" or %d\n", it->first.c_str(), it->second);
  }
}

enum ggml_ftype ggml_parse_ftype(const char *str) {
  enum ggml_ftype ftype;
  if (str[0] == 'q') {
    const auto it = GGML_FTYPE_MAP.find(str);
    if (it == GGML_FTYPE_MAP.end()) {
      fprintf(stderr, "%s: unknown ftype '%s'\n", __func__, str);
      return GGML_FTYPE_UNKNOWN;
    }
    ftype = it->second;
  } else {
    ftype = (enum ggml_ftype)atoi(str);
  }

  return ftype;
}

bool ggml_common_quantize_0(std::ifstream &finp, std::ofstream &fout,
                            const ggml_ftype ftype,
                            const std::vector<std::string> &to_quant,
                            const std::vector<std::string> &to_skip) {

  ggml_type qtype = GGML_TYPE_F32;

  switch (ftype) {
  case GGML_FTYPE_MOSTLY_Q4_0:
    qtype = GGML_TYPE_Q4_0;
    break;
  case GGML_FTYPE_MOSTLY_Q4_1:
    qtype = GGML_TYPE_Q4_1;
    break;
  case GGML_FTYPE_MOSTLY_Q5_0:
    qtype = GGML_TYPE_Q5_0;
    break;
  case GGML_FTYPE_MOSTLY_Q5_1:
    qtype = GGML_TYPE_Q5_1;
    break;
  case GGML_FTYPE_MOSTLY_Q8_0:
    qtype = GGML_TYPE_Q8_0;
    break;
  case GGML_FTYPE_MOSTLY_Q2_K:
    qtype = GGML_TYPE_Q2_K;
    break;
  case GGML_FTYPE_MOSTLY_Q3_K:
    qtype = GGML_TYPE_Q3_K;
    break;
  case GGML_FTYPE_MOSTLY_Q4_K:
    qtype = GGML_TYPE_Q4_K;
    break;
  case GGML_FTYPE_MOSTLY_Q5_K:
    qtype = GGML_TYPE_Q5_K;
    break;
  case GGML_FTYPE_MOSTLY_Q6_K:
    qtype = GGML_TYPE_Q6_K;
    break;
  case GGML_FTYPE_UNKNOWN:
  case GGML_FTYPE_ALL_F32:
  case GGML_FTYPE_MOSTLY_F16:
  case GGML_FTYPE_MOSTLY_Q4_1_SOME_F16: {
    fprintf(stderr, "%s: invalid model type %d\n", __func__, ftype);
    return false;
  }
  };

  if (!ggml_is_quantized(qtype)) {
    fprintf(stderr, "%s: invalid quantization type %d (%s)\n", __func__, qtype,
            ggml_type_name(qtype));
    return false;
  }

  size_t total_size_org = 0;
  size_t total_size_new = 0;

  std::vector<float> work;

  std::vector<uint8_t> data_u8;
  std::vector<ggml_fp16_t> data_f16;
  std::vector<float> data_f32;

  std::vector<int64_t> hist_all(1 << 4, 0);

  while (true) {
    int32_t n_dims;
    int32_t length;
    int32_t ttype;

    finp.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
    finp.read(reinterpret_cast<char *>(&length), sizeof(length));
    finp.read(reinterpret_cast<char *>(&ttype), sizeof(ttype));

    if (finp.eof()) {
      break;
    }

    int32_t nelements = 1;
    int32_t ne[4] = {1, 1, 1, 1};
    for (int i = 0; i < n_dims; ++i) {
      finp.read(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
      nelements *= ne[i];
    }

    std::string name(length, 0);
    finp.read(&name[0], length);

    printf("%64s - [%5d, %5d, %5d], type = %6s ", name.data(), ne[0], ne[1],
           ne[2], ggml_type_name((ggml_type)ttype));

    bool quantize = false;

    // check if we should quantize this tensor
    for (const auto &s : to_quant) {
      if (std::regex_match(name, std::regex(s))) {
        quantize = true;
        break;
      }
    }

    // check if we should skip this tensor
    for (const auto &s : to_skip) {
      if (std::regex_match(name, std::regex(s))) {
        quantize = false;
        break;
      }
    }

    // quantize only 2D tensors
    quantize &= (n_dims == 2);

    if (quantize) {
      if (ttype != GGML_TYPE_F32 && ttype != GGML_TYPE_F16) {
        fprintf(stderr,
                "%s: unsupported ttype %d (%s) for integer quantization\n",
                __func__, ttype, ggml_type_name((ggml_type)ttype));
        return false;
      }

      if (ttype == GGML_TYPE_F16) {
        data_f16.resize(nelements);
        finp.read(reinterpret_cast<char *>(data_f16.data()),
                  nelements * sizeof(ggml_fp16_t));
        data_f32.resize(nelements);
        for (int i = 0; i < nelements; ++i) {
          data_f32[i] = ggml_fp16_to_fp32(data_f16[i]);
        }
      } else {
        data_f32.resize(nelements);
        finp.read(reinterpret_cast<char *>(data_f32.data()),
                  nelements * sizeof(float));
      }

      ttype = qtype;
    } else {
      const int bpe = (ttype == 0) ? sizeof(float) : sizeof(uint16_t);

      data_u8.resize(nelements * bpe);
      finp.read(reinterpret_cast<char *>(data_u8.data()), nelements * bpe);
    }

    fout.write(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
    fout.write(reinterpret_cast<char *>(&length), sizeof(length));
    fout.write(reinterpret_cast<char *>(&ttype), sizeof(ttype));
    for (int i = 0; i < n_dims; ++i) {
      fout.write(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
    }
    fout.write(&name[0], length);

    if (quantize) {
      work.resize(nelements); // for quantization

      size_t cur_size = 0;
      std::vector<int64_t> hist_cur(1 << 4, 0);

      switch ((ggml_type)ttype) {
      case GGML_TYPE_Q4_0: {
        cur_size = ggml_quantize_q4_0(data_f32.data(), work.data(), nelements,
                                      ne[0], hist_cur.data());
      } break;
      case GGML_TYPE_Q4_1: {
        cur_size = ggml_quantize_q4_1(data_f32.data(), work.data(), nelements,
                                      ne[0], hist_cur.data());
      } break;
      case GGML_TYPE_Q5_0: {
        cur_size = ggml_quantize_q5_0(data_f32.data(), work.data(), nelements,
                                      ne[0], hist_cur.data());
      } break;
      case GGML_TYPE_Q5_1: {
        cur_size = ggml_quantize_q5_1(data_f32.data(), work.data(), nelements,
                                      ne[0], hist_cur.data());
      } break;
      case GGML_TYPE_Q8_0: {
        cur_size = ggml_quantize_q8_0(data_f32.data(), work.data(), nelements,
                                      ne[0], hist_cur.data());
      } break;
      case GGML_TYPE_Q2_K: {
        cur_size = ggml_quantize_q2_K(data_f32.data(), work.data(), nelements,
                                      ne[0], hist_cur.data());
      } break;
      case GGML_TYPE_Q3_K: {
        cur_size = ggml_quantize_q3_K(data_f32.data(), work.data(), nelements,
                                      ne[0], hist_cur.data());
      } break;
      case GGML_TYPE_Q4_K: {
        cur_size = ggml_quantize_q4_K(data_f32.data(), work.data(), nelements,
                                      ne[0], hist_cur.data());
      } break;
      case GGML_TYPE_Q6_K: {
        cur_size = ggml_quantize_q6_K(data_f32.data(), work.data(), nelements,
                                      ne[0], hist_cur.data());
      } break;
      case GGML_TYPE_Q5_K: {
        cur_size = ggml_quantize_q5_K(data_f32.data(), work.data(), nelements,
                                      ne[0], hist_cur.data());
      } break;
      case GGML_TYPE_Q8_K:
      case GGML_TYPE_F32:
      case GGML_TYPE_F16:
      case GGML_TYPE_I8:
      case GGML_TYPE_I16:
      case GGML_TYPE_I32:
      case GGML_TYPE_Q8_1:
      case GGML_TYPE_COUNT: {
        fprintf(stderr, "%s: unsupported quantization type %d (%s)\n", __func__,
                ttype, ggml_type_name((ggml_type)ttype));
        return false;
      }
      }

      fout.write(reinterpret_cast<char *>(work.data()), cur_size);
      total_size_new += cur_size;

      printf("size = %8.2f MB -> %8.2f MB | hist: ",
             nelements * sizeof(float) / 1024.0 / 1024.0,
             cur_size / 1024.0 / 1024.0);
      for (int i = 0; i < (int)hist_cur.size(); ++i) {
        hist_all[i] += hist_cur[i];
      }

      for (int i = 0; i < (int)hist_cur.size(); ++i) {
        printf("%5.3f ", hist_cur[i] / (float)nelements);
      }
      printf("\n");
    } else {
      printf("size = %8.3f MB\n", data_u8.size() / 1024.0 / 1024.0);
      fout.write(reinterpret_cast<char *>(data_u8.data()), data_u8.size());
      total_size_new += data_u8.size();
    }

    total_size_org += nelements * sizeof(float);
  }

  printf("%s: model size  = %8.2f MB\n", __func__,
         total_size_org / 1024.0 / 1024.0);
  printf("%s: quant size  = %8.2f MB | ftype = %d (%s)\n", __func__,
         total_size_new / 1024.0 / 1024.0, ftype, ggml_type_name(qtype));

  {
    int64_t sum_all = 0;
    for (int i = 0; i < (int)hist_all.size(); ++i) {
      sum_all += hist_all[i];
    }

    printf("%s: hist: ", __func__);
    for (int i = 0; i < (int)hist_all.size(); ++i) {
      printf("%5.3f ", hist_all[i] / (float)sum_all);
    }
    printf("\n");
  }

  return true;
}
// end ggml/examples/common-ggml.cpp

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

// quantize a model
bool mpt_model_quantize(const std::string &fname_inp,
                        const std::string &fname_out, ggml_ftype ftype) {
  printf("%s: loading model from '%s'\n", __func__, fname_inp.c_str());

  auto finp = std::ifstream(fname_inp, std::ios::binary);
  if (!finp) {
    fprintf(stderr, "%s: failed to open '%s' for reading\n", __func__,
            fname_inp.c_str());
    return false;
  }

  auto fout = std::ofstream(fname_out, std::ios::binary);
  if (!fout) {
    fprintf(stderr, "%s: failed to open '%s' for writing\n", __func__,
            fname_out.c_str());
    return false;
  }

  // verify magic
  {
    uint32_t magic;
    uint32_t version;
    finp.read((char *)&magic, sizeof(magic));
    if (magic != 0x67676d64) {
      fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__,
              fname_inp.c_str());
      return false;
    }
    finp.read((char *)&version, sizeof(version));
    if (version != 0) {
      fprintf(stderr, "%s: invalid model file '%s' (bad version)\n", __func__,
              fname_inp.c_str());
      return false;
    }

    fout.write((char *)&magic, sizeof(magic));
    fout.write((char *)&version, sizeof(version));
  }

  mpt_hparams hparams;

  // load hparams
  {
    finp.read((char *)&hparams.n_vocab, sizeof(hparams.n_vocab));
    finp.read((char *)&hparams.n_ctx, sizeof(hparams.n_ctx));
    finp.read((char *)&hparams.n_layer, sizeof(hparams.n_layer));
    finp.read((char *)&hparams.n_head, sizeof(hparams.n_head));
    finp.read((char *)&hparams.n_embd, sizeof(hparams.n_embd));
    finp.read((char *)&hparams.alibi_bias_max, sizeof(hparams.alibi_bias_max));
    finp.read((char *)&hparams.clip_qkv, sizeof(hparams.clip_qkv));
    finp.read((char *)&hparams.ftype, sizeof(hparams.ftype));

    printf("%s: n_vocab        = %d\n", __func__, hparams.n_vocab);
    printf("%s: n_ctx          = %d\n", __func__, hparams.n_ctx);
    printf("%s: n_embd         = %d\n", __func__, hparams.n_embd);
    printf("%s: n_head         = %d\n", __func__, hparams.n_head);
    printf("%s: n_layer        = %d\n", __func__, hparams.n_layer);
    printf("%s: alibi_bias_max = %f\n", __func__, hparams.alibi_bias_max);
    printf("%s: clip_qkv       = %f\n", __func__, hparams.clip_qkv);
    printf("%s: ftype          = %d\n", __func__, hparams.ftype);

    fout.write((char *)&hparams.n_vocab, sizeof(hparams.n_vocab));
    fout.write((char *)&hparams.n_ctx, sizeof(hparams.n_ctx));
    fout.write((char *)&hparams.n_layer, sizeof(hparams.n_layer));
    fout.write((char *)&hparams.n_head, sizeof(hparams.n_head));
    fout.write((char *)&hparams.n_embd, sizeof(hparams.n_embd));
    fout.write((char *)&hparams.alibi_bias_max, sizeof(hparams.alibi_bias_max));
    fout.write((char *)&hparams.clip_qkv, sizeof(hparams.clip_qkv));
    fout.write((char *)&ftype, sizeof(hparams.ftype));
  }

  // regexes of tensor names to be quantized
  const std::vector<std::string> to_quant = {
      ".*blocks.*weight",
  };

  if (!ggml_common_quantize_0(finp, fout, ftype, to_quant, {})) {
    fprintf(stderr, "%s: failed to quantize model '%s'\n", __func__,
            fname_inp.c_str());
    return false;
  }

  finp.close();
  fout.close();

  return true;
}

// usage:
//  ./gpt-2-quantize models/gpt-2-117M/ggml-model.bin
//  models/gpt-2-117M/ggml-model-quant.bin type
//
int main(int argc, char **argv) {
  if (argc != 4) {
    fprintf(stderr, "usage: %s model-f32.bin model-quant.bin type\n", argv[0]);
    ggml_print_ftypes(stderr);
    return 1;
  }

  // needed to initialize f16 tables
  {
    struct ggml_init_params params = {0, NULL, false};
    struct ggml_context *ctx = ggml_init(params);
    ggml_free(ctx);
  }

  const std::string fname_inp = argv[1];
  const std::string fname_out = argv[2];

  const ggml_ftype ftype = ggml_parse_ftype(argv[3]);

  const int64_t t_main_start_us = ggml_time_us();

  int64_t t_quantize_us = 0;

  // load the model
  {
    const int64_t t_start_us = ggml_time_us();

    if (!mpt_model_quantize(fname_inp, fname_out, ggml_ftype(ftype))) {
      fprintf(stderr, "%s: failed to quantize model from '%s'\n", __func__,
              fname_inp.c_str());
      return 1;
    }

    t_quantize_us = ggml_time_us() - t_start_us;
  }

  // report timing
  {
    const int64_t t_main_end_us = ggml_time_us();

    printf("\n");
    printf("%s: quantize time = %8.2f ms\n", __func__, t_quantize_us / 1000.0f);
    printf("%s:    total time = %8.2f ms\n", __func__,
           (t_main_end_us - t_main_start_us) / 1000.0f);
  }

  return 0;
}
