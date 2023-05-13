# minmpt.cpp

*minimalist* (no tokenizer, no sampler, no binary)
[ggml](http://github.com/ggerganov/ggml) implementation of MPT-7B with with C interface.
some parts taken from [llama.cpp](http://github.com/ggerganov/llama.cpp) and the ggml examples

**model files converted for this implementation should not be assumed to compatible with any other code implementing the same models**

note: this version of ggml has a modified implementation of `ggml_alibi` to match the implementation in the MPT models