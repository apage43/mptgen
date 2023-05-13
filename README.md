# mptgen

example of binding a **model-only** [ggml](https://github.com/ggerganov/ggml) implementation of [MPT-7B](https://www.mosaicml.com/blog/mpt-7b) to Rust, handling tokenizing and sampling in Rust instead of C(++)

expects to use the `-chat` [model](https://huggingface.co/mosaicml/mpt-7b-chat) with [ChatML](https://github.com/openai/openai-python/blob/main/chatml.md) format

does not properly handle attempting to go past max_seq_len (2048) yet - this requires either rotating old values out of the kv cache to discard them or expanding it (which should actually work somewhat thanks to MPT-7B's use of ALiBi)

```bash
# build minmpt
(mkdir -p minmpt.cpp/build && cd minmpt.cpp/build && cmake -G Ninja .. && ninja)
# convert a model with notebook
# quantize it
cd minmpt.cpp && mkdir -p models && ./build/bin/quantize /mnt/e/big_model/ggml-mpt-7b-chat-f32.bin models/ggml-mpt-7b-chat-q5_1.bin 
```

```bash
LD_LIBRARY_PATH="./minmpt.cpp/build" cargo run -- --temperature 0.7
    Finished dev [unoptimized + debuginfo] target(s) in 0.13s
     Running `target/debug/mptgen --temperature 0.7`
mpt_model_load: loading model from 'minmpt.cpp/models/ggml-mpt-7b-chat-q5_1.bin' - please wait ...
mpt_model_load: n_vocab        = 50432
mpt_model_load: n_ctx          = 2048
mpt_model_load: n_embd         = 4096
mpt_model_load: n_head         = 32
mpt_model_load: n_layer        = 32
mpt_model_load: alibi_bias_max = 8.000000
mpt_model_load: clip_qkv       = 0.000000
mpt_model_load: ftype          = 9
mpt_model_load: ggml ctx size = 6421.09 MB
mpt_model_load: memory_size =  1024.00 MB, n_mem = 65536
mpt_model_load: ........................ done
mpt_model_load: model size =  5397.02 MB / num tensors = 194
0/2048> list the 10 best emojis and explain your reasoning
1. 🤩 (the exploding head emoji) - because it represents a sudden and unexpected burst of excitement or happiness.
2. 🐶 (the bearded collie) - because it is a cute and playful emoji that represents loyalty and friendship.
3. 💩 (the poop emoji) - because it is a universally recognizable and relatable symbol of the human experience.
4. 🚀 (the rocket emoji) - because it represents a sense of upward mobility and progress.
5. 🏰 (the castle emoji) - because it represents a sense of grandeur and luxury.
6. 🍔 (the hamburger emoji) - because it represents a classic and beloved food item that is widely recognized and loved.
7. 🌎 (the globe emoji) - because it represents the interconnectedness of the world and the importance of global awareness.
8. 👩‍🦳 (the woman with headband emoji) - because it represents strength, determination, and perseverance.
9. 🤖 (the robot emoji) - because it represents innovation, technology, and the future.
10. 🐒 (the baby chick emoji) - because it represents new life, growth, and innocence.
313/2048>
```
