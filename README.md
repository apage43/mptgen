# mptgen

example of binding a **model-only** [ggml](https://github.com/ggerganov/ggml) implementation of [MPT-7B](https://www.mosaicml.com/blog/mpt-7b) to Rust, handling tokenizing and sampling in Rust instead of C(++)

`chat` binary supports "chat" and "instruct" models with the appropriate [ChatML](https://github.com/openai/openai-python/blob/main/chatml.md) or Alpaca-style prompting formats, autodetected from model file name.

`writer` binary supports the StoryWriter or base models.

max_seq_len is overridable with the `--n-ctx` flag, it does not attempt to handle continuing generation past max_seq_len yet

```bash
# build minmpt
(mkdir -p minmpt.cpp/build && cd minmpt.cpp/build && cmake -G Ninja .. && ninja)
# convert a model with notebook
# quantize it
cd minmpt.cpp && mkdir -p models && ./build/bin/quantize /mnt/e/big_model/ggml-mpt-7b-chat-f32.bin models/ggml-mpt-7b-chat-q5_1.bin 
```

```bash
LD_LIBRARY_PATH="./minmpt.cpp/build" cargo run --bin chat -- --temperature 0.7
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


```bash
❯ echo -n "Once upon a time, there was a frog named" > story.txt && LD_LIBRARY_PATH="./minmpt.cpp/build" cargo run --bin writer -- story.txt -t 0.8 -n 32
    Finished dev [unoptimized + debuginfo] target(s) in 0.13s
     Running `target/debug/writer story.txt -t 0.8 -n 32`
mpt_model_load: loading model from 'minmpt.cpp/models/ggml-mpt-7b-storywriter-q5_1.bin' - please wait ...
mpt_model_load: n_vocab        = 50432
mpt_model_load: n_ctx          = 65536
mpt_model_load: n_embd         = 4096
mpt_model_load: n_head         = 32
mpt_model_load: n_layer        = 32
mpt_model_load: alibi_bias_max = 16.000000
mpt_model_load: clip_qkv       = 6.000000
mpt_model_load: ftype          = 9
mpt_model_load: ggml ctx size = 38165.09 MB
mpt_model_load: memory_size = 32768.00 MB, n_mem = 2097152
mpt_model_load: ........................ done
mpt_model_load: model size =  5397.02 MB / num tensors = 194
Input story text (10 tokens):
Once upon a time, there was a frog named (end input) Bruce Wayne, who hung out with some other frogs and other amphibian friends—the Bat-Frogs, as they were called. But this was not to be his life forever. No, it
[M]ore/(w)rite/(r)eread/re(j)ect/(q)uit (52/65536)> 
 was destined that Bruce Wayne would grow up and learn how to fly, and then he would go off and join the Bat-Frogs, and then he would
[M]ore/(w)rite/(r)eread/re(j)ect/(q)uit (85/65536)> j
 was a temporary stage in his life. For a few years, the Bat-Frogs occupied Wayne Manor and were all Bruce Wayne wanted to do. Then one
[M]ore/(w)rite/(r)eread/re(j)ect/(q)uit (85/65536)> 
 day, a couple of years later, he decided to move on to other things.

Of course, then-Batman, Adam Strange, and most some
[M]ore/(w)rite/(r)eread/re(j)ect/(q)uit (118/65536)> w
Saved to "story.txt"
```