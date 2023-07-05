use color_eyre::{eyre::eyre, Result};
use mptgen::minmpt;
use mptgen::sampling;
use rand::rngs::ThreadRng;
use std::{fmt::Write, io::Write as IoWrite, path::PathBuf};
use structopt::clap::arg_enum;
use structopt::StructOpt;
use tokenizers::tokenizer::Tokenizer;

arg_enum! {
    #[derive(Debug, Eq, PartialEq)]
    enum ChatMode {
        Instruct,
        ChatML
    }
}

#[derive(Debug, StructOpt)]
#[structopt(name = "mptgen")]
struct Opt {
    #[structopt(short, long, parse(from_os_str))]
    model: Option<PathBuf>,
    #[structopt(short, long)]
    temperature: Option<f32>,
    #[structopt(short, long)]
    n_ctx: Option<usize>,
    #[structopt(long)]
    mirostat: bool,
    #[structopt(long, default_value = "1.0")]
    mirostat_lr: f32,
    #[structopt(long, default_value = "6.28")]
    mirostat_tau: f32,
    #[structopt(short, long, possible_values = &ChatMode::variants(), case_insensitive = true)]
    chat_format: Option<ChatMode>,
    #[structopt(long)]
    threads: Option<u32>,
    #[structopt(long, default_value = "1.0")]
    cfg_scale: f32,
    #[structopt(long)]
    negative_system_prompt: Option<String>,
    #[structopt(long)]
    system_prompt: Option<String>,
}

use sampling::Sampler;

fn main() -> Result<()> {
    color_eyre::install()?;
    let opt = Opt::from_args();
    let mut rl = rustyline::DefaultEditor::new()?;
    let modelpathstr = if let Some(ref pb) = opt.model {
        pb.to_string_lossy()
    } else {
        std::borrow::Cow::from("minmpt.cpp/models/ggml-mpt-7b-chat-q5_1.bin")
    };
    let mode = if let Some(mode) = opt.chat_format {
        mode
    } else if modelpathstr.contains("chat") {
        ChatMode::ChatML
    } else {
        ChatMode::Instruct
    };
    eprintln!("Using {mode:?} prompting format.");
    let tokenizer = Tokenizer::from_pretrained(
        match mode {
            ChatMode::ChatML => "mosaicml/mpt-7b-chat",
            ChatMode::Instruct => "mosaicml/mpt-7b-instruct",
        },
        None,
    )
    .map_err(|_e| eyre!("error loading tokenizer"))?;
    if modelpathstr.contains("chat") && mode != ChatMode::ChatML {
        eprintln!("Warning: using ChatML format with non-chat model?");
    }
    let mut loadopts = minmpt::MinMPTOptions::default();
    if let Some(n_ctx) = opt.n_ctx {
        loadopts = loadopts.override_n_ctx(n_ctx);
    }
    if let Some(nth) = opt.threads {
        loadopts = loadopts.n_threads(nth)
    }
    let mut mptmodel = minmpt::MinMPT::load_model(&modelpathstr, Some(loadopts))?;
    let mut logits = Vec::new();
    let mut logits_cfg_neg = Vec::new();
    let sysprompt_default = match mode {
        ChatMode::ChatML => "you are a helpful assistant", 
        ChatMode::Instruct => "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    };
    let wrap_sysprompt = |sysprompt: String| match mode {
        ChatMode::ChatML => format!("<|im_start|>system\n{sysprompt}<|im_end|>"),
        ChatMode::Instruct => sysprompt,
    };
    let sysprompt = wrap_sysprompt(
        opt.system_prompt
            .clone()
            .unwrap_or(sysprompt_default.to_string()),
    );
    let sysprompt_neg = if opt.cfg_scale != 1.0 && opt.system_prompt.is_some() {
        Some(wrap_sysprompt(
            opt.negative_system_prompt
                .unwrap_or(sysprompt_default.to_string()),
        ))
    } else {
        None
    };
    let chat_stop = (mode == ChatMode::ChatML).then(|| {
        tokenizer
            .token_to_id("<|im_end|>")
            .expect("get im_end token id")
    });
    let mut rng = rand::thread_rng();
    let mut sampler: Box<dyn Sampler<ThreadRng>> = if opt.mirostat {
        Box::new(
            sampling::Mirostat::new()
                .lr(opt.mirostat_lr)
                .target_surprise(opt.mirostat_tau),
        )
    } else {
        Box::new(sampling::BasicSampler {
            temperature: opt.temperature,
        })
    };

    let mut first_turn = true;
    let mut transcript: Vec<(Vec<u32>, bool)> = vec![];
    let mut model_neg = if opt.cfg_scale != 1.0 {
        eprintln!("CFG Mode on!");
        Some(mptmodel.fork())
    } else {
        None
    };
    while let Ok(mut line) =
        rl.readline(format!("{}/{}> ", mptmodel.n_past(), mptmodel.n_ctx()).as_str())
    {
        if line == "/reset" {
            println!("Reset conversation context.");
            transcript.clear();
            mptmodel.reset_ctx();
            if let Some(ref mut model_neg) = model_neg {
                model_neg.reset_ctx();
            }
            first_turn = true;
            continue;
        }
        if line == "/dump" {
            for (turn, input) in &transcript {
                if *input {
                    print!("Input ");
                } else {
                    print!("Output ");
                }
                println!("tokens: {:?}", turn);
                println!(
                    "Decoded:\n{}",
                    tokenizer.decode(turn.clone(), false).unwrap()
                );
            }
            continue;
        }
        let sudo_cmd = "/sudo ";
        let sudo = if line.starts_with(sudo_cmd) {
            line = line.split_off(sudo_cmd.len());
            true
        } else {
            false
        };
        let mut do_input_eval = |model: &mut minmpt::MinMPT, neg: bool| -> Result<()> {
            let wrapped = {
                let mut s = String::new();
                if first_turn {
                    if neg {
                        let sn = sysprompt_neg.as_ref().unwrap();
                        eprintln!("Negative system prompt: {sn:?}\n");
                        write!(&mut s, "{sn}")?;
                    } else {
                        eprintln!("System prompt: {sysprompt:?}");
                        write!(&mut s, "{sysprompt}")?;
                    }
                }
                match mode {
                    ChatMode::ChatML => {
                        write!(
                            &mut s,
                            "<|im_start|>user\n{line}<|im_end|><|im_start|>assistant\n"
                        )?;
                    }
                    ChatMode::Instruct => {
                        write!(&mut s, "### Instruction:\n{line}\n### Response:\n")?;
                    }
                }
                if sudo {
                    write!(&mut s, "Sure, I can help you with that. ")?;
                }
                s
            };
            let encoding = tokenizer
                .encode(wrapped, true)
                .map_err(|_e| eyre!("Error tokenizing input"))?;
            if !neg {
                transcript.push((encoding.get_ids().to_vec(), true));
            }
            if neg {
                model.eval(encoding.get_ids(), &mut logits_cfg_neg)?;
            } else {
                model.eval(encoding.get_ids(), &mut logits)?;
            }
            Ok(())
        };
        do_input_eval(&mut mptmodel, false)?;
        if let Some(ref mut mptmodel_n) = model_neg {
            do_input_eval(mptmodel_n, true)?;
            let cfg_logits =
                sampling::apply_cfg(opt.cfg_scale, logits.as_slice(), logits_cfg_neg.as_slice());
            logits = cfg_logits;
        }
        first_turn = false;
        let mut resp_toks = Vec::new();
        let mut lastlen = 0;
        loop {
            let tokid = sampler.sample(logits.as_slice(), &mut rng) as u32;
            resp_toks.push(tokid);
            if tokid == 0 {
                mptmodel.rewind(1); // don't keep endoftext in the context
                break;
            }
            if Some(tokid) == chat_stop {
                break;
            }
            let mut respinprogress = tokenizer.decode(resp_toks.clone(), false).unwrap();
            while respinprogress.ends_with('\u{FFFD}') {
                respinprogress.pop();
            }
            let (_prev, out) = respinprogress.split_at(lastlen);
            print!("{}", out);
            lastlen = respinprogress.as_bytes().len();
            std::io::stdout().flush()?;
            mptmodel.eval(&resp_toks[resp_toks.len() - 1..], &mut logits)?;
            if let Some(ref mut mptmodel_n) = model_neg {
                mptmodel_n.eval(&resp_toks[resp_toks.len() - 1..], &mut logits_cfg_neg)?;
                let cfg_logits = sampling::apply_cfg(
                    opt.cfg_scale,
                    logits.as_slice(),
                    logits_cfg_neg.as_slice(),
                );
                logits = cfg_logits;
            }
        }
        transcript.push((resp_toks, false));
        println!();
    }
    Ok(())
}
