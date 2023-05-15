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
    #[structopt(short, long, possible_values = &ChatMode::variants(), case_insensitive = true)]
    chat_format: Option<ChatMode>, // TODO mirostat options
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
    let mut mptmodel = minmpt::MinMPT::load_model(&modelpathstr, Some(loadopts))?;
    let mut logits = Vec::new();
    let sysprompt = match mode {
        ChatMode::ChatML => "<|im_start|>system\nyou are a helpful assistant<|im_end|>", 
        ChatMode::Instruct => "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    };
    let chat_stop = (mode == ChatMode::ChatML).then(|| {
        tokenizer
            .token_to_id("<|im_end|>")
            .expect("get im_end token id")
    });
    let mut rng = rand::thread_rng();
    let mut sampler: Box<dyn Sampler<ThreadRng>> = if opt.mirostat {
        Box::new(sampling::Mirostat::new())
    } else {
        Box::new(sampling::BasicSampler {
            temperature: opt.temperature,
        })
    };

    let mut first_turn = true;
    while let Ok(line) =
        rl.readline(format!("{}/{}> ", mptmodel.n_past(), mptmodel.n_ctx()).as_str())
    {
        let wrapped = {
            let mut s = String::new();
            if first_turn {
                write!(&mut s, "{sysprompt}")?;
                first_turn = false;
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
            s
        };
        let encoding = tokenizer
            .encode(wrapped, true)
            .map_err(|_e| eyre!("Error tokenizing input"))?;
        mptmodel.eval(encoding.get_ids(), &mut logits)?;
        let mut resp_toks = Vec::new();
        let mut lastlen = 0;
        loop {
            //let tokid = sampling::greedy(logits.as_slice()) as u32;
            let tokid = sampler.sample(logits.as_slice(), &mut rng) as u32;
            if tokid == 0 {
                mptmodel.rewind(1); // don't keep endoftext in the context
                break;
            }
            if Some(tokid) == chat_stop {
                break;
            }
            resp_toks.push(tokid);
            let mut respinprogress = tokenizer.decode(resp_toks.clone(), false).unwrap();
            while respinprogress.ends_with('\u{FFFD}') {
                respinprogress.pop();
            }
            let (_prev, out) = respinprogress.split_at(lastlen);
            print!("{}", out);
            lastlen = respinprogress.as_bytes().len();
            std::io::stdout().flush()?;
            mptmodel.eval(&resp_toks[resp_toks.len() - 1..], &mut logits)?;
        }
        println!();
    }
    Ok(())
}
