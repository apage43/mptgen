use color_eyre::{eyre::eyre, Result};
use rand::rngs::ThreadRng;
use std::{fmt::Write, io::Write as IoWrite, path::PathBuf};
use structopt::StructOpt;
use tokenizers::tokenizer::Tokenizer;
mod minmpt;
mod sampling;

#[derive(Debug, StructOpt)]
#[structopt(name = "mptgen")]
struct Opt {
    #[structopt(parse(from_os_str))]
    model: Option<PathBuf>,
    #[structopt(short, long)]
    temperature: Option<f32>,
    #[structopt(long)]
    mirostat: bool,
    // TODO mirostat options
}

use sampling::Sampler;

fn main() -> Result<()> {
    color_eyre::install()?;
    let opt = Opt::from_args();
    let mut rl = rustyline::DefaultEditor::new()?;
    let tokenizer = Tokenizer::from_pretrained("mosaicml/mpt-7b-chat", None)
        .map_err(|_e| eyre!("error loading tokenizer"))?;
    let mut mptmodel = if let Some(pb) = opt.model {
        minmpt::MinMPT::load_model(&pb.to_string_lossy())?
    } else {
        minmpt::MinMPT::load_model("minmpt.cpp/models/ggml-mpt-7b-chat-q5_1.bin")?
    };
    let mut logits = Vec::new();
    let sysprompt = "<|im_start|>system\nyou are a helpful assistant<|im_end|>";
    let chat_endtok = tokenizer
        .token_to_id("<|im_end|>")
        .expect("get im_end token id");
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
            write!(
                &mut s,
                "<|im_start|>user\n{line}<|im_end|><|im_start|>assistant\n"
            )?;
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
            if tokid == chat_endtok {
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
