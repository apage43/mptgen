use color_eyre::eyre::bail;
use color_eyre::{eyre::eyre, Result};
use mptgen::minmpt;
use mptgen::sampling;
use mptgen::sampling::Sampler;
use rand::rngs::ThreadRng;
use std::io::Read;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::{io::Write as IoWrite, path::PathBuf};
use structopt::StructOpt;
use tokenizers::tokenizer::Tokenizer;

#[derive(Debug, StructOpt)]
#[structopt(name = "mptgen")]
struct Opt {
    #[structopt(parse(from_os_str))]
    story: PathBuf,
    #[structopt(short, long, parse(from_os_str))]
    model: Option<PathBuf>,
    #[structopt(short="b", long, default_value="32")]
    input_batch_size: usize,
    #[structopt(short, long = "temp")]
    temperature: Option<f32>,
    #[structopt(long, short = "c")]
    n_ctx: Option<usize>,
    #[structopt(long)]
    mirostat: bool,
    #[structopt(long, default_value = "1.0")]
    mirostat_lr: f32,
    #[structopt(long, default_value = "6.28")]
    mirostat_tau: f32,
    #[structopt(
        long,
        short,
        default_value = "256",
        help = "amount of tokens to generate at once"
    )]
    n_gen: usize,
    #[structopt(long)]
    threads: Option<u32>,
}

fn main() -> Result<()> {
    color_eyre::install()?;
    let opt = Opt::from_args();
    let storyfile = opt.story;
    if !storyfile.exists() {
        bail!("Story file {} must exist", storyfile.display());
    }
    let modelpathstr = if let Some(ref pb) = opt.model {
        pb.to_string_lossy()
    } else {
        std::borrow::Cow::from("minmpt.cpp/models/ggml-mpt-7b-storywriter-q5_1.bin")
    };
    let tokenizer = Tokenizer::from_pretrained("mosaicml/mpt-7b-storywriter", None)
        .map_err(|e| eyre!("Error loading tokenizer: {e:?}"))?;
    let mut rl = rustyline::DefaultEditor::new()?;
    let mut loadopts = minmpt::MinMPTOptions::default();
    if let Some(n_ctx) = opt.n_ctx {
        loadopts = loadopts.override_n_ctx(n_ctx);
    }
    if let Some(nth) = opt.threads {
        loadopts = loadopts.n_threads(nth)
    }
    let mut mptmodel = minmpt::MinMPT::load_model(&modelpathstr, Some(loadopts))?;
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

    let read_story = || {
        let mut storytext = String::new();
        std::fs::File::open(&storyfile)?.read_to_string(&mut storytext)?;
        Ok::<_, color_eyre::eyre::Error>(if storytext.is_empty() {
            eprintln!("Warning: story text is empty!");
            vec![tokenizer
                .token_to_id("<|endoftext|>")
                .expect("endoftext token lookup")]
        } else {
            let stoks = tokenizer
                .encode(storytext.clone(), false)
                .map_err(|e| eyre!("Error tokenizing: {e:?}"))?;
            print!(
                "Input story text ({} tokens):\n{storytext} (end input)",
                stoks.get_ids().len()
            );
            Vec::from(stoks.get_ids())
        })
    };
    let mut storytokens = read_story()?;
    let mut stop = storytokens.len() + opt.n_gen;
    let mut resplen = 0;
    let stop_signal = Arc::new(AtomicBool::new(false));
    let r = stop_signal.clone();
    ctrlc::set_handler(move || {
        r.store(true, Ordering::SeqCst);
    })?;
    loop {
        let n_past = mptmodel.n_past();
        //eprintln!("n_past={}, storytokens.len={}", n_past, storytokens.len());
        stop_signal.store(false, Ordering::SeqCst);
        let mut resp_toks = vec![];
        if n_past < stop {
            let mut logits = Vec::new();
            for chunk in storytokens[n_past..].chunks(opt.input_batch_size) {
                mptmodel.eval(chunk, &mut logits)?;
            }
            let mut lastlen = 0;
            loop {
                //let tokid = sampling::greedy(logits.as_slice()) as u32;
                let tokid = sampler.sample(logits.as_slice(), &mut rng) as u32;
                if tokid == 0 {
                    mptmodel.rewind(1); // don't keep endoftext in the context
                    break;
                }
                resp_toks.push(tokid);
                if stop_signal.load(Ordering::SeqCst) {
                    eprintln!("(interrupted)");
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
                if n_past + resp_toks.len() >= stop {
                    break;
                }
                mptmodel.eval(&resp_toks[resp_toks.len() - 1..], &mut logits)?;
            }
            resplen = resp_toks.len();
        }
        storytokens.append(&mut resp_toks);
        println!();
        //eprintln!("n_past={}, storytokens.len={}, resplen={}", mptmodel.n_past(), storytokens.len(), resplen);
        let inp = rl
            .readline(&format!(
                "[M]ore/(w)rite/(r)eread/re(j)ect/(d)ump/(q)uit ({}/{})> ",
                storytokens.len(),
                mptmodel.n_ctx()
            ))?
            .to_lowercase();
        match inp.as_str() {
            "m" | "" => {
                stop = storytokens.len() + opt.n_gen;
            }
            "w" => {
                let storytext = tokenizer
                    .decode(storytokens.clone(), true)
                    .map_err(|e| eyre!("story detokenize {e:?}"))?;
                std::fs::File::options()
                    .write(true)
                    .open(&storyfile)?
                    .write_all(storytext.as_bytes())?;
                eprintln!("Saved to {storyfile:?}");
            }
            "d" => {
                eprintln!("Story tokens: {storytokens:?}");
                eprintln!("Story tokens e2e decode:");
                let storytext = tokenizer
                    .decode(storytokens.clone(), true)
                    .map_err(|e| eyre!("story detokenize {e:?}"))?;
                println!("{storytext}");
            }
            "r" => {
                let readtokens = read_story()?;
                let common_pfx_len = readtokens.iter().zip(storytokens.iter()).take_while(|(rt, st)| {
                    rt == st
                }).count();
                // if there was a common prefix, we still need to back up one
                // extra token in order to recompute the token that follows it
                // since we don't cache the actual final logits
                let common_pfx_len = common_pfx_len.saturating_sub(1);
                storytokens = readtokens;
                let rwlen = mptmodel.n_past() - common_pfx_len;
                eprintln!("Rewinding by {rwlen}");
                mptmodel.rewind(rwlen);
            }
            "j" => {
                storytokens.resize(storytokens.len() - resplen, 0);
                mptmodel.rewind(resplen);
            }
            "q" => {
                break;
            }
            _ => {
                continue;
            }
        };
    }
    Ok(())
}
