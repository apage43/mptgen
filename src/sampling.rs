use rand::{prelude::Distribution, Rng};
use std::f32::consts::TAU;

pub fn apply_cfg(cfg_scale: f32, pos_logits: &[f32], neg_logits: &[f32]) -> Vec<f32> {
    pos_logits
        .iter()
        .zip(neg_logits.iter())
        .map(|(p, n)| cfg_scale * (p - n) + n)
        .collect()
}

pub trait Sampler<R: Rng> {
    fn sample(&mut self, raw_logits: &[f32], rng: &mut R) -> usize;
}

struct Candidate {
    logit: f32,
    p: f32,
    token: usize,
}

fn sorted_softmax(raw_logits: &[f32]) -> Vec<Candidate> {
    let ps = softmax(raw_logits);
    let mut samples: Vec<Candidate> = raw_logits
        .iter()
        .enumerate()
        .map(|(i, l)| Candidate {
            logit: *l,
            p: ps[i],
            token: i,
        })
        .collect();
    samples.sort_by(|sa, sb| sa.logit.partial_cmp(&sb.logit).unwrap());
    samples
}

fn softmax_inplace(logits: &mut [f32]) {
    let lmax = *logits
        .iter()
        .max_by(|la, lb| la.partial_cmp(lb).unwrap())
        .unwrap();
    // shift max to 0 to avoid overflow
    logits.iter_mut().for_each(|l| *l = (*l - lmax).exp());
    let psum: f32 = logits.iter().sum();
    logits.iter_mut().for_each(|p| *p /= psum);
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let mut out = Vec::from(logits);
    softmax_inplace(&mut out);
    out
}

#[derive(Debug, Default)]
pub struct BasicSampler {
    pub temperature: Option<f32>,
}
impl<R: Rng> Sampler<R> for BasicSampler {
    fn sample(&mut self, raw_logits: &[f32], rng: &mut R) -> usize {
        if self.temperature == Some(0.) {
            return greedy(raw_logits);
        }
        let probs = if let Some(temp) = self.temperature {
            let mut templogits: Vec<f32> = raw_logits.iter().map(|l| *l / temp).collect();
            softmax_inplace(&mut templogits);
            templogits
        } else {
            softmax(raw_logits)
        };
        let dist = rand::distributions::WeightedIndex::new(probs)
            .expect("create probability distribution");
        dist.sample(rng)
    }
}

fn estimate_surprise(orig: &[Candidate], m: usize) -> f32 {
    let end = (m - 1).min(orig.len() - 1);
    let mut num = 0.;
    let mut den = 0.;
    for i in 0..end {
        let fi = i as f32;
        let ti = ((fi + 2.) / (fi + 1.)).log10();
        let bi = (orig[i].p / orig[i + 1].p).log10();
        num += ti * bi;
        den += ti * ti;
    }
    num / den
}

#[derive(Debug)]
pub struct Mirostat {
    lr: f32,
    max_surprise: f32,
    target_surprise: f32,
    m: usize, // # of tokens for surprise estimate
}

impl Mirostat {
    pub fn new() -> Self {
        Mirostat {
            lr: 1.0,
            target_surprise: TAU,
            max_surprise: 2.0 * TAU,
            m: 100,
        }
    }
    pub fn lr(self, lr: f32) -> Self {
        Self { lr, ..self }
    }
    pub fn target_surprise(self, target_surprise: f32) -> Self {
        Self {
            target_surprise,
            max_surprise: target_surprise * 2.0,
            ..self
        }
    }
}

impl Default for Mirostat {
    fn default() -> Self {
        Self::new()
    }
}

impl<R: Rng> Sampler<R> for Mirostat {
    fn sample(&mut self, raw_logits: &[f32], rng: &mut R) -> usize {
        let mut cands = sorted_softmax(raw_logits);
        let n_cand = cands.len() as f32;
        let surp = estimate_surprise(&cands, self.m);
        let eps = surp - 1.;
        let k =
            ((eps * (2.0_f32.powf(self.max_surprise))) / (1. - n_cand.powf(-eps))).powf(1.0 / surp);
        let k = (k.round() + 1.) as usize;
        cands.shrink_to(k);
        let mut p_logits: Vec<f32> = cands.iter().map(|c| c.logit).collect();
        let probs = {
            softmax_inplace(&mut p_logits);
            p_logits
        };
        let dist = rand::distributions::WeightedIndex::new(probs)
            .expect("create probability distribution");
        let pidx = dist.sample(rng);
        let isurp = cands[pidx].p.recip().log2();
        let error_surprise = isurp - self.target_surprise;
        self.max_surprise -= self.lr * error_surprise;
        cands[pidx].token
    }
}

pub fn greedy(logits: &[f32]) -> usize {
    assert!(!logits.is_empty());
    let (i, _v) = logits
        .iter()
        .enumerate()
        .max_by(|(_, va), (_, vb)| va.partial_cmp(vb).unwrap())
        .unwrap();
    i
}
