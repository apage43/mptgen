use std::ptr::null_mut;
use thiserror::Error;

#[allow(dead_code)]
#[allow(non_camel_case_types)]
mod binding {
    include!(concat!(env!("OUT_DIR"), "/minmpt_bindings.rs"));
}

#[derive(Error, Debug)]
pub enum MinMPTError {
    #[error("Exceeded model context size")]
    ContextLimit,
    #[error("Invalid input")]
    InvalidInput,
    #[error("Internal failure")]
    Failure,
    #[error("Unknown error")]
    Unknown,
}

impl MinMPTError {
    fn from_code(code: i32) -> Self {
        match code as u32 {
            binding::MINMPT_FAILURE => Self::Failure,
            binding::MINMPT_INVALID => Self::InvalidInput,
            binding::MINMPT_CTX_LIMIT => Self::ContextLimit,
            _ => Self::Unknown,
        }
    }
}

#[derive(Default, Debug)]
pub struct MinMPTOptions {
    n_ctx_override: Option<usize>,
    n_threads: Option<u32>,
}

impl MinMPTOptions {
    pub fn override_n_ctx(self, n_ctx: usize) -> Self {
        Self {
            n_ctx_override: Some(n_ctx),
            ..self
        }
    }
    pub fn n_threads(self, n_threads: u32) -> Self {
        Self {
            n_threads: Some(n_threads),
            ..self
        }
    }
}

pub struct MinMPT {
    handle: binding::minmpt_handle,
}

#[allow(dead_code)]
impl MinMPT {
    pub fn load_model(
        path: &str,
        load_options: Option<MinMPTOptions>,
    ) -> Result<MinMPT, MinMPTError> {
        let mut me = MinMPT { handle: null_mut() };
        let load_options = load_options.unwrap_or_default();
        let href: *mut binding::minmpt_handle = &mut me.handle;
        let err = unsafe {
            binding::minmpt_load(
                href,
                path.as_bytes().as_ptr().cast(),
                path.as_bytes().len(),
                load_options.n_ctx_override.unwrap_or(0),
            )
        };
        if err == binding::MINMPT_OK as i32 {
            if let Some(nth) = load_options.n_threads {
                me.set_n_threads(nth);
            }
            Ok(me)
        } else {
            Err(MinMPTError::from_code(err))
        }
    }
    /// Creates a second model instance with its own kvcache but sharing the original model's
    /// weight data
    pub fn fork(&self) -> MinMPT {
        let mut child = MinMPT { handle: null_mut() };
        let selfref: binding::minmpt_handle = self.handle;
        let childref: *mut binding::minmpt_handle = &mut child.handle;
        unsafe{
            binding::minmpt_fork(selfref, childref);
        }
        child
    }
    pub fn set_n_threads(&self, n_threads: u32) {
        unsafe { binding::minmpt_set_n_threads(self.handle, n_threads) }
    }
    pub fn n_vocab(&self) -> usize {
        unsafe { binding::minmpt_n_vocab(self.handle) }
    }
    pub fn n_past(&self) -> usize {
        unsafe { binding::minmpt_n_past(self.handle) }
    }
    pub fn rewind(&self, n: usize) {
        unsafe { binding::minmpt_rewind(self.handle, n) }
    }
    pub fn n_ctx(&self) -> usize {
        unsafe { binding::minmpt_n_ctx(self.handle) }
    }
    pub fn reset_ctx(&mut self) {
        unsafe {
            binding::minmpt_reset_ctx(self.handle);
        }
    }
    pub fn eval(&mut self, ids: &[u32], logits_out: &mut Vec<f32>) -> Result<(), MinMPTError> {
        if ids.is_empty() {
            return Err(MinMPTError::InvalidInput)
        }
        logits_out.resize(self.n_vocab(), 0.0);
        let err = unsafe {
            binding::minmpt_eval_logits(
                self.handle,
                ids.as_ptr(),
                ids.len(),
                logits_out.as_mut_ptr(),
            )
        };
        if err == binding::MINMPT_OK as i32 {
            Ok(())
        } else {
            Err(MinMPTError::from_code(err))
        }
    }
}

impl Drop for MinMPT {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                binding::minmpt_free(self.handle);
            }
        }
    }
}
