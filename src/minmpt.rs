use std::ptr::null_mut;
use thiserror::Error;

#[allow(dead_code)]
#[allow(non_camel_case_types)]
mod binding {
    include!(concat!(env!("OUT_DIR"), "/minmpt_bindings.rs"));
}

#[derive(Error, Debug)]
pub enum MinMPTError {
    #[error("Invalid input")]
    InvalidInput,
    #[error("Internal failure")]
    Failure,
    #[error("Unknown error")]
    Unknown,
}

impl MinMPTError {
    fn from_code(code: i32) -> Self {
        if code == binding::MINMPT_FAILURE as i32 {
            Self::Failure 
        } else if code == binding::MINMPT_INVALID as i32 {
            Self::InvalidInput
        } else {
            Self::Unknown
        }
    }
}

pub struct MinMPT {
    handle: binding::minmpt_handle,
}

#[allow(dead_code)]
impl MinMPT {
    pub fn load_model(path: &str) -> Result<MinMPT, MinMPTError> {
        let mut me = MinMPT { handle: null_mut() };
        let href: *mut binding::minmpt_handle = &mut me.handle;
        let err = unsafe {
            binding::minmpt_load(href, path.as_bytes().as_ptr().cast(), path.as_bytes().len())
        };
        if err == binding::MINMPT_OK as i32 {
            Ok(me)
        } else {
            Err(MinMPTError::from_code(err))
        }
    }

    pub fn n_vocab(&self) -> usize {
        unsafe {
            binding::minmpt_n_vocab(self.handle)
        }
    }
    pub fn n_past(&self) -> usize {
        unsafe {
            binding::minmpt_n_past(self.handle)
        }
    }
    pub fn rewind(&self, n: usize) {
        unsafe {
            binding::minmpt_rewind(self.handle, n)
        }
    }
    pub fn n_ctx(&self) -> usize {
        unsafe {
            binding::minmpt_n_ctx(self.handle)
        }
    }
    pub fn reset_ctx(&mut self) {
        unsafe {
            binding::minmpt_reset_ctx(self.handle);
        }
    }
    pub fn eval(&mut self, ids: &[u32], logits_out: &mut Vec<f32>) -> Result<(), MinMPTError> {
        logits_out.resize(self.n_vocab(), 0.0);
        let err = unsafe {
            binding::minmpt_eval_logits(self.handle, ids.as_ptr(), ids.len(), logits_out.as_mut_ptr())
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