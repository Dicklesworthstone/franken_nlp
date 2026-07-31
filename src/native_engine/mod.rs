//! CPU-reference model engine surface.

pub mod attention;
pub mod batchsched;
pub mod decode;
pub mod diagnostic_f32;
pub mod dispatch;
pub mod hf_bf16_eager;
pub mod int8;
pub mod kv;
pub mod layer;
pub mod lmhead;
pub mod looprun;
pub mod nn;
pub mod quant_algebra;
pub mod rope;
pub mod sampler;
pub mod tensor;
pub mod weights;
