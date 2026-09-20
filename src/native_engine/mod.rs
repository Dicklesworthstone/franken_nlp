//! CPU-reference model engine surface.

pub mod artifact_bridge;
pub mod attention;
pub mod batchsched;
pub mod constrained;
pub mod constrained_int8;
pub mod constrained_sparse;
pub mod decode;
pub mod diagnostic_f32;
pub mod dispatch;
pub mod generation;
pub mod hf_bf16_eager;
pub mod int8;
pub mod kv;
pub mod layer;
pub mod lmhead;
pub mod looprun;
pub mod nn;
pub mod portable_int8;
pub mod quant_algebra;
pub mod rope;
pub mod sampler;
pub mod strict_int8;
pub mod tensor;
pub mod weights;
