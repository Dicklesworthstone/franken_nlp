//! Complete-document sentiment adapters for the bounded corpus runner.
pub mod grouped_int4;
/// Independently sampled score/justification bundles on one resident model.
pub mod sampled;
/// Complete original documents on one resident INT8 invocation.
pub mod quantized;
