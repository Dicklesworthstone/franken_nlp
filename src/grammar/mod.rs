//! Bounded constrained-decoding grammar surface.

pub mod compiler;
pub mod execution;
pub mod mask;
pub mod schema;
pub mod source;

pub use compiler::{
    AutomatonEstimate, CompileLimits, CompiledSchema, MASK_BYTES_PER_STATE, SchemaNode,
    SourceAnnotation, TypedJsonAutomaton, compile_json_schema,
};
pub use schema::{ExactDecimal, ScalarValue, SchemaError};
