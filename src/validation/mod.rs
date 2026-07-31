//! Independent output-validation surface.
//!
//! This module deliberately owns its parser, bounded decimal implementation,
//! schema walker, and source-span verifier.  It may consume a compiled
//! `SchemaNode` as an immutable description, but it must not
//! reuse grammar automata, masks, transitions, or acceptance code.

pub mod json;
pub mod offsets;
pub mod schema;
pub mod source_membership;

pub use json::{
    Decimal, DecimalError, IntegerValue, JsonLimits, JsonParseError, JsonParseErrorKind, JsonValue,
    parse_json, parse_json_with_limits,
};
pub use offsets::{
    OffsetError, SourceSpan, byte_index_for_scalar, scalar_index_for_byte, validate_source_span,
};
pub use schema::{
    GroundedValue, ValidationError, ValidationErrorKind, validate_json, validate_value,
    validate_with_grounding,
};
pub use source_membership::{GroundingError, verify_source_membership};
