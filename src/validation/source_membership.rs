//! Source-membership validation independent of the grammar product surface.

use std::fmt;

use super::offsets::{OffsetError, SourceSpan, validate_source_span};

/// A typed failure from source-membership verification.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GroundingError {
    Offset(OffsetError),
}

impl fmt::Display for GroundingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Offset(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for GroundingError {}

impl From<OffsetError> for GroundingError {
    fn from(value: OffsetError) -> Self {
        Self::Offset(value)
    }
}

/// Require the output text to be exactly the checked half-open source slice.
///
/// A substring search is intentionally insufficient: output evidence carries
/// the location it claims, and both byte and scalar coordinates must agree.
pub fn verify_source_membership(
    source: &str,
    output_text: &str,
    span: SourceSpan,
) -> Result<(), GroundingError> {
    validate_source_span(source, output_text, span).map_err(Into::into)
}
