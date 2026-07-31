//! Independent byte and Unicode-scalar offset verification.

use std::fmt;

/// A half-open source range expressed in both byte and Unicode-scalar units.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SourceSpan {
    pub byte_start: usize,
    pub byte_end: usize,
    pub scalar_start: usize,
    pub scalar_end: usize,
}

impl SourceSpan {
    pub const fn new(
        byte_start: usize,
        byte_end: usize,
        scalar_start: usize,
        scalar_end: usize,
    ) -> Self {
        Self {
            byte_start,
            byte_end,
            scalar_start,
            scalar_end,
        }
    }
}

/// Failure from an independent source-span check.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OffsetError {
    ReversedByteRange {
        byte_start: usize,
        byte_end: usize,
    },
    OutOfBounds {
        byte_offset: usize,
        source_len: usize,
    },
    NonBoundary {
        byte_offset: usize,
    },
    ScalarCoordinateMismatch {
        byte_offset: usize,
        declared_scalar: usize,
        computed_scalar: usize,
    },
    TextMismatch {
        byte_start: usize,
        byte_end: usize,
    },
}

impl fmt::Display for OffsetError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ReversedByteRange {
                byte_start,
                byte_end,
            } => write!(
                formatter,
                "half-open byte range [{byte_start}, {byte_end}) is reversed"
            ),
            Self::OutOfBounds {
                byte_offset,
                source_len,
            } => write!(
                formatter,
                "byte offset {byte_offset} is outside source length {source_len}"
            ),
            Self::NonBoundary { byte_offset } => {
                write!(
                    formatter,
                    "byte offset {byte_offset} is not a UTF-8 boundary"
                )
            }
            Self::ScalarCoordinateMismatch {
                byte_offset,
                declared_scalar,
                computed_scalar,
            } => write!(
                formatter,
                "byte offset {byte_offset} has scalar coordinate {computed_scalar}, not declared {declared_scalar}"
            ),
            Self::TextMismatch {
                byte_start,
                byte_end,
            } => write!(
                formatter,
                "source bytes [{byte_start}, {byte_end}) do not equal the grounded text"
            ),
        }
    }
}

impl std::error::Error for OffsetError {}

/// Return the Unicode-scalar coordinate of a checked UTF-8 byte boundary.
pub fn scalar_index_for_byte(source: &str, byte_offset: usize) -> Result<usize, OffsetError> {
    if byte_offset > source.len() {
        return Err(OffsetError::OutOfBounds {
            byte_offset,
            source_len: source.len(),
        });
    }
    if !source.is_char_boundary(byte_offset) {
        return Err(OffsetError::NonBoundary { byte_offset });
    }
    Ok(source[..byte_offset].chars().count())
}

/// Return the byte coordinate of a checked Unicode-scalar boundary.
pub fn byte_index_for_scalar(source: &str, scalar_index: usize) -> Option<usize> {
    if scalar_index == source.chars().count() {
        return Some(source.len());
    }
    source
        .char_indices()
        .nth(scalar_index)
        .map(|(index, _)| index)
}

/// Verify a grounded value's exact source slice and both coordinate systems.
pub fn validate_source_span(
    source: &str,
    expected_text: &str,
    span: SourceSpan,
) -> Result<(), OffsetError> {
    if span.byte_start > span.byte_end {
        return Err(OffsetError::ReversedByteRange {
            byte_start: span.byte_start,
            byte_end: span.byte_end,
        });
    }
    let computed_start = scalar_index_for_byte(source, span.byte_start)?;
    let computed_end = scalar_index_for_byte(source, span.byte_end)?;
    if computed_start != span.scalar_start {
        return Err(OffsetError::ScalarCoordinateMismatch {
            byte_offset: span.byte_start,
            declared_scalar: span.scalar_start,
            computed_scalar: computed_start,
        });
    }
    if computed_end != span.scalar_end {
        return Err(OffsetError::ScalarCoordinateMismatch {
            byte_offset: span.byte_end,
            declared_scalar: span.scalar_end,
            computed_scalar: computed_end,
        });
    }
    if source[span.byte_start..span.byte_end] != *expected_text {
        return Err(OffsetError::TextMismatch {
            byte_start: span.byte_start,
            byte_end: span.byte_end,
        });
    }
    Ok(())
}
