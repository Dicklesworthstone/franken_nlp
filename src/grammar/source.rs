//! Request-owned source-language products for grounded JSON strings.
//!
//! A `verbatim` string is not post-hoc substring checked.  Its logical,
//! unescaped UTF-8 bytes advance this machine while the JSON lexer and the
//! tokenizer detokenization transducer advance their own states.  The source
//! machine starts at every UTF-8 boundary, so every accepting run identifies a
//! source interval and cannot spell bytes that are absent from the source.
//!
//! The typed JSON compiler intentionally keeps its transition graph private.
//! This module therefore composes its *checked cardinalities* with the public
//! compiler plan, while exposing the source-side byte transitions that the
//! execution compiler will drive together with JSON and tokenizer states.
//! Source-derived state is request-owned: neither [`SourceIndex`] nor
//! [`SourceProductAutomaton`] is a candidate for a cross-request mask cache.

use std::{fmt, mem::size_of};

use super::CompiledSchema;

/// Request-owned caps for a document substring index and its product plan.
///
/// `max_index_bytes` accounts for the owned source copy, its UTF-8-boundary
/// index, and the two bounded cursor vectors before either vector is allocated.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SourceProductLimits {
    /// Maximum bytes in the source document for one request.
    pub max_source_bytes: usize,
    /// Maximum bytes reserved for the source and occurrence-index working set.
    pub max_index_bytes: usize,
    /// Maximum logical states in the source × JSON × detokenizer product.
    pub max_product_states: usize,
    /// Maximum logical transitions in the source × JSON × detokenizer product.
    pub max_product_transitions: usize,
}

impl Default for SourceProductLimits {
    fn default() -> Self {
        Self {
            max_source_bytes: 1024 * 1024,
            max_index_bytes: 8 * 1024 * 1024,
            max_product_states: 256 * 1024,
            max_product_transitions: 1024 * 1024,
        }
    }
}

/// The explicit policy for repeated source occurrences.
///
/// `All` retains every compatible interval; it never invents a preferred one.
/// `Unique` rejects a repeated match, and `AtByteOffset` makes a caller's
/// separately constrained offset choice explicit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OccurrenceSpec {
    All,
    Unique,
    AtByteOffset(usize),
}

/// Source-language limits for one `x-fnlp-source: "verbatim"` field.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VerbatimFieldSpec {
    /// Minimum logical unescaped UTF-8 byte length.
    pub min_bytes: usize,
    /// Maximum logical unescaped UTF-8 byte length.
    pub max_bytes: usize,
    /// The required handling for repeated intervals.
    pub occurrence: OccurrenceSpec,
}

impl VerbatimFieldSpec {
    /// A bounded field whose every compatible interval is retained.
    pub const fn all(max_bytes: usize) -> Self {
        Self {
            min_bytes: 0,
            max_bytes,
            occurrence: OccurrenceSpec::All,
        }
    }

    fn validate(self) -> Result<(), SourceProductError> {
        if self.min_bytes > self.max_bytes {
            return Err(SourceProductError::InvalidFieldBounds {
                min_bytes: self.min_bytes,
                max_bytes: self.max_bytes,
            });
        }
        Ok(())
    }
}

/// One UTF-8-safe occurrence emitted by an accepting source run.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct SourceMatch {
    pub byte_start: usize,
    pub byte_end: usize,
    pub scalar_start: usize,
    pub scalar_end: usize,
}

/// Source index preflight retained with a request-owned product.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SourceIndexEstimate {
    pub source_bytes: usize,
    pub boundary_count: usize,
    pub boundary_index_bytes: usize,
    pub cursor_bytes: usize,
    pub total_bytes: usize,
}

/// The finite source document language used by a request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceIndex {
    source: String,
    boundaries: Vec<SourceBoundary>,
    estimate: SourceIndexEstimate,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SourceBoundary {
    byte_offset: usize,
    scalar_offset: usize,
}

impl SourceIndex {
    /// Price the source index and every bounded occurrence cursor before
    /// allocating a source-derived structure.
    pub fn preflight(
        source: &str,
        limits: SourceProductLimits,
    ) -> Result<SourceIndexEstimate, SourceProductError> {
        if source.len() > limits.max_source_bytes {
            return Err(SourceProductError::SourceTooLarge {
                observed_bytes: source.len(),
                max_bytes: limits.max_source_bytes,
            });
        }

        let boundary_count = source.chars().count().checked_add(1).ok_or(
            SourceProductError::ArithmeticOverflow {
                surface: "source UTF-8 boundary count",
            },
        )?;
        let boundary_index_bytes = checked_product(
            boundary_count,
            size_of::<SourceBoundary>(),
            "source boundary index bytes",
        )?;
        let active_cursor_bytes = checked_product(
            boundary_count,
            size_of::<ActiveCandidate>(),
            "source active-candidate bytes",
        )?;
        let completed_cursor_bytes = checked_product(
            boundary_count,
            size_of::<SourceMatch>(),
            "source completed-match bytes",
        )?;
        let cursor_bytes = checked_sum(
            active_cursor_bytes,
            completed_cursor_bytes,
            "source cursor bytes",
        )?;
        let total_bytes = checked_sum(
            checked_sum(source.len(), boundary_index_bytes, "source index bytes")?,
            cursor_bytes,
            "source index working-set bytes",
        )?;
        if total_bytes > limits.max_index_bytes {
            return Err(SourceProductError::IndexBudgetExceeded {
                required_bytes: total_bytes,
                max_bytes: limits.max_index_bytes,
            });
        }
        Ok(SourceIndexEstimate {
            source_bytes: source.len(),
            boundary_count,
            boundary_index_bytes,
            cursor_bytes,
            total_bytes,
        })
    }

    /// Build one request-owned source index after successful preflight.
    pub fn build(source: &str, limits: SourceProductLimits) -> Result<Self, SourceProductError> {
        let estimate = Self::preflight(source, limits)?;
        let mut boundaries = Vec::with_capacity(estimate.boundary_count);
        for (scalar_offset, (byte_offset, _)) in source.char_indices().enumerate() {
            boundaries.push(SourceBoundary {
                byte_offset,
                scalar_offset,
            });
        }
        boundaries.push(SourceBoundary {
            byte_offset: source.len(),
            scalar_offset: estimate.boundary_count - 1,
        });
        Ok(Self {
            source: source.to_owned(),
            boundaries,
            estimate,
        })
    }

    pub const fn estimate(&self) -> SourceIndexEstimate {
        self.estimate
    }

    pub const fn source_len(&self) -> usize {
        self.source.len()
    }

    pub const fn boundary_count(&self) -> usize {
        self.boundaries.len()
    }

    /// Begin a start-anywhere substring run for one logical JSON string.
    pub fn start(&self, field: VerbatimFieldSpec) -> Result<SubstringRun<'_>, SourceProductError> {
        field.validate()?;
        let mut active = Vec::with_capacity(self.boundaries.len());
        for (boundary_index, boundary) in self.boundaries.iter().enumerate() {
            active.push(ActiveCandidate {
                start_boundary: boundary_index,
                next_byte: boundary.byte_offset,
            });
        }
        Ok(SubstringRun {
            index: self,
            field,
            active,
            logical_bytes: 0,
        })
    }

    /// Accept a complete logical, unescaped string against the finite source
    /// language.  JSON escaping is intentionally absent from this API.
    pub fn accept(
        &self,
        logical_text: &str,
        field: VerbatimFieldSpec,
    ) -> Result<Vec<SourceMatch>, SourceProductError> {
        let mut run = self.start(field)?;
        run.push_bytes(logical_text.as_bytes())?;
        run.finish()
    }

    fn boundary_at_byte(&self, byte_offset: usize) -> Option<SourceBoundary> {
        self.boundaries
            .binary_search_by_key(&byte_offset, |boundary| boundary.byte_offset)
            .ok()
            .map(|index| self.boundaries[index])
    }
}

/// Source-side state for a single triple-product field run.
///
/// The execution compiler feeds this state only bytes emitted by the tokenizer
/// detokenizer after the JSON lexer has unescaped them.  A failed transition is
/// a typed unsatisfiable-grounding result, never permission to free-generate.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SubstringRun<'a> {
    index: &'a SourceIndex,
    field: VerbatimFieldSpec,
    active: Vec<ActiveCandidate>,
    logical_bytes: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ActiveCandidate {
    start_boundary: usize,
    next_byte: usize,
}

impl<'a> SubstringRun<'a> {
    /// Return the exact source-derived next-byte language for this state.
    pub fn legal_next_bytes(&self) -> [bool; 256] {
        let mut legal = [false; 256];
        if self.logical_bytes == self.field.max_bytes {
            return legal;
        }
        for candidate in &self.active {
            if let Some(byte) = self.index.source.as_bytes().get(candidate.next_byte) {
                legal[usize::from(*byte)] = true;
            }
        }
        legal
    }

    /// Advance one logical unescaped byte through every surviving occurrence.
    pub fn push_byte(&mut self, byte: u8) -> Result<(), SourceProductError> {
        let attempted_bytes =
            self.logical_bytes
                .checked_add(1)
                .ok_or(SourceProductError::ArithmeticOverflow {
                    surface: "logical source string length",
                })?;
        if attempted_bytes > self.field.max_bytes {
            return Err(SourceProductError::MaximumLengthExceeded {
                observed_bytes: attempted_bytes,
                max_bytes: self.field.max_bytes,
            });
        }

        let mut next = Vec::with_capacity(self.active.len());
        for candidate in self.active.drain(..) {
            if self.index.source.as_bytes().get(candidate.next_byte) == Some(&byte) {
                next.push(ActiveCandidate {
                    start_boundary: candidate.start_boundary,
                    next_byte: candidate.next_byte + 1,
                });
            }
        }
        self.logical_bytes = attempted_bytes;
        self.active = next;
        if self.active.is_empty() {
            return Err(SourceProductError::NoSourceContinuation {
                logical_bytes: self.logical_bytes,
            });
        }
        Ok(())
    }

    /// Advance a complete byte-fallback piece.  The caller may split a UTF-8
    /// scalar across pieces; acceptance still requires a final UTF-8 boundary.
    pub fn push_bytes(&mut self, bytes: &[u8]) -> Result<(), SourceProductError> {
        for byte in bytes {
            self.push_byte(*byte)?;
        }
        Ok(())
    }

    pub const fn logical_bytes(&self) -> usize {
        self.logical_bytes
    }

    /// Finish at a logical UTF-8 boundary and recover every compatible source
    /// interval according to the requested occurrence policy.
    pub fn finish(self) -> Result<Vec<SourceMatch>, SourceProductError> {
        if self.logical_bytes < self.field.min_bytes {
            return Err(SourceProductError::MinimumLengthUnmet {
                observed_bytes: self.logical_bytes,
                min_bytes: self.field.min_bytes,
            });
        }

        let mut matches = Vec::with_capacity(self.active.len());
        for candidate in self.active {
            let Some(end) = self.index.boundary_at_byte(candidate.next_byte) else {
                continue;
            };
            let start = self.index.boundaries[candidate.start_boundary];
            matches.push(SourceMatch {
                byte_start: start.byte_offset,
                byte_end: end.byte_offset,
                scalar_start: start.scalar_offset,
                scalar_end: end.scalar_offset,
            });
        }
        if matches.is_empty() {
            return Err(SourceProductError::NoSourceMatch {
                logical_bytes: self.logical_bytes,
            });
        }

        match self.field.occurrence {
            OccurrenceSpec::All => Ok(matches),
            OccurrenceSpec::Unique if matches.len() == 1 => Ok(matches),
            OccurrenceSpec::Unique => Err(SourceProductError::AmbiguousOccurrence {
                match_count: matches.len(),
            }),
            OccurrenceSpec::AtByteOffset(byte_start) => matches
                .into_iter()
                .find(|matched| matched.byte_start == byte_start)
                .map(|matched| vec![matched])
                .ok_or(SourceProductError::RequestedOccurrenceAbsent { byte_start }),
        }
    }
}

/// The tokenizer-detokenization component's checked logical graph dimensions.
///
/// A tokenizer owner supplies these from its approved vocab-trie/transducer
/// plan.  Raw piece text is not accepted here as a substitute for emitted
/// logical bytes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DetokenizationTransducerBounds {
    pub state_count: usize,
    pub transition_count: usize,
}

/// Checked cardinalities for the source × typed-JSON × detokenizer product.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SourceProductEstimate {
    pub substring_state_bound: usize,
    pub substring_transition_bound: usize,
    pub typed_json_states: usize,
    pub typed_json_transitions: usize,
    pub detokenizer_states: usize,
    pub detokenizer_transitions: usize,
    pub product_states: usize,
    pub product_transitions: usize,
}

/// A bounded product admission record plus its executable source component.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceProductAutomaton {
    source: SourceIndex,
    field: VerbatimFieldSpec,
    estimate: SourceProductEstimate,
}

impl SourceProductAutomaton {
    /// Compose a request source language with a compiled `verbatim` schema and
    /// an approved detokenization graph.  All product dimensions are checked
    /// before a request can allocate/enter the source product.
    pub fn compose(
        compiled: &CompiledSchema,
        source: SourceIndex,
        field: VerbatimFieldSpec,
        detokenizer: DetokenizationTransducerBounds,
        limits: SourceProductLimits,
    ) -> Result<Self, SourceProductError> {
        field.validate()?;
        if !compiled.requires_verbatim_source() {
            return Err(SourceProductError::SchemaHasNoVerbatimField);
        }
        if detokenizer.state_count == 0 || detokenizer.transition_count == 0 {
            return Err(SourceProductError::EmptyDetokenizationGraph);
        }

        let substring_state_bound = checked_sum(
            checked_product(
                source.boundary_count(),
                field
                    .max_bytes
                    .checked_add(1)
                    .ok_or(SourceProductError::ArithmeticOverflow {
                        surface: "substring length state bound",
                    })?,
                "substring state bound",
            )?,
            1,
            "substring initial state",
        )?;
        let substring_transition_bound =
            checked_product(substring_state_bound, 256, "substring transition bound")?;
        let typed_json_states = compiled.automaton().state_count();
        let typed_json_transitions = compiled.automaton().transition_count();
        let product_states = checked_product(
            checked_product(
                substring_state_bound,
                typed_json_states,
                "substring × typed-JSON states",
            )?,
            detokenizer.state_count,
            "source product states",
        )?;
        let product_transitions = checked_product(
            checked_product(
                substring_transition_bound,
                typed_json_transitions,
                "substring × typed-JSON transitions",
            )?,
            detokenizer.transition_count,
            "source product transitions",
        )?;
        if product_states > limits.max_product_states {
            return Err(SourceProductError::ProductStateBudgetExceeded {
                required_states: product_states,
                max_states: limits.max_product_states,
            });
        }
        if product_transitions > limits.max_product_transitions {
            return Err(SourceProductError::ProductTransitionBudgetExceeded {
                required_transitions: product_transitions,
                max_transitions: limits.max_product_transitions,
            });
        }

        Ok(Self {
            source,
            field,
            estimate: SourceProductEstimate {
                substring_state_bound,
                substring_transition_bound,
                typed_json_states,
                typed_json_transitions,
                detokenizer_states: detokenizer.state_count,
                detokenizer_transitions: detokenizer.transition_count,
                product_states,
                product_transitions,
            },
        })
    }

    pub const fn estimate(&self) -> SourceProductEstimate {
        self.estimate
    }

    /// Begin the source component at the same field boundary at which the JSON
    /// lexer begins exposing logical unescaped bytes.
    pub fn start(&self) -> Result<SubstringRun<'_>, SourceProductError> {
        self.source.start(self.field)
    }

    /// Test a complete logical field value through the source component.
    pub fn accept_logical_text(
        &self,
        logical_text: &str,
    ) -> Result<Vec<SourceMatch>, SourceProductError> {
        self.source.accept(logical_text, self.field)
    }
}

/// A typed refusal from source-product construction or execution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SourceProductError {
    InvalidFieldBounds {
        min_bytes: usize,
        max_bytes: usize,
    },
    SourceTooLarge {
        observed_bytes: usize,
        max_bytes: usize,
    },
    IndexBudgetExceeded {
        required_bytes: usize,
        max_bytes: usize,
    },
    ProductStateBudgetExceeded {
        required_states: usize,
        max_states: usize,
    },
    ProductTransitionBudgetExceeded {
        required_transitions: usize,
        max_transitions: usize,
    },
    ArithmeticOverflow {
        surface: &'static str,
    },
    SchemaHasNoVerbatimField,
    EmptyDetokenizationGraph,
    MaximumLengthExceeded {
        observed_bytes: usize,
        max_bytes: usize,
    },
    MinimumLengthUnmet {
        observed_bytes: usize,
        min_bytes: usize,
    },
    NoSourceContinuation {
        logical_bytes: usize,
    },
    NoSourceMatch {
        logical_bytes: usize,
    },
    AmbiguousOccurrence {
        match_count: usize,
    },
    RequestedOccurrenceAbsent {
        byte_start: usize,
    },
}

impl fmt::Display for SourceProductError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFieldBounds {
                min_bytes,
                max_bytes,
            } => write!(
                formatter,
                "verbatim field minimum {min_bytes} exceeds maximum {max_bytes}"
            ),
            Self::SourceTooLarge {
                observed_bytes,
                max_bytes,
            } => write!(
                formatter,
                "source has {observed_bytes} bytes and exceeds request cap {max_bytes}"
            ),
            Self::IndexBudgetExceeded {
                required_bytes,
                max_bytes,
            } => write!(
                formatter,
                "source index requires {required_bytes} bytes and exceeds request cap {max_bytes}"
            ),
            Self::ProductStateBudgetExceeded {
                required_states,
                max_states,
            } => write!(
                formatter,
                "source product requires {required_states} states and exceeds cap {max_states}"
            ),
            Self::ProductTransitionBudgetExceeded {
                required_transitions,
                max_transitions,
            } => write!(
                formatter,
                "source product requires {required_transitions} transitions and exceeds cap {max_transitions}"
            ),
            Self::ArithmeticOverflow { surface } => {
                write!(formatter, "checked arithmetic overflow in {surface}")
            }
            Self::SchemaHasNoVerbatimField => {
                formatter.write_str("source product requires x-fnlp-source:verbatim")
            }
            Self::EmptyDetokenizationGraph => {
                formatter.write_str("detokenization graph must have states and transitions")
            }
            Self::MaximumLengthExceeded {
                observed_bytes,
                max_bytes,
            } => write!(
                formatter,
                "logical source string has {observed_bytes} bytes and exceeds {max_bytes}"
            ),
            Self::MinimumLengthUnmet {
                observed_bytes,
                min_bytes,
            } => write!(
                formatter,
                "logical source string has {observed_bytes} bytes and is shorter than {min_bytes}"
            ),
            Self::NoSourceContinuation { logical_bytes } => write!(
                formatter,
                "logical source prefix of {logical_bytes} bytes has no source continuation"
            ),
            Self::NoSourceMatch { logical_bytes } => write!(
                formatter,
                "logical source string of {logical_bytes} bytes has no UTF-8-boundary source match"
            ),
            Self::AmbiguousOccurrence { match_count } => write!(
                formatter,
                "source string has {match_count} compatible occurrences; an explicit occurrence is required"
            ),
            Self::RequestedOccurrenceAbsent { byte_start } => write!(
                formatter,
                "no compatible source occurrence begins at byte {byte_start}"
            ),
        }
    }
}

impl std::error::Error for SourceProductError {}

fn checked_product(
    left: usize,
    right: usize,
    surface: &'static str,
) -> Result<usize, SourceProductError> {
    left.checked_mul(right)
        .ok_or(SourceProductError::ArithmeticOverflow { surface })
}

fn checked_sum(
    left: usize,
    right: usize,
    surface: &'static str,
) -> Result<usize, SourceProductError> {
    left.checked_add(right)
        .ok_or(SourceProductError::ArithmeticOverflow { surface })
}

#[cfg(test)]
mod tests {
    use super::{
        DetokenizationTransducerBounds, OccurrenceSpec, SourceIndex, SourceProductAutomaton,
        SourceProductError, SourceProductLimits, VerbatimFieldSpec,
    };
    use crate::grammar::{CompileLimits, compile_json_schema};

    fn index(source: &str) -> SourceIndex {
        SourceIndex::build(source, SourceProductLimits::default())
            .expect("small source fixture must fit the request index budget")
    }

    #[test]
    fn substring_membership_exposes_only_source_derived_next_bytes() {
        let source = index("eclair: e\u{301}clair");
        let mut run = source
            .start(VerbatimFieldSpec {
                min_bytes: 0,
                max_bytes: 16,
                occurrence: OccurrenceSpec::All,
            })
            .expect("valid field bounds");
        assert!(run.legal_next_bytes()[usize::from(b'e')]);
        assert!(!run.legal_next_bytes()[usize::from(b'z')]);
        run.push_bytes("e\u{301}c".as_bytes())
            .expect("the logical unescaped bytes occur in the source");
        let matches = run.finish().expect("substring must finish at a boundary");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].byte_start, 8);
        assert_eq!(matches[0].scalar_start, 8);
        assert_eq!(matches[0].scalar_end, 11);

        let error = source
            .accept("invented", VerbatimFieldSpec::all(16))
            .expect_err("off-source bytes cannot enter the source language");
        assert!(matches!(
            error,
            SourceProductError::NoSourceContinuation { .. }
        ));
    }

    #[test]
    fn occurrence_rules_keep_repeated_overlapping_and_empty_matches_explicit() {
        let source = index("ababa");
        let field = VerbatimFieldSpec::all(3);
        let repeated = source.accept("aba", field).expect("all matches retained");
        assert_eq!(
            repeated
                .iter()
                .map(|matched| (matched.byte_start, matched.byte_end))
                .collect::<Vec<_>>(),
            vec![(0, 3), (2, 5)]
        );
        let ambiguous = source
            .accept(
                "aba",
                VerbatimFieldSpec {
                    occurrence: OccurrenceSpec::Unique,
                    ..field
                },
            )
            .expect_err("a repeated source match is not silently selected");
        assert_eq!(
            ambiguous,
            SourceProductError::AmbiguousOccurrence { match_count: 2 }
        );
        let selected = source
            .accept(
                "aba",
                VerbatimFieldSpec {
                    occurrence: OccurrenceSpec::AtByteOffset(2),
                    ..field
                },
            )
            .expect("a separately constrained occurrence is explicit");
        assert_eq!((selected[0].byte_start, selected[0].byte_end), (2, 5));

        let empty = source
            .accept("", VerbatimFieldSpec::all(0))
            .expect("the empty language member has every source boundary");
        assert_eq!(empty.len(), 6);
        let too_short = source
            .accept(
                "",
                VerbatimFieldSpec {
                    min_bytes: 1,
                    max_bytes: 3,
                    occurrence: OccurrenceSpec::All,
                },
            )
            .expect_err("minimum length remains a source-product transition");
        assert!(matches!(
            too_short,
            SourceProductError::MinimumLengthUnmet { .. }
        ));
    }

    #[test]
    fn product_automaton_preflights_all_three_component_bounds() {
        let compiled = compile_json_schema(
            r#"{"type":"object","additionalProperties":false,"properties":{"quote":{"type":"string","maxLength":8,"x-fnlp-source":"verbatim"}},"required":["quote"]}"#,
            CompileLimits::default(),
        )
        .expect("fixture schema compiles");
        let product = SourceProductAutomaton::compose(
            &compiled,
            index("alpha beta"),
            VerbatimFieldSpec::all(8),
            DetokenizationTransducerBounds {
                state_count: 2,
                transition_count: 4,
            },
            SourceProductLimits {
                max_product_states: 10_000,
                max_product_transitions: 10_000_000,
                ..SourceProductLimits::default()
            },
        )
        .expect("small source product fits checked bounds");
        assert!(product.estimate().substring_state_bound > 0);
        assert_eq!(
            product
                .accept_logical_text("beta")
                .expect("product source component accepts source text")[0]
                .byte_start,
            6
        );

        let error = SourceProductAutomaton::compose(
            &compiled,
            index("alpha beta"),
            VerbatimFieldSpec::all(8),
            DetokenizationTransducerBounds {
                state_count: 2,
                transition_count: 4,
            },
            SourceProductLimits {
                max_product_states: 1,
                ..SourceProductLimits::default()
            },
        )
        .expect_err("product state cap rejects before product admission");
        assert!(matches!(
            error,
            SourceProductError::ProductStateBudgetExceeded { .. }
        ));
    }
}
