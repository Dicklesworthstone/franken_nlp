//! Bounded PII detection and redaction. Detector coverage is versioned and
//! deliberately narrow; neither a clean scan nor pseudonyms imply anonymity.

use std::{error::Error, fmt};
use serde::{Deserialize, Serialize};
use crate::validation::grounded_fields::VerifiedSourceSpan;

pub mod detectors;

/// Type names are stable inputs to policy and pseudonym domain separation.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PiiKind {
    Email, Phone, Url, IpAddress, CreditCard, Date,
    Person, Organization, Location, Time, Money, Product, Event,
}
impl PiiKind {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Email => "email", Self::Phone => "phone", Self::Url => "url",
            Self::IpAddress => "ip_address", Self::CreditCard => "credit_card", Self::Date => "date",
            Self::Person => "person", Self::Organization => "organization", Self::Location => "location",
            Self::Time => "time", Self::Money => "money", Self::Product => "product", Self::Event => "event",
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Detector { EmailAsciiV1, PhoneAsciiV1, UrlHttpV1, IpLiteralV1, CardLuhnV1, DateIsoV1, NerSourceV1 }

/// Original-source coordinates, never copied private matched text. This is
/// descriptive input, not authority: editing must recheck every coordinate.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Detection {
    pub kind: PiiKind,
    pub detector: Detector,
    pub span: VerifiedSourceSpan,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RedactError {
    InvalidOptions, InputBudget, DetectionBudget, WorkBudget, CandidateBudget,
    AllocationRefused, InvalidSpan, InvalidNerEvidence, OutputBudget, Serialization,
    MissingPolicy, MissingKey, KeyMismatch, Collision, VerificationResidual { count: usize },
}
impl fmt::Display for RedactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Errors intentionally contain neither matched text nor key material.
        write!(f, "redaction has no result: {}", match self {
            Self::InvalidOptions => "invalid options", Self::InputBudget => "input byte budget",
            Self::DetectionBudget => "detection count budget", Self::WorkBudget => "work budget",
            Self::CandidateBudget => "candidate length budget", Self::AllocationRefused => "allocation refused",
            Self::InvalidSpan => "invalid source coordinates", Self::InvalidNerEvidence => "invalid or incomplete NER evidence",
            Self::OutputBudget => "complete output budget", Self::Serialization => "serialization failed",
            Self::MissingPolicy => "missing type action", Self::MissingKey => "pseudonym key required",
            Self::KeyMismatch => "pseudonym key commitment mismatch", Self::Collision => "pseudonym collision",
            Self::VerificationResidual { .. } => "residual detector findings",
        })
    }
}
impl Error for RedactError {}
