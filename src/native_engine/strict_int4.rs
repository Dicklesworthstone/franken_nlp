//! Explicit full-model grouped-INT4 candidate. This is not an INT8 profile,
//! canonical fnlpq format, activation authority, or BF16 fidelity claim.

pub mod weights;
pub use weights::{Int4ModelError, Int4ModelIdentity, Int4ModelLoadBudget,
    Int4ModelLoadRequirement, Int4ModelWeights};

pub const STRICT_INT4_PROFILE: &str = "strict-grouped-int4-bf16-rails-v1";
pub const STRICT_INT4_EXECUTION: &str = "grouped-int4-bf16-rails-eager-gqa-v1";
