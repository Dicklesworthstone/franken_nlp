//! Independent, finite-candidate affect dimensions for sentiment-v1.
//!
//! A distribution over one dimension is not a distribution over emotions,
//! calibrated correctness confidence, or a psychological measurement.
//! Each dimension binds its own exact TaskIR prompt and complete candidate set.

mod distribution;
pub use distribution::{
    SentimentAnchor, SentimentAxis, SentimentAxisInput, SentimentDecision,
    SentimentDimension, SentimentError, SentimentLimits, SentimentLogits,
    SentimentOptions, SentimentPlan, SentimentPolicy, SentimentResult, SentimentWork,
};
