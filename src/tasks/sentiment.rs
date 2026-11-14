//! Independent, finite-candidate affect dimensions for sentiment-v1.
//!
//! A distribution over one dimension is not a distribution over emotions,
//! calibrated correctness confidence, or a psychological measurement.
//! Each dimension binds its own exact TaskIR prompt and complete candidate set.

mod distribution;
mod native;
mod planning;
pub(crate) mod aggregation;
pub use aggregation::{
    SENTIMENT_AGGREGATION_PROFILE, SentimentAggregate, SentimentAggregateDimension,
    SentimentAggregationError, SentimentAggregationPolicy, SentimentMixtureWeight,
};
pub use distribution::{
    SentimentAnchor, SentimentAxis, SentimentAxisInput, SentimentDecision,
    SentimentDimension, SentimentError, SentimentLimits, SentimentLogits,
    SentimentOptions, SentimentPlan, SentimentPolicy, SentimentResult, SentimentWork,
};
pub use native::{EagerSentimentRun, SENTIMENT_NATIVE_EXECUTION, SentimentNativeError};
pub use planning::{SENTIMENT_PROMPT_VERSION, SentimentPlanner, SentimentPlanningError, SentimentRequest};
pub mod quantized;
pub mod grouped_int4;
pub mod batch;
/// Independently sampled, schema-locked scores with bounded justifications.
pub mod sampled;
