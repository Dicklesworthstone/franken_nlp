//! The closed, statically dispatched NLP task surface.
//!
//! Built-ins use the associated-type [`Task`] contract.  They intentionally do
//! not form a `Vec<Box<dyn Task>>`: a heterogeneous erased task/plugin layer
//! would weaken the bounded TaskIR architecture and is not part of this crate.

use serde::{Serialize, de::DeserializeOwned};

use crate::error::FnlpError;

pub mod chat;
pub mod classify;
pub mod extract;
pub mod ir;
pub mod judge;
pub mod mapreduce;
pub mod ner;
pub mod presets;
pub mod recipe;
pub mod redact;
pub mod sentiment;

pub use ir::{DecodeOutput, IndependentValidator, PlanContext, TaskPlan, TaskSpec};

/// The exact static task contract shared by library, CLI, and NDJSON callers.
pub trait Task {
    /// A serde request type shared by every public entrypoint for this task.
    type Request: DeserializeOwned + Serialize;
    /// A serde response type shared by every public entrypoint for this task.
    type Response: DeserializeOwned + Serialize;

    /// Immutable task metadata: name, version, schemas, and preset identifiers.
    fn spec(&self) -> &'static TaskSpec;

    /// Compile prompt, finite decode strategy, grammar reference, and bounds
    /// before model admission.
    fn plan(&self, req: &Self::Request, ctx: &PlanContext<'_>) -> Result<TaskPlan, FnlpError>;

    /// Consume decoded bytes only after independent validation.  The request
    /// and plan remain available for source-grounding and offset checks.
    fn finalize(
        &self,
        req: &Self::Request,
        plan: &TaskPlan,
        raw: DecodeOutput,
        validator: &IndependentValidator,
    ) -> Result<Self::Response, FnlpError>;
}

const NO_PRESETS: &[&str] = &[];
const CLASSIFY_PRESETS: &[&str] = &["topic-v1", "intent-v1", "moderation-v1"];
const SENTIMENT_PRESETS: &[&str] = &["reviews-v1", "earnings-v1", "support-v1"];
const REDACT_PRESETS: &[&str] = &["pii-default-v1"];

static EXTRACT_SPEC: TaskSpec = TaskSpec::new(
    "extract",
    "v1",
    "extract-request-v1",
    "extract-response-v1",
    NO_PRESETS,
);
static NER_SPEC: TaskSpec =
    TaskSpec::new("ner", "v1", "ner-request-v1", "ner-response-v1", NO_PRESETS);
static RESOLVE_SPEC: TaskSpec = TaskSpec::new(
    "resolve",
    "v1",
    "resolve-request-v1",
    "resolve-response-v1",
    NO_PRESETS,
);
static SENTIMENT_SPEC: TaskSpec = TaskSpec::new(
    "sentiment",
    "v1",
    "sentiment-request-v1",
    "sentiment-response-v1",
    SENTIMENT_PRESETS,
);
static CLASSIFY_SPEC: TaskSpec = TaskSpec::new(
    "classify",
    "v1",
    "classify-request-v1",
    "classify-response-v1",
    CLASSIFY_PRESETS,
);
static JUDGE_SPEC: TaskSpec = TaskSpec::new(
    "judge",
    "v1",
    "judge-request-v1",
    "judge-response-v1",
    NO_PRESETS,
);
static REDACT_SPEC: TaskSpec = TaskSpec::new(
    "redact",
    "v1",
    "redact-request-v1",
    "redact-response-v1",
    REDACT_PRESETS,
);
static SUMMARIZE_SPEC: TaskSpec = TaskSpec::new(
    "summarize",
    "v1",
    "summarize-request-v1",
    "summarize-response-v1",
    NO_PRESETS,
);
static KEYPHRASES_SPEC: TaskSpec = TaskSpec::new(
    "keyphrases",
    "v1",
    "keyphrases-request-v1",
    "keyphrases-response-v1",
    NO_PRESETS,
);
static ANSWER_SPEC: TaskSpec = TaskSpec::new(
    "answer",
    "v1",
    "answer-request-v1",
    "answer-response-v1",
    NO_PRESETS,
);
static GENERATE_SPEC: TaskSpec = TaskSpec::new(
    "generate",
    "v1",
    "generate-request-v1",
    "generate-response-v1",
    NO_PRESETS,
);
static CHAT_SPEC: TaskSpec = TaskSpec::new(
    "chat",
    "v1",
    "chat-request-v1",
    "chat-response-v1",
    NO_PRESETS,
);

/// All built-ins are a closed enum with static dispatch.  Public data-only
/// recipes compile to TaskIR; they never add a Rust enum variant at runtime.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BuiltInTask {
    Extract,
    Ner,
    Resolve,
    Sentiment,
    Classify,
    Judge,
    Redact,
    Summarize,
    Keyphrases,
    Answer,
    Generate,
    Chat,
}

impl BuiltInTask {
    /// Every admitted built-in task.  Additions require changing this list and
    /// the exhaustive match below in the same reviewable commit.
    pub const ALL: [Self; 12] = [
        Self::Extract,
        Self::Ner,
        Self::Resolve,
        Self::Sentiment,
        Self::Classify,
        Self::Judge,
        Self::Redact,
        Self::Summarize,
        Self::Keyphrases,
        Self::Answer,
        Self::Generate,
        Self::Chat,
    ];

    /// Static registry dispatch, intentionally with no wildcard arm.
    #[must_use]
    pub fn spec(self) -> &'static TaskSpec {
        match self {
            Self::Extract => &EXTRACT_SPEC,
            Self::Ner => &NER_SPEC,
            Self::Resolve => &RESOLVE_SPEC,
            Self::Sentiment => &SENTIMENT_SPEC,
            Self::Classify => &CLASSIFY_SPEC,
            Self::Judge => &JUDGE_SPEC,
            Self::Redact => &REDACT_SPEC,
            Self::Summarize => &SUMMARIZE_SPEC,
            Self::Keyphrases => &KEYPHRASES_SPEC,
            Self::Answer => &ANSWER_SPEC,
            Self::Generate => &GENERATE_SPEC,
            Self::Chat => &CHAT_SPEC,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn built_in_task_registry_is_closed_and_exhaustive() {
        assert_eq!(BuiltInTask::ALL.len(), 12);
        for task in BuiltInTask::ALL {
            let expected_name = match task {
                BuiltInTask::Extract => "extract",
                BuiltInTask::Ner => "ner",
                BuiltInTask::Resolve => "resolve",
                BuiltInTask::Sentiment => "sentiment",
                BuiltInTask::Classify => "classify",
                BuiltInTask::Judge => "judge",
                BuiltInTask::Redact => "redact",
                BuiltInTask::Summarize => "summarize",
                BuiltInTask::Keyphrases => "keyphrases",
                BuiltInTask::Answer => "answer",
                BuiltInTask::Generate => "generate",
                BuiltInTask::Chat => "chat",
            };
            let spec = task.spec();
            spec.validate().expect("static task spec must validate");
            assert_eq!(spec.name(), expected_name);
            assert_eq!(spec.identity(), format!("{expected_name}-v1"));
        }
    }
}
