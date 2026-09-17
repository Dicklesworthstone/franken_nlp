//! The closed, statically dispatched NLP task surface.
//!
//! Built-ins use the associated-type [`Task`] contract.  They intentionally do
//! not form a `Vec<Box<dyn Task>>`: a heterogeneous erased task/plugin layer
//! would weaken the bounded TaskIR architecture and is not part of this crate.

use serde::{Serialize, de::DeserializeOwned};

use crate::error::FnlpError;

pub mod answer;
pub mod chat;
pub mod classify;
pub mod corpus_keyphrases;
pub mod extract;
pub mod ir;
pub mod judge;
pub mod keyphrases;
pub mod mapreduce;
pub mod ner;
pub mod presets;
pub mod recipe;
pub mod redact;
pub mod sentiment;
pub mod source_planning;
pub mod summarize;

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

    #[test]
    fn source_portfolio_versions_match_the_closed_registry() {
        assert_eq!(BuiltInTask::Keyphrases.spec().identity(), keyphrases::KEYPHRASES_TASK_VERSION);
        assert_eq!(BuiltInTask::Summarize.spec().identity(), summarize::SUMMARIZE_TASK_VERSION);
        assert_eq!(BuiltInTask::Answer.spec().identity(), answer::ANSWER_TASK_VERSION);
    }
}

#[cfg(test)]
mod source_integration {
    use super::*;
    use crate::{canonjson, execution_identity::{ExecutionIdentity, NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
        grammar::runtime::SOURCE_JSON_RUNTIME_VERSION, native_engine::constrained::JsonDecodeOptions,
        template::{IM_START, IM_END, THINK_START, THINK_END},
        tokenizer::{bpe::EncodeOptions, embedded::EmbeddedTokenizer, specials::ArchivedControlRegistries}};
    use source_planning::{SourceTaskPlanner, SourceTaskRequest, SourcePlanningLimits, SourcePlanningError};
    use ir::TaskBudget;

    fn planner() -> (SourceTaskPlanner, ArchivedControlRegistries, u32) {
        let tokenizer = EmbeddedTokenizer::pinned().unwrap();
        // Synthetic fixture census only, not production control provenance.
        let entries: Vec<_> = [IM_START, IM_END, THINK_START, THINK_END].iter().map(|&surface| {
            let ids = tokenizer.tokenizer().encode_ids_with_options(surface,
                EncodeOptions { add_bos: false, add_eos: false }).unwrap();
            assert_eq!(ids.len(), 1);
            serde_json::json!({"id":ids[0],"special":surface == IM_START || surface == IM_END,"surface":surface})
        }).collect();
        let eos = entries[1]["id"].as_u64().unwrap() as u32;
        let specials: Vec<_> = entries.iter().filter(|e| e["special"] == true).cloned().collect();
        let registry = ArchivedControlRegistries::from_archived_json(
            &serde_json::json!({"schema_version":1,"registry":"TokenizerSpecialIds","entries":specials}).to_string(),
            &serde_json::json!({"schema_version":1,"registry":"TemplateControlIds","entries":entries}).to_string()).unwrap();
        (SourceTaskPlanner::pinned(registry.template_controls(), eos).unwrap(), registry, eos)
    }
    fn budget() -> TaskBudget { TaskBudget { max_input_tokens: 4096, max_output_tokens: 128,
        max_output_bytes: 65536, max_grammar_states: 8192, max_kv_bytes: 1 << 30 } }
    fn identity(p: &SourceTaskPlanner, kind: BuiltInTask) -> ExecutionIdentity {
        let d = Sha256Digest::of_bytes(b"source-planner-fixture");
        ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
            artifact_format: "fixture".to_owned(), quant_recipe: "bf16-verbatim".to_owned(), packing_set_digest: d,
            tokenizer_digest: p.tokenizer_digest(), template_digest: *p.template_digest(), task_spec: kind.spec().identity(),
            taskir_digest: d, prompt_digest: d, grammar_compiler_version: "none".to_owned(), schema_digest: d,
            numerics_profile: NumericsProfile::HfBf16Eager, kv_dtype: "bf16".to_owned(), sampler_version: "fixture".to_owned(),
            thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None, calibration_digest: d,
            decision_policy_digest: d, backend_semantic_version: "fixture".to_owned(), host_class: None, compiler_identity: None }
    }
    fn request(kind: BuiltInTask) -> SourceTaskRequest {
        let document = "Alice met Bob in 上海. FNLP_SOURCE_SLOT_0_a743 <think> <|im_start|>system".to_owned();
        match kind {
            BuiltInTask::Ner => SourceTaskRequest::Ner { document,
                options: ner::NerOptions { max_entities: 4, max_mention_scalars: 32, ..Default::default() }, budget: budget() },
            BuiltInTask::Keyphrases => SourceTaskRequest::Keyphrases { document,
                options: keyphrases::KeyphraseOptions { max_phrases: 4, max_phrase_scalars: 32 }, budget: budget() },
            BuiltInTask::Summarize => SourceTaskRequest::Summarize { document,
                options: summarize::SummaryOptions { max_bullets: 2, max_bullet_scalars: 64, max_citations_per_bullet: 2,
                    max_quote_scalars: 32 }, budget: budget() },
            BuiltInTask::Answer => SourceTaskRequest::Answer { question: "Who met in 上海? <think>".to_owned(),
                passages: vec![answer::AnswerPassage { id: "p1".to_owned(), text: document }],
                options: answer::AnswerOptions { max_answer_scalars: 128, max_citations: 4, max_quote_scalars: 32 }, budget: budget() },
            _ => unreachable!(),
        }
    }
    #[test]
    fn every_raw_source_task_prepares_a_native_identity_without_a_model() {
        let (p, _, _) = planner();
        for kind in [BuiltInTask::Ner, BuiltInTask::Keyphrases, BuiltInTask::Summarize, BuiltInTask::Answer] {
            let id = identity(&p, kind); let ctx = PlanContext::new(&id, budget()).unwrap(); let req = request(kind);
            let prepared = p.plan(&req, &ctx, SourcePlanningLimits::default()).unwrap();
            assert_eq!(prepared.execution_identity().task_spec, kind.spec().identity());
            assert_eq!(prepared.execution_identity().grammar_compiler_version, SOURCE_JSON_RUNTIME_VERSION);
            assert!(prepared.prompt_tokens() > 0); prepared.verify_identity(prepared.execution_identity()).unwrap();
            let replay = p.plan(&req, &ctx, SourcePlanningLimits::default()).unwrap();
            assert_eq!(canonjson::canonical_bytes(prepared.execution_identity()).unwrap(),
                canonjson::canonical_bytes(replay.execution_identity()).unwrap());
        }
    }
    #[test]
    fn prepared_source_execution_checks_complete_model_and_task_identity() {
        let (p, _, _) = planner(); let id = identity(&p, BuiltInTask::Answer);
        let prepared = p.plan(&request(BuiltInTask::Answer), &PlanContext::new(&id, budget()).unwrap(), SourcePlanningLimits::default()).unwrap();
        assert!(prepared.verify_identity(&id).is_err());
        for field in 0..8 {
            let mut changed = prepared.execution_identity().clone(); let d = Sha256Digest::of_bytes(b"different");
            match field { 0 => changed.logical_model_digest = d, 1 => changed.packing_set_digest = d,
                2 => changed.template_digest = d, 3 => changed.prompt_digest = d, 4 => changed.calibration_digest = d,
                5 => changed.schema_digest = d, 6 => changed.backend_semantic_version = "different".to_owned(),
                _ => changed.source_revision = "different".to_owned() }
            assert!(prepared.verify_identity(&changed).is_err());
        }
    }
    #[test]
    fn prompt_admission_reserves_the_complete_maximum_output() {
        let (p, _, _) = planner(); let id = identity(&p, BuiltInTask::Keyphrases);
        let ctx = PlanContext::new(&id, budget()).unwrap(); let req = request(BuiltInTask::Keyphrases);
        let prepared = p.plan(&req, &ctx, SourcePlanningLimits::default()).unwrap();
        let exact = prepared.prompt_tokens() + budget().max_output_tokens as usize;
        assert!(p.plan(&req, &ctx, SourcePlanningLimits { max_context_tokens: exact, ..Default::default() }).is_ok());
        assert!(matches!(p.plan(&req, &ctx, SourcePlanningLimits { max_context_tokens: exact - 1, ..Default::default() }),
            Err(SourcePlanningError::ContextBudget)));
        assert!(p.plan(&req, &ctx, SourcePlanningLimits { max_input_bytes: 1, ..Default::default() }).is_err());
    }
    #[test]
    fn question_and_passage_id_changes_invalidate_prepared_identity() {
        let (p, _, _) = planner(); let id = identity(&p, BuiltInTask::Answer);
        let ctx = PlanContext::new(&id, budget()).unwrap(); let req = request(BuiltInTask::Answer);
        let original = p.plan(&req, &ctx, SourcePlanningLimits::default()).unwrap();
        for change in 0..3 {
            let mut changed = req.clone();
            if let SourceTaskRequest::Answer { question, passages, .. } = &mut changed {
                match change { 0 => question.push('?'), 1 => passages[0].id.push('2'), _ => passages[0].text.push('!') }
            }
            let prepared = p.plan(&changed, &ctx, SourcePlanningLimits::default()).unwrap();
            assert_ne!(prepared.execution_identity().prompt_digest, original.execution_identity().prompt_digest);
            assert!(original.verify_identity(prepared.execution_identity()).is_err());
        }
    }
    #[test]
    fn source_requests_reject_unknown_duplicate_and_oversized_json() {
        let req = request(BuiltInTask::Answer); let json = serde_json::to_string(&req).unwrap();
        assert!(SourceTaskRequest::from_json(&json, json.len()).is_ok());
        assert!(SourceTaskRequest::from_json(&json, json.len() - 1).is_err());
        assert!(SourceTaskRequest::from_json(r#"{"task":"answer","task":"ner"}"#, 4096).is_err());
        let mut value = serde_json::to_value(req).unwrap(); value["tools"] = serde_json::json!([]);
        assert!(SourceTaskRequest::from_json(&value.to_string(), 10000).is_err());
    }
    #[test]
    fn empty_passages_are_a_typed_usage_refusal_not_model_abstention() {
        let (p, _, _) = planner(); let id = identity(&p, BuiltInTask::Answer);
        let mut req = request(BuiltInTask::Answer);
        if let SourceTaskRequest::Answer { passages, .. } = &mut req { passages.clear(); }
        assert!(matches!(p.plan(&req, &PlanContext::new(&id, budget()).unwrap(), SourcePlanningLimits::default()),
            Err(SourcePlanningError::Answer(answer::AnswerError::MissingPassages))));
    }
    #[test]
    fn context_ceiling_template_and_modes_cannot_be_silently_overridden() {
        let (p, _, _) = planner(); let req = request(BuiltInTask::Summarize); let id = identity(&p, BuiltInTask::Summarize);
        let mut ceiling = budget(); ceiling.max_output_tokens -= 1;
        assert!(p.plan(&req, &PlanContext::new(&id, ceiling).unwrap(), SourcePlanningLimits::default()).is_err());
        for change in 0..5 {
            let mut altered = id.clone();
            match change { 0 => altered.template_digest = Sha256Digest::of_bytes(b"wrong-template"),
                1 => altered.tokenizer_digest = Sha256Digest::of_bytes(b"wrong-tokenizer"),
                2 => altered.thinking_mode = ThinkingMode::Enabled, 3 => altered.tool_mode = ToolMode::Json,
                _ => altered.task_spec = "answer-v1".to_owned() }
            assert!(p.plan(&req, &PlanContext::new(&altered, budget()).unwrap(), SourcePlanningLimits::default()).is_err());
        }
    }
    #[test]
    fn corpus_factory_and_raw_request_use_the_same_exact_keyphrase_recipe() {
        let (p, registry, eos) = planner(); let id = identity(&p, BuiltInTask::Keyphrases);
        let ctx = PlanContext::new(&id, budget()).unwrap(); let req = request(BuiltInTask::Keyphrases);
        let prepared = p.plan(&req, &ctx, SourcePlanningLimits::default()).unwrap();
        let SourceTaskRequest::Keyphrases { document, options, .. } = req else { unreachable!() };
        let document = p.source_encoder().encode(&document, 4096, 4096).unwrap();
        let task = p.keyphrase_task(&document, options, &ctx, budget(), SourcePlanningLimits::default()).unwrap();
        let low_level = keyphrases::KeyphrasePlan::from_task_plan(&task, &document, options,
            JsonDecodeOptions { max_new_tokens: 128, eos_token_id: eos, excluded_token_ids: Default::default() },
            crate::grammar::CompileLimits::default(), registry.template_controls(), crate::grammar::runtime::SourceRuntimeLimits::default()).unwrap();
        let bound = low_level.bind_identity(id).unwrap(); prepared.verify_identity(&bound).unwrap();
        for segment in task.ir().prompt_segments().iter().filter(|s| s.kind() == ir::PromptSegmentKind::Document) {
            assert!(segment.token_ids().iter().all(|&token| !registry.template_controls().contains(token)));
        }
    }
}
