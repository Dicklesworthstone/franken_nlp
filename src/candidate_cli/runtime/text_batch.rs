//! Hosted text corpus requests with one resident model and one deadline.
//! Serial by default; explicit document cohorts use shared-weight INT8 work.
use super::*;
use super::source_tasks::Session;
use crate::candidate_cli::text_batch::{self as wire, Content, Ledger, Output, Record,
    ReservedWork, TextBatchCommand, FOOTER_BYTES, PROTOCOL};
use crate::tasks::chat::quantized::PreparedInt8Chat;
use crate::native_engine::decode::{DecodeCancellationKind, DecodeStepControl};
use crate::tokenizer::untrusted::UntrustedEncoding;
use std::io::{BufRead, BufReader};
mod cohort;

#[cfg(test)]
struct NeverCancelled;
#[cfg(test)]
impl DecodeStepControl for NeverCancelled {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}

pub(in crate::candidate_cli) fn execute(command: TextBatchCommand, limits: Limits,
    input: &mut impl Read, output: &mut impl Write) -> Result<(), CandidateError> {
    // This owner is declared before all corpus/model/planner/transport values.
    let session = Session::new(&command.common, limits)?;
    if command.common.input.as_os_str() == "-" {
        run(&session, &command, limits, &mut BufReader::new(input), output)
    } else {
        let file = File::open(&command.common.input).map_err(|_| CandidateError::Input)?;
        run(&session, &command, limits, &mut BufReader::new(file), output)
    }
}

struct Planner { native: Int8ChatPlanner, options: GenerationOptions, budget: TaskBudget }
impl Planner {
    fn new(args: &CandidateArgs, limits: Limits, facts: &ArtifactIdentity) -> Result<Self, CandidateError> {
        let controls = pinned_controls::pinned().map_err(|_| CandidateError::Identity)?;
        let eos = controls.template_controls().entries().iter()
            .find(|entry| entry.special && entry.surface == crate::template::IM_END)
            .map(|entry| entry.id).ok_or(CandidateError::Identity)?;
        let budget = TaskBudget { max_input_tokens: limits.max_prompt_tokens as u32,
            max_output_tokens: args.max_new_tokens as u32, max_output_bytes: limits.result_bytes as u64,
            max_grammar_states: 1, max_kv_bytes: limits.kv_bytes };
        let native = Int8ChatPlanner::pinned(controls.template_controls(), eos, candidate_identity(facts)?, budget,
            ChatLimits { max_messages: 128, max_message_bytes: args.max_input_bytes,
                max_total_message_bytes: args.max_input_bytes, generation: args.generation_limits(limits) })
            .map_err(|_| CandidateError::Planning)?
            .with_document_encoding(args.input_encoding_policy(limits)?)
            .map_err(|_| CandidateError::Planning)?;
        Ok(Self { native, options: args.options(eos)?, budget })
    }
    fn prepare_with_control(&self, record: Record, control: &mut impl DecodeStepControl)
        -> Result<(String, PreparedInt8Chat), CandidateError> {
        planning_checkpoint(control)?;
        let Record { id, sample_index, content } = record;
        let prepared = match content {
            Content::Generate(prompt) => self.native.plan_generate_with_control(&GenerateRequest {
                item_id: id.clone(), sample_index, prompt, generation: self.options.clone(), budget: self.budget }, control),
            Content::Chat(messages) => self.native.plan_chat_with_control(&ChatRequest {
                item_id: id.clone(), sample_index, messages, generation: self.options.clone(), budget: self.budget }, control),
        }.map_err(|error| match error.cancellation() {
            Some(cause) => planning_cancellation(cause), None => CandidateError::Planning,
        })?;
        planning_checkpoint(control)?;
        Ok((id, prepared))
    }
    #[cfg(test)]
    fn prepare(&self, record: Record) -> Result<(String, PreparedInt8Chat), CandidateError> {
        self.prepare_with_control(record, &mut NeverCancelled)
    }
}

fn planning_cancellation(cause: DecodeCancellationKind) -> CandidateError {
    match cause {
        DecodeCancellationKind::Deadline | DecodeCancellationKind::Timeout => CandidateError::Timeout,
        _ => CandidateError::Execution,
    }
}
fn planning_checkpoint(control: &mut impl DecodeStepControl) -> Result<(), CandidateError> {
    control.checkpoint(0).map_or(Ok(()), |cause| Err(planning_cancellation(cause)))
}

fn run(session: &Session, command: &TextBatchCommand, limits: Limits,
    input: &mut impl BufRead, output: &mut impl Write) -> Result<(), CandidateError> {
    if let Some(width) = command.cohort_rows { return cohort::run(session, command, limits, input, output, width); }
    let args = &command.common;
    let encoding = args.input_encoding_policy(limits)?;
    let mut input_bytes = 0;
    let mut transport = Output::new(output, command.max_total_output_bytes);
    let mut ledger = Ledger::default();
    let first = wire::read_line(input, args.max_input_bytes, &mut input_bytes, command.max_total_input_bytes)?;
    session.remaining()?;
    let Some(first) = first else { return complete(&ledger, input_bytes, &encoding,
        &mut transport, &|| session.remaining().map(|_| ())); };
    // Refuse malformed first input before opening even model metadata.
    let first = wire::parse_record(command.task, &first, args.max_input_bytes)?;
    let facts = session.facts(args)?;
    let planner = Planner::new(args, limits, &facts)?;
    let cancellation = CancellationToken::default();
    let mut model = None;
    let mut pending = Some(first);
    let frame_cap = limits.result_bytes.checked_add(8192).ok_or(CandidateError::Output)?;
    while let Some(record) = pending.take() {
        session.remaining()?;
        let (id, prepared) = planner.prepare_with_control(record, &mut session.control())?;
        let work = prepared.planned_work();
        if work.forward_positions > args.context_tokens as u64 { return Err(CandidateError::Planning); }
        ledger.admit(&id, ReservedWork { forward_positions: work.forward_positions,
            projected_logits: work.projected_logits }, command)?;
        // Admit worst-case delivery before any model load or native request.
        transport.admit_frame(frame_cap)?;
        if model.is_none() { model = Some(session.load(args, limits, &facts, cancellation.clone())?); }
        let resident = model.as_ref().ok_or(CandidateError::Model)?;
        let result = execute_chat(&session.engine, resident, prepared, ledger.records,
            session.native(args)?, args, cancellation.clone())?;
        session.remaining()?;
        let response = CandidateResponse { schema_version: 1, scope: "real-artifact-current-candidate",
            evidence: "non_authoritative", model_id: &facts.model_id, source_revision: &facts.revision,
            source_root_sha256: &facts.source_root_sha256, logical_model_sha256: &facts.logical_model_sha256,
            quant_recipe: &facts.recipe_id, output: &result };
        let frame = ResultFrame { protocol: PROTOCOL, schema_version: 1, event: "result",
            id: &id, request_seq: ledger.records, response,
            input_encoding: input_encoding::reported_policy(&encoding) };
        // The hosted result's output guard survives complete write AND flush.
        publish_checked(&frame, frame_cap, &mut transport, &|| session.remaining().map(|_| ()))?;
        drop(result);
        session.remaining()?;
        pending = wire::read_line(input, args.max_input_bytes, &mut input_bytes, command.max_total_input_bytes)?
            .map(|line| wire::parse_record(command.task, &line, args.max_input_bytes)).transpose()?;
    }
    session.remaining()?;
    complete(&ledger, input_bytes, &encoding, &mut transport, &|| session.remaining().map(|_| ()))
}

#[derive(Serialize)]
struct ResultFrame<'a, T: Serialize> {
    protocol: &'static str, schema_version: u32, event: &'static str,
    id: &'a str, request_seq: u64, response: CandidateResponse<'a, T>,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_encoding: Option<&'a UntrustedEncoding>,
}
#[derive(Serialize)]
struct Complete<'a> {
    protocol: &'static str, schema_version: u32, event: &'static str,
    scope: &'static str, evidence: &'static str, records: u64, input_bytes: u64,
    result_frame_bytes: u64, reserved_native_work: ReservedWork,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_encoding: Option<&'a UntrustedEncoding>,
}
fn complete<W: Write>(ledger: &Ledger, input_bytes: u64, encoding: &UntrustedEncoding,
    output: &mut Output<'_, W>, check: &impl Fn() -> Result<(), CandidateError>) -> Result<(), CandidateError> {
    let frame = Complete { protocol: PROTOCOL, schema_version: 1, event: "batch_complete",
        scope: "real-artifact-current-candidate", evidence: "non_authoritative", records: ledger.records,
        input_bytes, result_frame_bytes: output.written, reserved_native_work: ledger.work,
        input_encoding: input_encoding::reported_policy(encoding) };
    publish_checked(&frame, FOOTER_BYTES, output, check)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidate_cli::text_batch::{parse_record, TextTask};
    use crate::tokenizer::{embedded::EmbeddedTokenizer,
        untrusted::{SafeBpeLimits, SafeBpeWork, UntrustedDocumentEncoder}};

    /// Measure the pinned encoder itself, before choosing a tight test budget.
    /// The checkpoint count also identifies a cancellation inside real encoding.
    pub(super) fn measured_encoding(text: &str) -> (SafeBpeWork, usize) {
        let tokenizer = EmbeddedTokenizer::pinned().unwrap(); let controls = pinned_controls::pinned().unwrap();
        let encoder = UntrustedDocumentEncoder::new(tokenizer.tokenizer(), controls.template_controls());
        let policy = UntrustedEncoding::SafeBpeV1(SafeBpeLimits {
            max_input_bytes: SafeBpeLimits::MAX_INPUT_BYTES, max_output_tokens: SafeBpeLimits::MAX_OUTPUT_TOKENS,
            max_pair_checks: SafeBpeLimits::MAX_PAIR_CHECKS, max_pair_bytes: SafeBpeLimits::MAX_PAIR_BYTES });
        let mut work = SafeBpeWork::default(); let mut checkpoints = 0;
        let encoded = encoder.encode_with_policy(text.as_bytes(), &policy, &mut work,
            &mut || { checkpoints += 1; true }).unwrap();
        assert_eq!(tokenizer.tokenizer().decode_bytes(encoded.ids()).unwrap(), text.as_bytes());
        (work, checkpoints)
    }

    fn planner() -> Planner {
        let args = crate::candidate_cli::tests::args(&[]); let limits = args.validate().unwrap();
        let facts = ArtifactIdentity { model_id: "Nanbeige4.2-3B".into(),
            revision: "f56ec5a9650268aa098496734743c25ea778bd2d".into(), recipe_id: "metadata-only-unit-fixture".into(),
            source_root_sha256: "ab".repeat(32), logical_model_sha256: "cd".repeat(32) };
        Planner::new(&args, limits, &facts).unwrap()
    }
    #[test]
    fn item_and_sample_addresses_bind_plans_without_physical_row_numbers() {
        let planner = planner();
        let make = |id: &str, sample: u64| parse_record(TextTask::Generate,
            &format!(r#"{{"id":"{id}","sample_index":{sample},"prompt":"hello"}}"#), 4096).unwrap();
        let (_, first) = planner.prepare(make("a", 0)).unwrap();
        let (_, other) = planner.prepare(make("b", 0)).unwrap();
        let (_, repeated) = planner.prepare(make("a", 0)).unwrap();
        let (_, sample) = planner.prepare(make("a", 1)).unwrap();
        let key = |plan: &PreparedInt8Chat| canonjson::canonical_bytes(plan.execution_identity()).unwrap();
        assert_eq!(key(&first), key(&repeated));
        assert_ne!(key(&first), key(&other)); assert_ne!(key(&first), key(&sample));
    }
    #[test]
    fn completion_is_explicit_and_counts_only_preceding_result_transport() {
        let mut bytes = Vec::new(); let mut output = Output::new(&mut bytes, 8192);
        output.write_all(b"{}\n").unwrap();
        complete(&Ledger::default(), 0, &UntrustedEncoding::ByteFallback, &mut output, &|| Ok(())).unwrap();
        let text = String::from_utf8(bytes).unwrap();
        let footer: serde_json::Value = serde_json::from_str(text.lines().nth(1).unwrap()).unwrap();
        assert_eq!(footer["event"], "batch_complete"); assert_eq!(footer["result_frame_bytes"], 3);
    }

    #[test]
    fn compact_work_renews_between_records_but_not_between_transcript_messages() {
        let text = "The quick brown fox jumps over the lazy dog.";
        let (work, _) = measured_encoding(text);
        assert!(work.pair_checks() > 0 && work.pair_bytes() > 0);
        for (flag, cap) in [("--max-tokenizer-pair-checks", work.pair_checks()),
            ("--max-tokenizer-pair-bytes", work.pair_bytes())] {
            let cap = cap.to_string();
            let args = crate::candidate_cli::tests::args(&["--input-encoding", "safe-bpe-v1", flag, &cap]);
            let facts = ArtifactIdentity { model_id: "Nanbeige4.2-3B".into(),
                revision: "f56ec5a9650268aa098496734743c25ea778bd2d".into(), recipe_id: "metadata-only-unit-fixture".into(),
                source_root_sha256: "ab".repeat(32), logical_model_sha256: "cd".repeat(32) };
            let planner = Planner::new(&args, args.validate().unwrap(), &facts).unwrap();
            let generate = |id: &str| parse_record(TextTask::Generate,
                &serde_json::json!({"id": id, "prompt": text}).to_string(), args.max_input_bytes).unwrap();
            let (_, first) = planner.prepare(generate("a")).unwrap();
            let (_, second) = planner.prepare(generate("b")).unwrap();
            assert_eq!(first.execution_identity().prompt_digest, second.execution_identity().prompt_digest);
            assert_eq!(first.planned_work(), second.planned_work());
            let chat = |messages: serde_json::Value| parse_record(TextTask::Chat,
                &serde_json::json!({"id": "chat", "messages": messages}).to_string(), args.max_input_bytes).unwrap();
            planner.prepare(chat(serde_json::json!([{"role": "user", "content": text}]))).unwrap();
            assert!(matches!(planner.prepare(chat(serde_json::json!([
                {"role": "system", "content": text}, {"role": "user", "content": text}]))),
                Err(CandidateError::Planning)));
        }
    }

    #[test]
    fn result_and_completion_report_encoding_without_changing_default_frames_or_expired_delivery() {
        use std::cell::Cell;
        let facts = ArtifactIdentity { model_id: "Nanbeige4.2-3B".into(),
            revision: "f56ec5a9650268aa098496734743c25ea778bd2d".into(), recipe_id: "framing-only-fixture".into(),
            source_root_sha256: "ab".repeat(32), logical_model_sha256: "cd".repeat(32) };
        let value = serde_json::json!({"framing_fixture": true});
        for extra in [vec![], vec!["--input-encoding", "safe-bpe-v1"]] {
            let args = crate::candidate_cli::tests::args(&extra);
            let policy = args.input_encoding_policy(args.validate().unwrap()).unwrap();
            let response = CandidateResponse { schema_version: 1, scope: "real-artifact-current-candidate",
                evidence: "non_authoritative", model_id: &facts.model_id, source_revision: &facts.revision,
                source_root_sha256: &facts.source_root_sha256, logical_model_sha256: &facts.logical_model_sha256,
                quant_recipe: &facts.recipe_id, output: &value };
            let expected_response = serde_json::to_value(&response).unwrap();
            let frame = ResultFrame { protocol: PROTOCOL, schema_version: 1, event: "result", id: "a",
                request_seq: 1, response, input_encoding: input_encoding::reported_policy(&policy) };
            let mut result_bytes = Vec::new(); publish_checked(&frame, 8192, &mut result_bytes, &|| Ok(())).unwrap();
            let mut expected = serde_json::json!({"protocol": PROTOCOL, "schema_version": 1,
                "event": "result", "id": "a", "request_seq": 1, "response": expected_response});
            if policy != UntrustedEncoding::ByteFallback {
                expected["input_encoding"] = serde_json::to_value(&policy).unwrap();
            }
            assert_eq!(serde_json::from_slice::<serde_json::Value>(&result_bytes).unwrap(), expected);
            let mut completed = Vec::new(); let mut output = Output::new(&mut completed, FOOTER_BYTES as u64);
            complete(&Ledger::default(), 0, &policy, &mut output, &|| Ok(())).unwrap();
            let footer: serde_json::Value = serde_json::from_slice(&completed).unwrap();
            if policy == UntrustedEncoding::ByteFallback {
                assert_eq!(completed, concat!("{\"protocol\":\"fnlp-candidate-text-batch-v1\",\"schema_version\":1,",
                    "\"event\":\"batch_complete\",\"scope\":\"real-artifact-current-candidate\",",
                    "\"evidence\":\"non_authoritative\",\"records\":0,\"input_bytes\":0,",
                    "\"result_frame_bytes\":0,\"reserved_native_work\":{\"forward_positions\":0,\"projected_logits\":0}}\n").as_bytes());
            } else { assert_eq!(footer["input_encoding"], serde_json::to_value(&policy).unwrap()); }

            // First check admits serialization; the second observes expiration
            // after staging and must prevent every externally visible byte.
            let checks = Cell::new(0); let mut refused = Vec::new();
            let mut output = Output::new(&mut refused, FOOTER_BYTES as u64);
            let check = || { checks.set(checks.get() + 1);
                if checks.get() == 2 { Err(CandidateError::Timeout) } else { Ok(()) } };
            assert_eq!(complete(&Ledger::default(), 0, &policy, &mut output, &check), Err(CandidateError::Timeout));
            assert_eq!(checks.get(), 2); assert_eq!(output.written, 0); assert!(refused.is_empty());
        }
    }
}
