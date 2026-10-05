//! Hosted text corpus requests with one resident model and one deadline.
//! Serial by default; explicit document cohorts use shared-weight INT8 work.
use super::*;
use super::source_tasks::Session;
use crate::candidate_cli::text_batch::{self as wire, Content, Ledger, Output, Record,
    ReservedWork, TextBatchCommand, FOOTER_BYTES, PROTOCOL};
use crate::tasks::chat::quantized::PreparedInt8Chat;
use std::io::{BufRead, BufReader};
mod cohort;

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
            .map_err(|_| CandidateError::Planning)?;
        Ok(Self { native, options: args.options(eos)?, budget })
    }
    fn prepare(&self, record: Record) -> Result<(String, PreparedInt8Chat), CandidateError> {
        let Record { id, sample_index, content } = record;
        let prepared = match content {
            Content::Generate(prompt) => self.native.plan_generate(&GenerateRequest {
                item_id: id.clone(), sample_index, prompt, generation: self.options.clone(), budget: self.budget }),
            Content::Chat(messages) => self.native.plan_chat(&ChatRequest {
                item_id: id.clone(), sample_index, messages, generation: self.options.clone(), budget: self.budget }),
        }.map_err(|_| CandidateError::Planning)?;
        Ok((id, prepared))
    }
}

fn run(session: &Session, command: &TextBatchCommand, limits: Limits,
    input: &mut impl BufRead, output: &mut impl Write) -> Result<(), CandidateError> {
    if let Some(width) = command.cohort_rows { return cohort::run(session, command, limits, input, output, width); }
    let args = &command.common;
    let mut input_bytes = 0;
    let mut transport = Output::new(output, command.max_total_output_bytes);
    let mut ledger = Ledger::default();
    let first = wire::read_line(input, args.max_input_bytes, &mut input_bytes, command.max_total_input_bytes)?;
    session.remaining()?;
    let Some(first) = first else { return complete(&ledger, input_bytes, &mut transport); };
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
        let (id, prepared) = planner.prepare(record)?;
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
            id: &id, request_seq: ledger.records, response };
        // The hosted result's output guard survives complete write AND flush.
        publish(&frame, frame_cap, &mut transport)?;
        drop(result);
        session.remaining()?;
        pending = wire::read_line(input, args.max_input_bytes, &mut input_bytes, command.max_total_input_bytes)?
            .map(|line| wire::parse_record(command.task, &line, args.max_input_bytes)).transpose()?;
    }
    session.remaining()?;
    complete(&ledger, input_bytes, &mut transport)
}

#[derive(Serialize)]
struct ResultFrame<'a, T: Serialize> {
    protocol: &'static str, schema_version: u32, event: &'static str,
    id: &'a str, request_seq: u64, response: CandidateResponse<'a, T>,
}
#[derive(Serialize)]
struct Complete {
    protocol: &'static str, schema_version: u32, event: &'static str,
    scope: &'static str, evidence: &'static str, records: u64, input_bytes: u64,
    result_frame_bytes: u64, reserved_native_work: ReservedWork,
}
fn complete<W: Write>(ledger: &Ledger, input_bytes: u64, output: &mut Output<'_, W>) -> Result<(), CandidateError> {
    let frame = Complete { protocol: PROTOCOL, schema_version: 1, event: "batch_complete",
        scope: "real-artifact-current-candidate", evidence: "non_authoritative", records: ledger.records,
        input_bytes, result_frame_bytes: output.written, reserved_native_work: ledger.work };
    publish(&frame, FOOTER_BYTES, output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidate_cli::text_batch::{parse_record, TextTask};
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
        complete(&Ledger::default(), 0, &mut output).unwrap();
        let text = String::from_utf8(bytes).unwrap();
        let footer: serde_json::Value = serde_json::from_str(text.lines().nth(1).unwrap()).unwrap();
        assert_eq!(footer["event"], "batch_complete"); assert_eq!(footer["result_frame_bytes"], 3);
    }
}
