//! Actual pinned corpus planning, without loaded weights or native-success fixtures.
use super::*;
use crate::{batch::{BatchDocument, classify::ClassificationBatchArgs},
    candidate_cli::scored_batch::tests::command,
    native_engine::decode::DecodeCancellationKind,
    tasks::sentiment::{SentimentAxis, batch::{Int8SentimentBatchPlanner, SentimentBatchArgs}}};
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn facts() -> ArtifactIdentity {
    ArtifactIdentity { model_id: "Nanbeige4.2-3B".to_owned(), revision: "f56ec5a9650268aa098496734743c25ea778bd2d".to_owned(),
        recipe_id: "metadata-only-test-fixture".to_owned(), source_root_sha256: "ab".repeat(32), logical_model_sha256: "cd".repeat(32) }
}
fn doc<A>() -> BatchDocument<A> { BatchDocument { id: "x".to_owned(), text: "Great café <tool_call>".to_owned(), task_args: None } }
#[test]
fn classification_configuration_and_real_item_plan_keep_the_selected_model() {
    let c = command("classify", &[]); let (_, l, _) = c.validate().unwrap(); let b = c.host.budget(l);
    let defaults = c.parse_defaults(Some(r#"{"labels":[{"id":"happy"},{"id":"sad"}]}"#), b).unwrap();
    let Corpus::Classify(p, cfg) = configure(&c, &facts(), l, defaults).unwrap() else { panic!("wrong task") };
    assert_eq!(cfg.max_model_work, c.host.work_ceiling());
    let compiler = Int8ClassificationBatchPlanner::new(&p, cfg.identity, b, cfg.planning, cfg.defaults).unwrap();
    let plan = compiler.prepare_with_control(doc::<ClassificationBatchArgs>(), &mut Continue).unwrap();
    assert_eq!(plan.head_count(), 1);
    assert_eq!(plan.execution_identity().logical_model_digest.to_hex(), facts().logical_model_sha256);
    assert_eq!(plan.execution_identity().task_spec, "classify-v1");
    assert_ne!(plan.execution_identity().prompt_digest, Sha256Digest::of_bytes(b"null"));
}
#[test]
fn sentiment_item_override_does_not_change_the_next_records_default_axes() {
    let c = command("sentiment", &[]); let (_, l, _) = c.validate().unwrap(); let b = c.host.budget(l);
    let defaults = c.parse_defaults(None, b).unwrap();
    let Corpus::Sentiment(p, cfg) = configure(&c, &facts(), l, defaults).unwrap() else { panic!("wrong task") };
    assert_eq!(cfg.max_model_work, c.host.work_ceiling());
    let compiler = Int8SentimentBatchPlanner::new(&p, cfg).unwrap();
    let mut one = doc(); one.task_args = Some(SentimentBatchArgs { axes: vec![SentimentAxis::Valence], budget: b });
    let a = compiler.prepare_with_control(one, &mut Continue).unwrap();
    let z = compiler.prepare_with_control(doc(), &mut Continue).unwrap();
    assert_eq!(a.head_count(), 1); assert_eq!(z.head_count(), 4);
    assert_ne!(a.execution_identity().taskir_digest, z.execution_identity().taskir_digest);
    assert_eq!(z.execution_identity().logical_model_digest.to_hex(), facts().logical_model_sha256);
    assert_eq!(z.execution_identity().numerics_profile, NumericsProfile::StrictQuantized { version: 1 });
}
#[test]
fn sentiment_policy_is_bound_to_the_run_not_accepted_as_an_item_override() {
    let c = command("sentiment", &[]); let (_, l, _) = c.validate().unwrap(); let b = c.host.budget(l);
    let a = c.parse_defaults(None, b).unwrap();
    let z = c.parse_defaults(Some(r#"{"policy":{"minimum_peak_weight_ppm":900000,"maximum_normalized_entropy_ppm":500000}}"#), b).unwrap();
    let Corpus::Sentiment(_, a) = configure(&c, &facts(), l, a).unwrap() else { panic!("wrong task") };
    let Corpus::Sentiment(_, z) = configure(&c, &facts(), l, z).unwrap() else { panic!("wrong task") };
    assert_ne!(a.identity.template_digest, z.identity.template_digest);
    assert!(serde_json::from_value::<SentimentBatchArgs>(serde_json::json!({
        "axes":["valence"],"budget":b,"policy":{"minimum_peak_weight_ppm":0}
    })).is_err());
}
#[test]
fn wrong_task_settings_or_model_revision_never_become_a_stream_configuration() {
    let c = command("sentiment", &[]); let (_, l, _) = c.validate().unwrap(); let b = c.host.budget(l);
    assert!(configure(&c, &facts(), l, Defaults::Classify(None)).is_err());
    let mut changed = facts(); changed.revision = "other".to_owned();
    assert!(configure(&c, &changed, l, c.parse_defaults(None, b).unwrap()).is_err());
}

#[test]
fn an_item_allowance_cannot_exceed_the_whole_sentiment_stream_allowance() {
    let c = command("sentiment", &[]); let (_, l, _) = c.validate().unwrap(); let b = c.host.budget(l);
    let defaults = c.parse_defaults(None, b).unwrap();
    let Corpus::Sentiment(p, mut cfg) = configure(&c, &facts(), l, defaults).unwrap() else { panic!("wrong task") };
    cfg.max_item_work.attention_pairs = cfg.max_model_work.attention_pairs + 1;
    assert!(cfg.validate(&p).is_err());
    cfg.max_item_work = cfg.max_model_work;
    cfg.max_item_work.projections.multiply_accumulates += 1;
    assert!(cfg.validate(&p).is_err());
}

#[test]
fn compact_classification_stream_uses_explicit_bound_plans_for_each_record() {
    let c = command("classify", &["--input-encoding", "safe-bpe-v1"]);
    let (_, limits, _) = c.validate().unwrap(); let budget = c.host.budget(limits);
    let defaults = c.parse_defaults(Some(r#"{"labels":[{"id":"happy"},{"id":"sad"}],"mode":"multi_label"}"#), budget).unwrap();
    let Corpus::Classify(planner, cfg) = configure(&c, &facts(), limits, defaults).unwrap() else { panic!("wrong task") };
    assert_eq!(planner.document_encoding(), &c.host.encoding_policy(&c.encoding).unwrap());
    assert_eq!(cfg.max_model_work, c.host.work_ceiling());
    assert!(Int8ClassificationBatchPlanner::new(&planner, cfg.identity.clone(), budget,
        cfg.planning, cfg.defaults.clone()).is_err());
    let mut wrong = cfg.identity.clone(); wrong.template_digest = Sha256Digest::of_bytes(b"other policy");
    assert!(Int8ClassificationBatchPlanner::new_with_document_encoding(&planner, wrong, budget,
        cfg.planning, cfg.defaults.clone()).is_err());
    let compiler = Int8ClassificationBatchPlanner::new_with_document_encoding(&planner, cfg.identity,
        budget, cfg.planning, cfg.defaults).unwrap();
    let first = compiler.prepare_with_control(doc::<ClassificationBatchArgs>(), &mut Continue).unwrap();
    let second = compiler.prepare_with_control(doc::<ClassificationBatchArgs>(), &mut Continue).unwrap();
    assert_eq!(first.head_count(), 2);
    assert_eq!(first.execution_identity(), second.execution_identity());
    assert_eq!(first.model_work(), second.model_work());
}

#[test]
fn compact_sentiment_stream_keeps_one_policy_for_every_axis_and_record() {
    let c = command("sentiment", &["--input-encoding", "safe-bpe-v1"]);
    let (_, limits, _) = c.validate().unwrap(); let budget = c.host.budget(limits);
    let Corpus::Sentiment(planner, mut cfg) = configure(&c, &facts(), limits,
        c.parse_defaults(None, budget).unwrap()).unwrap() else { panic!("wrong task") };
    assert_eq!(planner.document_encoding(), &c.host.encoding_policy(&c.encoding).unwrap());
    assert!(cfg.validate(&planner).is_err());
    let digest = cfg.identity.template_digest;
    cfg.identity.template_digest = Sha256Digest::of_bytes(b"other policy");
    assert!(cfg.validate_with_document_encoding(&planner).is_err());
    cfg.identity.template_digest = digest;
    let compiler = Int8SentimentBatchPlanner::new_with_document_encoding(&planner, cfg).unwrap();
    let first = compiler.prepare_with_control(doc::<SentimentBatchArgs>(), &mut Continue).unwrap();
    let second = compiler.prepare_with_control(doc::<SentimentBatchArgs>(), &mut Continue).unwrap();
    assert_eq!(first.head_count(), 4);
    assert_eq!(first.execution_identity(), second.execution_identity());
    assert_eq!(first.model_work(), second.model_work());
}

#[test]
fn exact_compact_policy_is_reported_on_flush_and_empty_terminal_frames_only_when_selected() {
    use std::io::Write;
    for extra in [&[][..], &["--input-encoding", "safe-bpe-v1"][..]] {
        let c = command("sentiment", extra);
        let encoding = c.host.encoding_policy(&c.encoding).unwrap(); let facts = facts();
        let provenance = Provenance { protocol: "fnlp-candidate-batch-v1", schema_version: 1,
            scope: "real-artifact-current-candidate", evidence: "non_authoritative",
            model_id: &facts.model_id, source_revision: &facts.revision, source_root_sha256: &facts.source_root_sha256,
            logical_model_sha256: &facts.logical_model_sha256, quant_recipe: &facts.recipe_id, task: "sentiment",
            input_encoding: input_encoding::reported_policy(&encoding),
            compute_threads: None, prefill_rows: None };
        let mut output = Vec::new();
        {
            let mut writer = CandidateWriter::new(&mut output, &provenance, 1024, 16 * 1024).unwrap();
            // Transport framing only: these fixtures do not invent scored results.
            for kind in ["flush", "run_complete"] {
                writeln!(writer, "{}", serde_json::json!({"kind":kind})).unwrap();
                writer.flush().unwrap();
            }
        }
        let frames = serde_json::Deserializer::from_slice(&output).into_iter::<serde_json::Value>()
            .collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(frames.len(), 2);
        for (frame, kind) in frames.iter().zip(["flush", "run_complete"]) {
            assert_eq!(frame["record"], serde_json::json!({"kind":kind}));
            assert!(frame.get("compute_threads").is_none()); assert!(frame.get("prefill_rows").is_none());
            if extra.is_empty() { assert!(frame.get("input_encoding").is_none()); }
            else { assert_eq!(frame["input_encoding"], serde_json::to_value(&encoding).unwrap()); }
        }
    }
}

#[test]
fn every_physical_schedule_preserves_pinned_classification_and_sentiment_plans() {
    fn snapshot(c: &ScoreBatchCommand) -> (ExecutionIdentity, crate::native_engine::strict_int8::Int8Work, usize) {
        let (_, limits, _) = c.validate().unwrap(); let budget = c.host.budget(limits);
        let text = (c.kind().unwrap() == ScoreTask::Classify)
            .then_some(r#"{"labels":[{"id":"happy"},{"id":"sad"}],"mode":"multi_label"}"#);
        let defaults = c.parse_defaults(text, budget).unwrap();
        match configure(c, &facts(), limits, defaults).unwrap() {
            Corpus::Classify(planner, config) => {
                let compiler = Int8ClassificationBatchPlanner::new_with_document_encoding(&planner, config.identity,
                    budget, config.planning, config.defaults).unwrap();
                let plan = compiler.prepare_with_control(doc(), &mut Continue).unwrap();
                (plan.execution_identity().clone(), plan.model_work(), plan.head_count())
            }
            Corpus::Sentiment(planner, config) => {
                let compiler = Int8SentimentBatchPlanner::new_with_document_encoding(&planner, config).unwrap();
                let plan = compiler.prepare_with_control(doc(), &mut Continue).unwrap();
                (plan.execution_identity().clone(), plan.model_work(), plan.head_count())
            }
            Corpus::Judge(_, _) => panic!("judge has a separate complete-head regression"),
        }
    }
    for task in ["classify", "sentiment"] {
        for encoding in [&[][..], &["--input-encoding", "safe-bpe-v1"][..]] {
            let expected = snapshot(&command(task, encoding));
            for schedule in [&["--prefill-rows", "4"][..], &["--compute-threads", "4"][..],
                &["--compute-threads", "4", "--prefill-rows", "4"][..]] {
                let mut options = encoding.to_vec(); options.extend_from_slice(schedule);
                assert_eq!(snapshot(&command(task, &options)), expected);
            }
        }
    }
}

#[test]
fn stale_team_admission_is_rejected_before_constructing_a_session_or_touching_owned_io() {
    struct Never;
    impl Read for Never { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("input read"); } }
    impl Write for Never {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("output write"); }
        fn flush(&mut self) -> io::Result<()> { panic!("output flush"); }
    }
    for task in ["classify", "sentiment", "judge"] {
        let c = command(task, &["--compute-threads", "4", "--prefill-rows", "4",
            "--defaults", "never-open.json"]);
        let (args, mut limits, envelope) = c.validate().unwrap();
        limits.compute.workspace_bytes -= 1;
        assert_eq!(execute(c, args, limits, envelope, Never, Never), Err(CandidateError::Arguments));
        let c = command(task, &["--compute-threads", "4", "--defaults", "never-open.json"]);
        let (mut args, limits, envelope) = c.validate().unwrap();
        args.compute = crate::candidate_cli::compute::Args::default();
        assert_eq!(execute(c, args, limits, envelope, Never, Never), Err(CandidateError::Arguments));
    }
}

#[test]
fn schedule_provenance_is_optional_and_never_alters_transport_records() {
    for (threads, rows) in [(None, None), (Some(4), None), (None, Some(4)), (Some(4), Some(4))] {
        let facts = facts();
        let provenance = Provenance { protocol: "fnlp-candidate-batch-v1", schema_version: 1,
            scope: "real-artifact-current-candidate", evidence: "non_authoritative",
            model_id: &facts.model_id, source_revision: &facts.revision, source_root_sha256: &facts.source_root_sha256,
            logical_model_sha256: &facts.logical_model_sha256, quant_recipe: &facts.recipe_id, task: "classify",
            input_encoding: None, compute_threads: threads, prefill_rows: rows };
        let mut output = Vec::new();
        let record = serde_json::json!({"kind":"run_complete","succeeded":0,"failed":0});
        {
            let mut writer = CandidateWriter::new(&mut output, &provenance, 1024, 16384).unwrap();
            writeln!(writer, "{record}").unwrap(); writer.flush().unwrap();
        }
        let frame: serde_json::Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(frame["record"], record);
        assert_eq!(frame.get("compute_threads").and_then(serde_json::Value::as_u64), threads.map(|n| n as u64));
        assert_eq!(frame.get("prefill_rows").and_then(serde_json::Value::as_u64), rows.map(|n| n as u64));
        assert!(frame.get("input_encoding").is_none());
    }
}
