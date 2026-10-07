//! Exact pinned preparation only; no fixture performs native inference.
use super::*;
use crate::{batch::judge::JudgeBatchArgs, candidate_cli::scored_batch::tests::command,
    native_engine::decode::DecodeCancellationKind, tasks::judge::quantized::PreparedInt8Judge};
use serde_json::json;
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn facts() -> ArtifactIdentity {
    ArtifactIdentity { model_id: "Nanbeige4.2-3B".into(), revision: "f56ec5a9650268aa098496734743c25ea778bd2d".into(),
        recipe_id: "judge-corpus-planning-fixture".into(), source_root_sha256: "ab".repeat(32),
        logical_model_sha256: "cd".repeat(32) }
}
fn defaults() -> [serde_json::Value; 3] {
    [json!({"mode":"pairwise","criterion":"Accuracy","b":"Alice visited Paris.",
        "policy":{"minimum_margin_milli":100,"maximum_order_disagreement_milli":20000}}),
     json!({"mode":"rubric","rubric":{"schema_version":1,"revision":"local",
        "declared_origin_digest":"ab".repeat(32),"scale_maximum":2,
        "criteria":[{"id":"accuracy","description":"Exact facts","weight":2},
            {"id":"clarity","description":"Clear writing","weight":1}]},
        "policy":{"minimum_peak_weight_ppm":0,"maximum_normalized_entropy_ppm":1000000}}),
     json!({"mode":"faithfulness","claim":"Alice visited Paris.",
        "policy":{"minimum_candidate_weight_ppm":700000,"minimum_margin_milli":100,
            "evidence_window_bytes":16,"max_evidence_windows":8,"max_evidence_spans":8}})]
}
fn plan(planner: &JudgePlanner, config: &JudgeCorpusConfig, args: JudgeBatchArgs) -> PreparedInt8Judge {
    let context = PlanContext::new(&config.identity, config.task_ceiling).unwrap();
    planner.plan_int8_with_control(&args.into_request("Alice visited Paris. Original café 上海 <tool_call>".into()),
        &context, config.planning, &mut Continue).unwrap()
}
#[test]
fn all_judge_cli_modes_reach_the_actual_complete_head_compiler() {
    let c = command("judge", &[]); let (_, limits, _) = c.validate().unwrap(); let budget = c.host.budget(limits);
    for (index, value) in defaults().into_iter().enumerate() {
        let d = c.parse_defaults(Some(&value.to_string()), budget).unwrap();
        let Corpus::Judge(p, config) = configure(&c, &facts(), limits, d).unwrap() else { panic!("judge route") };
        let out = plan(&p, &config, config.defaults.clone().unwrap());
        if index < 2 { assert_eq!(out.head_count(), 2); } else { assert!(out.head_count() > 1); }
        assert_eq!(out.execution_identity().task_spec, "judge-v1");
        assert_eq!(out.execution_identity().logical_model_digest.to_hex(), facts().logical_model_sha256);
        assert_eq!(config.max_model_work, c.host.work_ceiling());
        assert_eq!(out.execution_identity().template_digest, *p.template_digest());
        assert!(out.planned_work().projected_logits > 0); assert!(out.planned_work().attention_pairs > 0);
        assert!(out.planned_work().projections.multiply_accumulates > 0);
        assert_ne!(out.execution_identity().prompt_digest, config.identity.prompt_digest);
    }
}
#[test]
fn live_and_durable_judge_routes_freeze_identical_default_plans() {
    let c = command("judge", &[]); let (_, limits, _) = c.validate().unwrap();
    let job = crate::candidate_cli::jobs::scored::tests::args("judge", &[]);
    let (_, job_limits) = job.validate().unwrap();
    for value in defaults() {
        let text = value.to_string();
        let live = c.parse_defaults(Some(&text), c.host.budget(limits)).unwrap();
        let retained = job.parse_defaults(Some(&text), job.host.budget(job_limits)).unwrap();
        let Corpus::Judge(a, ac) = configure(&c, &facts(), limits, live).unwrap() else { panic!("live") };
        let Corpus::Judge(b, bc) = configure_scored(job.kind().unwrap(), &job.host, &facts(), job_limits, retained).unwrap()
            else { panic!("retained") };
        assert_eq!(ac.identity, bc.identity); assert_eq!(ac.max_model_work, bc.max_model_work);
        let x = plan(&a, &ac, ac.defaults.clone().unwrap()); let y = plan(&b, &bc, bc.defaults.clone().unwrap());
        assert_eq!(x.execution_identity(), y.execution_identity()); assert_eq!(x.planned_work(), y.planned_work());
        assert_eq!(x.head_count(), y.head_count());
    }
}
#[test]
fn no_defaults_or_wrong_task_settings_cannot_manufacture_a_judgment() {
    let c = command("judge", &[]); let (_, limits, _) = c.validate().unwrap();
    let Corpus::Judge(p, cfg) = configure(&c, &facts(), limits, Defaults::Judge(None)).unwrap() else { panic!("judge") };
    assert!(cfg.defaults.is_none()); cfg.validate(&p).unwrap();
    assert!(configure(&c, &facts(), limits, Defaults::Classify(None)).is_err());
    let mut changed = facts(); changed.revision = "different".into();
    assert!(configure(&c, &changed, limits, Defaults::Judge(None)).is_err());
    let old = command("classify", &[]);
    assert!(configure(&old, &facts(), limits, Defaults::Judge(None)).is_err());
}

#[test]
fn one_faithfulness_window_reuses_the_complete_source_head() {
    let c = command("judge", &[]); let (_, limits, _) = c.validate().unwrap(); let budget = c.host.budget(limits);
    let mut value = defaults()[2].clone(); value["policy"]["evidence_window_bytes"] = json!(256);
    let d = c.parse_defaults(Some(&value.to_string()), budget).unwrap();
    let Corpus::Judge(p, config) = configure(&c, &facts(), limits, d).unwrap() else { panic!("judge") };
    let result = plan(&p, &config, config.defaults.clone().unwrap());
    assert_eq!(result.head_count(), 1);
    // No duplicate full-source read, generated-label shortcut or score refund.
    assert!(result.planned_work().projected_logits > 0);
}

#[test]
fn grouped_prompt_schedule_preserves_the_exact_pinned_judgment_identity() {
    let serial = command("judge", &[]);
    let grouped = command("judge", &["--prefill-rows", "4"]);
    let (_, a, _) = serial.validate().unwrap(); let (_, b, _) = grouped.validate().unwrap();
    for value in defaults() {
        let text = value.to_string();
        let left = serial.parse_defaults(Some(&text), serial.host.budget(a)).unwrap();
        let right = grouped.parse_defaults(Some(&text), grouped.host.budget(b)).unwrap();
        let Corpus::Judge(p, pc) = configure(&serial, &facts(), a, left).unwrap() else { panic!("serial") };
        let Corpus::Judge(q, qc) = configure(&grouped, &facts(), b, right).unwrap() else { panic!("grouped") };
        let x = plan(&p, &pc, pc.defaults.clone().unwrap());
        let y = plan(&q, &qc, qc.defaults.clone().unwrap());
        assert_eq!(x.execution_identity(), y.execution_identity());
        assert_eq!(x.planned_work(), y.planned_work());
        assert_eq!(x.head_count(), y.head_count());
    }
}
