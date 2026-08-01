//! Structural exactness gates for the constrained-execution compiler.

use franken_nlp::grammar::{CompileLimits, compile_json_schema};
use franken_nlp::grammar::{
    execution::{
        AdditionalProcessor, DiagnosticValue, EosTransition, ExecutionCompileError,
        ExecutionCompiler, ExecutionPrimitive, ForcedDisableReason, ForcedPath, ForcedRun,
        ForcedRunVisitError, ForcedTokenWitness, ForcedWitnessOutcome, ForcedWitnessRequest,
        FullProjection, KvPointEqualityEvidence, PayloadTokenAlphabet, ProductExecutionState,
        ProjectLegal, ProjectionTelemetry, SourceProductState, SparseProjectionThreshold,
    },
    mask::DenseTokenMask,
};

const VOCAB_SIZE: usize = 8;
const EOS_ID: u32 = 6;
const TEMPLATE_CONTROL_ID: u32 = 7;

fn mask(ids: &[u32]) -> DenseTokenMask {
    let mut mask = DenseTokenMask::empty(VOCAB_SIZE);
    for &token_id in ids {
        mask.set_legal(token_id)
            .expect("synthetic token id must fit the synthetic vocabulary");
    }
    mask
}

fn alphabet() -> PayloadTokenAlphabet {
    PayloadTokenAlphabet::new(
        [TEMPLATE_CONTROL_ID],
        EosTransition::Forbidden { token_id: EOS_ID },
    )
}

fn state(state_id: usize, legal_ids: &[u32]) -> ProductExecutionState {
    ProductExecutionState {
        state_id,
        legal_tokens: mask(legal_ids),
        payload_alphabet: alphabet(),
        source_product: None,
        forced_path: None,
    }
}

fn witness_request(
    grammar: DenseTokenMask,
    eos: DenseTokenMask,
    banned: DenseTokenMask,
    penalties: DenseTokenMask,
    bias: DenseTokenMask,
    phase: DenseTokenMask,
) -> ForcedWitnessRequest {
    ForcedWitnessRequest {
        grammar_tokenizer_legal: grammar,
        eos_min_length_stop_legal: eos,
        banned_token_legal: banned,
        repetition_presence_frequency_legal: penalties,
        logit_bias_legal: bias,
        tool_thinking_phase_legal: phase,
        other_processors: Vec::new(),
        telemetry: ProjectionTelemetry::default(),
        payload_alphabet: alphabet(),
    }
}

fn unique_witness(token_id: u32) -> ForcedTokenWitness {
    let legal = mask(&[token_id]);
    let outcome = ForcedTokenWitness::build(&witness_request(
        legal.clone(),
        legal.clone(),
        legal.clone(),
        legal.clone(),
        legal.clone(),
        legal,
    ))
    .expect("synthetic request is well-formed");
    match outcome {
        ForcedWitnessOutcome::Eligible(witness) => witness,
        ForcedWitnessOutcome::Disabled(reason) => panic!("expected unique witness, got {reason:?}"),
    }
}

fn compiler() -> ExecutionCompiler {
    ExecutionCompiler::new(
        SparseProjectionThreshold::new(3, "synthetic-matrix-v1")
            .expect("threshold identity is nonempty"),
        true,
    )
}

#[test]
fn primitive_selection_matrix_uses_one_exact_path_per_state() {
    let sparse = compiler()
        .compile_state(state(0, &[1, 3]))
        .expect("two legal rows are below the strict sparse threshold");
    assert!(matches!(
        sparse.primitive(),
        ExecutionPrimitive::ProjectLegal(_)
    ));
    assert_eq!(sparse.log().primitive, "ProjectLegal");

    let full = compiler()
        .compile_state(state(1, &[1, 3, 4]))
        .expect("threshold equality stays on universal full projection");
    assert!(matches!(
        full.primitive(),
        ExecutionPrimitive::FullProjection(_)
    ));
    assert_eq!(full.log().primitive, "FullProjection");

    let witness = unique_witness(3);
    let forced = compiler()
        .compile_state(ProductExecutionState {
            forced_path: Some(ForcedPath::Proven(
                ForcedRun::new(vec![3], vec![witness], None)
                    .expect("one exact token and one witness form a forced run"),
            )),
            ..state(2, &[3])
        })
        .expect("single forced token is a valid primitive");
    assert!(matches!(
        forced.primitive(),
        ExecutionPrimitive::FeedForced { .. }
    ));
    assert_eq!(forced.log().primitive, "FeedForced");

    let copy = compiler()
        .compile_state(ProductExecutionState {
            source_product: Some(SourceProductState {
                source_state_id: 91,
            }),
            ..state(3, &[1])
        })
        .expect("source product state has a dedicated primitive");
    assert!(matches!(
        copy.primitive(),
        ExecutionPrimitive::CopyFromSource(SourceProductState {
            source_state_id: 91
        })
    ));
    assert_eq!(copy.log().primitive, "CopyFromSource");

    let mismatched_witness = compiler()
        .compile_state(ProductExecutionState {
            forced_path: Some(ForcedPath::Proven(
                ForcedRun::new(vec![3], vec![unique_witness(3)], None)
                    .expect("the proposed token has a complete one-token witness"),
            )),
            ..state(4, &[2, 3])
        })
        .expect("a stale witness falls back rather than becoming a forced token");
    assert!(matches!(
        mismatched_witness.primitive(),
        ExecutionPrimitive::ProjectLegal(_)
    ));
    assert_eq!(
        mismatched_witness.log().fallback_reason,
        Some(ForcedDisableReason::WitnessDoesNotMatchUniversalMask)
    );
}

#[test]
fn sparse_projection_equals_masked_full_projection_and_never_invents_audit_fields() {
    let mut seed = 0x6a09_e667_f3bc_c909_u64;
    for case in 0..96 {
        let mut legal_ids = Vec::new();
        for token_id in 0..u32::try_from(VOCAB_SIZE).expect("tiny vocabulary fits u32") {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            if (seed >> 61) & 1 == 1 {
                legal_ids.push(token_id);
            }
        }
        if legal_ids.is_empty() {
            legal_ids.push(0);
        }
        let legal = mask(&legal_ids);
        let mut logits = Vec::with_capacity(VOCAB_SIZE);
        for _ in 0..VOCAB_SIZE {
            seed = seed
                .wrapping_mul(1_442_695_040_888_963_407)
                .wrapping_add(17);
            let centered =
                i32::try_from((seed >> 32) & 0xffff).expect("bounded value fits i32") - 32_768;
            logits.push(centered as f32 / 257.0);
        }
        let full_projection = FullProjection::new(legal.clone(), true);
        let sparse_projection = ProjectLegal::from_mask(&legal);
        assert_eq!(
            sparse_projection
                .legal_row_logits(&logits)
                .expect("sparse path evaluates every legal row"),
            full_projection
                .legal_row_logits(&logits)
                .expect("full scan preserves each legal row"),
            "sparse legal-row values diverged from full mask in case {case}"
        );
        let full = full_projection
            .select(&logits)
            .expect("nonempty dense mask selects one row");
        let sparse = sparse_projection
            .select(&logits)
            .expect("complete sparse rows select one row");
        assert_eq!(
            sparse.token_id, full.token_id,
            "sparse selection diverged from full mask in case {case}"
        );
        assert_eq!(sparse.logit, full.logit);
        assert!(matches!(
            sparse.audit.pre_mask_argmax,
            DiagnosticValue::NotComputed
        ));
        assert!(matches!(
            sparse.audit.legal_probability_mass,
            DiagnosticValue::NotComputed
        ));
    }
}

#[test]
fn forced_witness_requires_every_processor_and_disables_on_any_unknown_semantics() {
    let both = mask(&[2, 3]);
    for changed_processor in 0..6 {
        let only_three = mask(&[3]);
        let masks: [DenseTokenMask; 6] = std::array::from_fn(|index| {
            if index == changed_processor {
                only_three.clone()
            } else {
                both.clone()
            }
        });
        let request = witness_request(
            masks[0].clone(),
            masks[1].clone(),
            masks[2].clone(),
            masks[3].clone(),
            masks[4].clone(),
            masks[5].clone(),
        );
        assert!(matches!(
            ForcedTokenWitness::build(&request),
            Ok(ForcedWitnessOutcome::Eligible(_))
        ));

        let all_unconstrained = witness_request(
            both.clone(),
            both.clone(),
            both.clone(),
            both.clone(),
            both.clone(),
            both.clone(),
        );
        assert!(matches!(
            ForcedTokenWitness::build(&all_unconstrained),
            Ok(ForcedWitnessOutcome::Disabled(
                ForcedDisableReason::NotExactlyOneLegalToken {
                    legal_token_count: 2
                }
            ))
        ));
    }

    let mut telemetry = witness_request(
        mask(&[3]),
        mask(&[3]),
        mask(&[3]),
        mask(&[3]),
        mask(&[3]),
        mask(&[3]),
    );
    telemetry.telemetry.logprobs = true;
    assert!(matches!(
        ForcedTokenWitness::build(&telemetry),
        Ok(ForcedWitnessOutcome::Disabled(
            ForcedDisableReason::ProjectionTelemetryRequested
        ))
    ));
    let telemetry_projection = compiler()
        .compile_state(ProductExecutionState {
            forced_path: Some(ForcedPath::Disabled(
                ForcedDisableReason::ProjectionTelemetryRequested,
            )),
            ..state(10, &[3])
        })
        .expect("requested logprobs retain a projection path");
    assert!(matches!(
        telemetry_projection.primitive(),
        ExecutionPrimitive::FullProjection(_)
    ));

    let mut unsupported = witness_request(
        mask(&[3]),
        mask(&[3]),
        mask(&[3]),
        mask(&[3]),
        mask(&[3]),
        mask(&[3]),
    );
    unsupported
        .other_processors
        .push(AdditionalProcessor::Unsupported {
            name: "unmodeled-processor".to_owned(),
        });
    assert!(matches!(
        ForcedTokenWitness::build(&unsupported),
        Ok(ForcedWitnessOutcome::Disabled(
            ForcedDisableReason::UnsupportedProcessor { .. }
        ))
    ));

    let disabled = compiler()
        .compile_state(ProductExecutionState {
            forced_path: Some(ForcedPath::Disabled(
                ForcedDisableReason::UnsupportedProcessor {
                    name: "unmodeled-processor".to_owned(),
                },
            )),
            ..state(11, &[2, 3])
        })
        .expect("unsupported forcing falls back to an exact projection path");
    assert!(matches!(
        disabled.primitive(),
        ExecutionPrimitive::ProjectLegal(_)
    ));
    assert!(matches!(
        disabled.log().fallback_reason,
        Some(ForcedDisableReason::UnsupportedProcessor { .. })
    ));
}

#[test]
fn token_alphabet_and_eos_rules_fail_closed_before_path_selection() {
    let control_error = compiler()
        .compile_state(state(0, &[TEMPLATE_CONTROL_ID]))
        .expect_err("template controls may not be payload tokens");
    assert!(matches!(
        control_error,
        ExecutionCompileError::TemplateControlRepresentable { .. }
    ));

    let forbidden_eos = compiler()
        .compile_state(state(1, &[EOS_ID]))
        .expect_err("EOS needs an explicit accepting transition");
    assert!(matches!(
        forbidden_eos,
        ExecutionCompileError::ForbiddenEosLegal { .. }
    ));

    let missing_explicit_eos = compiler()
        .compile_state(ProductExecutionState {
            payload_alphabet: PayloadTokenAlphabet::new(
                [TEMPLATE_CONTROL_ID],
                EosTransition::ExplicitAccepting { token_id: EOS_ID },
            ),
            ..state(2, &[3])
        })
        .expect_err("an accepting state must retain its explicit EOS transition");
    assert!(matches!(
        missing_explicit_eos,
        ExecutionCompileError::MissingExplicitEosTransition { .. }
    ));
}

#[test]
fn forced_runs_default_to_sequential_and_micro_prefill_keeps_its_receipt() {
    let first = unique_witness(3);
    let second = unique_witness(3);
    let sequential = compiler()
        .compile_state(ProductExecutionState {
            forced_path: Some(ForcedPath::Proven(
                ForcedRun::new(vec![3, 3], vec![first.clone(), second.clone()], None)
                    .expect("two exact tokens retain two witnesses"),
            )),
            ..state(0, &[3])
        })
        .expect("unverified micro-prefill falls back to sequential forced feeding");
    match sequential.primitive() {
        ExecutionPrimitive::FeedForced { strategy, .. } => {
            assert!(matches!(
                strategy,
                franken_nlp::grammar::execution::ForcedFeedStrategy::Sequential
            ));
            assert_eq!(
                sequential.log().fallback_reason,
                Some(ForcedDisableReason::MicroPrefillNotVerified)
            );
        }
        other => panic!("expected FeedForced, got {other:?}"),
    }

    let evidence = KvPointEqualityEvidence::exact_44_slot(
        "hf-bf16-eager",
        "kv-44-slot-receipt",
        2,
        KvPointEqualityEvidence::REQUIRED_KV_SLOT_COUNT,
        true,
    )
    .expect("receipt records exact equality across all required KV slots");
    let micro_prefill = compiler()
        .compile_state(ProductExecutionState {
            forced_path: Some(ForcedPath::Proven(
                ForcedRun::new(vec![3, 3], vec![first, second], Some(evidence.clone()))
                    .expect("validated evidence enables the optional strategy"),
            )),
            ..state(1, &[3])
        })
        .expect("micro-prefill remains an exact primitive only with evidence");
    match micro_prefill.primitive() {
        ExecutionPrimitive::FeedForced { strategy, .. } => assert!(matches!(
            strategy,
            franken_nlp::grammar::execution::ForcedFeedStrategy::MicroPrefill {
                evidence: observed
            } if observed == &evidence
        )),
        other => panic!("expected FeedForced, got {other:?}"),
    }
}

#[test]
fn micro_prefill_evidence_refuses_partial_kv_coverage_or_nonexact_rows() {
    assert!(matches!(
        KvPointEqualityEvidence::exact_44_slot("hf-bf16-eager", "partial", 1, 43, true),
        Err(ExecutionCompileError::IncompleteKvEqualityEvidence {
            compared_kv_slot_count: 43,
            every_kv_point_byte_exact: true
        })
    ));
    assert!(matches!(
        KvPointEqualityEvidence::exact_44_slot(
            "hf-bf16-eager",
            "nonexact",
            1,
            KvPointEqualityEvidence::REQUIRED_KV_SLOT_COUNT,
            false,
        ),
        Err(ExecutionCompileError::IncompleteKvEqualityEvidence {
            compared_kv_slot_count: 44,
            every_kv_point_byte_exact: false
        })
    ));
}

#[test]
fn sequential_forced_runs_checkpoint_before_long_run_tokens_and_never_report_partial_success() {
    let witness = unique_witness(3);
    let run = ForcedRun::new(
        vec![3, 3, 3, 3, 3],
        vec![
            witness.clone(),
            witness.clone(),
            witness.clone(),
            witness.clone(),
            witness,
        ],
        None,
    )
    .expect("every exact token has a matching witness");
    let mut fed = Vec::new();
    let cancelled = run
        .visit_sequentially(
            2,
            |progress| progress.next_token_index < 4,
            |token_id| fed.push(token_id),
        )
        .expect_err("checkpoint cancellation must refuse partial success");
    assert_eq!(fed, vec![3, 3, 3, 3]);
    assert_eq!(
        cancelled,
        ForcedRunVisitError::CancelledBeforeToken {
            next_token_index: 4
        }
    );
    assert!(matches!(
        run.visit_sequentially(0, |_| true, |_| {}),
        Err(ForcedRunVisitError::InvalidCheckpointInterval)
    ));
}

#[test]
fn forced_run_construction_requires_a_nonzero_explicit_bound() {
    let witness = unique_witness(3);
    let too_long =
        ForcedRun::new_bounded(vec![3, 3], vec![witness.clone(), witness.clone()], None, 1)
            .expect_err("teacher-fed runs must remain bounded");
    assert_eq!(
        too_long,
        ExecutionCompileError::ForcedRunExceedsLimit {
            token_count: 2,
            max_tokens: 1
        }
    );
    assert!(matches!(
        ForcedRun::new_bounded(vec![3], vec![witness], None, 0),
        Err(ExecutionCompileError::ZeroForcedRunLimit)
    ));
}

#[test]
fn schema_plan_covers_every_compiler_state_once() {
    let schema = compile_json_schema(
        r#"{"type":"string","maxLength":4}"#,
        CompileLimits::default(),
    )
    .expect("tiny synthetic schema compiles");
    let states = (0..schema.automaton().state_count())
        .map(|state_id| state(state_id, &[1]))
        .collect::<Vec<_>>();
    let plan = compiler()
        .compile_schema(&schema, states)
        .expect("all logical grammar states receive exactly one primitive");
    assert_eq!(plan.states().count(), schema.automaton().state_count());
    assert!(plan.states().all(|(_, compiled)| {
        matches!(compiled.primitive(), ExecutionPrimitive::ProjectLegal(_))
    }));
}
