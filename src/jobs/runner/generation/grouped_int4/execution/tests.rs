//! UNRUN invocation-ledger arithmetic fixtures. Durable attempt persistence,
//! spool acknowledgement and recovery remain owned by the existing JobRunner.
use super::*;
use crate::native_engine::portable_int8::ProjectionWork;
fn work(n: u64) -> JobWork {
    JobWork::from_int4(Int4Work { forward_positions: n, projected_logits: n, attention_pairs: n,
        projections: ProjectionWork { dot_products: n, multiply_accumulates: n } }, 0)
}
fn limits() -> JobLimits {
    JobLimits { max_items: 2, max_id_bytes: 128, max_input_bytes_per_item: 4096,
        max_snapshot_bytes: 65_536, max_result_bytes: 65_536, max_spool_bytes: 1 << 20,
        max_materialized_bytes: 1 << 20, max_journal_bytes: 1 << 20, max_attempts: 2, max_work: work(2) }
}
fn context(sequence: u64) -> BatchRequestContext {
    BatchRequestContext { request_seq: sequence, epoch: 1, input_line: sequence, byte_offset: 0 }
}
#[test]
fn failed_attempt_stays_charged_and_poisoned() {
    let mut ledger = Ledger::default();
    ledger.begin(work(1), context(7), limits()).unwrap();
    assert_eq!(ledger.work, work(1)); assert_eq!(ledger.attempts, 1);
    assert!(ledger.ready().is_err());
    assert!(ledger.begin(work(1), context(8), limits()).is_err());
    assert_eq!(ledger.work, work(1)); assert_eq!(ledger.attempts, 1);
}
#[test]
fn successful_early_finish_never_refunds_reserved_work_or_attempts() {
    let mut ledger = Ledger::default();
    for sequence in [7, 8] {
        ledger.begin(work(1), context(sequence), limits()).unwrap();
        ledger.failed = false; // Only fully validated native success clears it.
    }
    assert_eq!(ledger.work, work(2)); assert_eq!(ledger.attempts, 2);
    assert_eq!(ledger.begin(work(1), context(9), limits()).err().unwrap().fault.code, BatchCode::WorkLimit);
    assert!(ledger.ready().is_err());
}
#[test]
fn every_work_axis_and_overflow_refuse_before_model_admission() {
    for axis in 0..5 {
        let mut excess = work(1);
        match axis {
            0 => excess.model.forward_positions = 3, 1 => excess.model.projected_logits = 3,
            2 => excess.model.attention_pairs = 3, 3 => excess.model.projections.dot_products = 3,
            _ => excess.model.projections.multiply_accumulates = 3,
        }
        let mut ledger = Ledger::default();
        assert_eq!(ledger.begin(excess, context(1), limits()).err().unwrap().fault.code, BatchCode::WorkLimit);
        assert!(ledger.ready().is_err());
    }
    let mut ledger = Ledger { work: work(u64::MAX), ..Ledger::default() };
    let mut ceiling = limits(); ceiling.max_work = work(u64::MAX);
    assert_eq!(ledger.begin(work(1), context(1), ceiling).err().unwrap().fault.code, BatchCode::WorkLimit);
    assert!(ledger.ready().is_err());
}
#[test]
fn zero_or_repeated_delivery_coordinates_poison_before_admission() {
    for invalid in [context(0), BatchRequestContext { epoch: 0, ..context(1) },
        BatchRequestContext { input_line: 0, ..context(1) }] {
        let mut ledger = Ledger::default();
        assert!(ledger.begin(work(1), invalid, limits()).is_err()); assert!(ledger.ready().is_err());
    }
    let mut ledger = Ledger::default(); ledger.begin(work(1), context(7), limits()).unwrap();
    ledger.failed = false;
    assert!(ledger.begin(work(1), context(7), limits()).is_err()); assert!(ledger.ready().is_err());
}

mod schedules {
    use super::*;
    use crate::{
        jobs::runner::generation::grouped_int4::Int4GenerationJobConfig,
        native_engine::{decode::DecodeCancellationKind,
            generation::{GenerationLimits, GenerationSampling}, lmhead::NANBEIGE_VOCAB_SIZE,
            strict_int4::{Int4MemoryBudget, Int4RunBudget,
                weights::{Int4ModelWeights, projection_fixture}}},
        tokenizer::embedded::EmbeddedTokenizer,
    };
    use std::cell::Cell;

    struct Continue;
    impl DecodeStepControl for Continue {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
    }
    struct Cancel;
    impl DecodeStepControl for Cancel {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            Some(DecodeCancellationKind::Deadline)
        }
    }
    struct Admission<'a>(&'a Cell<usize>);
    impl Int4GenerationBatchAdmission for Admission<'_> {
        type Guard = ();
        fn admit(&mut self, _: Int4GenerationAdmission<'_>)
            -> Result<(ExecutionIdentity, Self::Guard), BatchItemFailure> {
            self.0.set(self.0.get() + 1);
            Err(BatchItemFailure::fatal(BatchCode::Admission))
        }
    }
    fn setup(task: GenerationJobTask) -> (Int4ModelWeights, Int4GenerationJobPlanner, Int4MemoryRequirement) {
        let (weights, _) = projection_fixture(4);
        let memory = Int4MemoryRequirement::for_context(256, 4).unwrap();
        let ceiling = Int4Work::for_sequence(4, 0, 256, 8 * NANBEIGE_VOCAB_SIZE).unwrap();
        let mut job = limits();
        job.max_work = JobWork::from_int4(ceiling.checked_add(ceiling).unwrap(), 0);
        let eos = EmbeddedTokenizer::pinned().unwrap().eos_token_id().unwrap();
        let mut generation = GenerationOptions::greedy(8, 4096, eos);
        generation.sampling = GenerationSampling::Seeded { effective_seed: [7; 32],
            temperature_milli: 1000, top_k: Some(32), top_p_ppm: 900_000 };
        let planner = Int4GenerationJobPlanner::for_model(weights.identity(), Int4GenerationJobConfig {
            task, generation, budget: TaskBudget { max_input_tokens: 240, max_output_tokens: 8,
                max_output_bytes: 65_536, max_grammar_states: 1, max_kv_bytes: memory.kv_bytes },
            planning: ChatLimits { max_messages: 4, max_message_bytes: 128, max_total_message_bytes: 256,
                generation: GenerationLimits { max_prompt_tokens: 240, max_new_tokens: 8,
                    max_output_bytes: 4096, max_sampler_bytes: 32 << 20 } },
            native: Int4GenerationBudget { native: Int4RunBudget::exact(ceiling),
                max_kv_bytes: memory.kv_bytes, max_sampler_bytes: 32 << 20 }, context_tokens: 256,
        }, job).unwrap();
        (weights, planner, memory)
    }
    fn document() -> BatchDocument<GenerationJobArgs> {
        BatchDocument { id: "original-item".to_owned(), text: "Continue the story.".to_owned(),
            task_args: Some(GenerationJobArgs { sample_index: 7, history: Vec::new() }) }
    }
    fn prefill(rows: usize) -> Int4PrefillLimits {
        Int4PrefillLimits { max_batch_rows: rows,
            max_extra_scratch_bytes: Int4PrefillLimits::required_extra_scratch_bytes(rows).unwrap() }
    }

    #[test]
    fn both_schedules_preserve_preparation_and_charge_full_cold_work_before_cancellation() {
        // Tiny weights authenticate construction only. Cancellation precedes
        // admission and every forward; this is not a model-parity fixture.
        for task in [GenerationJobTask::Generate, GenerationJobTask::Chat] {
            let (weights, planner, memory) = setup(task);
            let mut engine = StrictInt4Engine::new(&weights, 256, Int4MemoryBudget {
                max_kv_bytes: memory.kv_bytes, max_rope_bytes: memory.rope_bytes,
                max_scratch_payload_bytes: memory.scratch_payload_bound }).unwrap();
            let recipe = canonjson::canonical_bytes(planner.job_recipe()).unwrap();
            let expected = planner.prepare_with_control(document(), &mut Continue).unwrap();
            let calls = Cell::new(0);
            for schedule in [None, Some(prefill(1)), Some(prefill(4)), Some(prefill(64))] {
                let mut processor = match schedule {
                    None => Int4GenerationJobProcessor::new(&planner, &mut engine, Admission(&calls)),
                    Some(prefill) => Int4GenerationJobProcessor::new_layer_major(
                        &planner, &mut engine, Admission(&calls), prefill),
                }.unwrap();
                let prepared = processor.prepare_with_control(document(), &mut Continue).unwrap();
                assert_eq!(prepared.execution_identity(), expected.execution_identity());
                assert_eq!(processor.durable_work(&prepared).unwrap(), expected.work);
                assert_eq!(canonjson::canonical_bytes(processor.job_recipe()).unwrap(), recipe);
                let error = processor.execute_with_context(prepared, context(7), &mut Cancel).err().unwrap();
                assert_eq!(error.fault, BatchFault::cancelled(DecodeCancellationKind::Deadline));
                assert_eq!(processor.ledger.work, expected.work);
                assert_eq!(processor.ledger.attempts, 1);
                assert!(processor.ledger.ready().is_err());
                assert!(processor.prepare_with_control(document(), &mut Continue).is_err());
                assert_eq!(calls.get(), 0);
                assert!(!processor.engine.is_poisoned());
                assert!(processor.engine.kv_cache().all_slots_have_len(0));
            }
            for invalid in [Int4PrefillLimits { max_batch_rows: 0, max_extra_scratch_bytes: u64::MAX },
                Int4PrefillLimits { max_batch_rows: 65, max_extra_scratch_bytes: u64::MAX },
                Int4PrefillLimits { max_batch_rows: 4, max_extra_scratch_bytes: prefill(4).max_extra_scratch_bytes - 1 }] {
                assert_eq!(Int4GenerationJobProcessor::new_layer_major(&planner, &mut engine,
                    Admission(&calls), invalid).err().unwrap().code, BatchCode::InvalidLimits);
                assert_eq!(calls.get(), 0);
                assert!(!engine.is_poisoned());
                assert!(engine.kv_cache().all_slots_have_len(0));
            }
        }
    }
}
