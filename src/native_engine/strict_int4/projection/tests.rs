use super::*;
use crate::native_engine::{decode::DecodeCancellationKind,
    portable_int8::{LinearError, ProjectionWork},
    strict_int4::{Int4MemoryBudget, Int4MemoryRequirement, Int4RunBudget, Int4Work,
        StrictInt4Engine, weights::projection_fixture}};

struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}

// A serial, two-range proof adapter, not a worker team or performance claim.
// The hosted tests exercise the persistent real scoped workers separately.
struct PartitionExecutor<'model> {
    model: &'model Int4ModelWeights, scratch: Vec<i8>, calls: usize,
    stop: Option<DecodeCancellationKind>, stats: Int4ProjectionStats,
}
impl<'model> PartitionExecutor<'model> {
    fn new(model: &'model Int4ModelWeights) -> Self {
        Self { model, scratch: vec![0; model.group_width()], calls: 0, stop: None,
            stats: Int4ProjectionStats { team_width: 1, ..Default::default() } }
    }
    fn entered(&mut self) -> Result<(), StrictInt4Error> {
        self.calls += 1;
        match self.stop { Some(cause) => Err(StrictInt4Error::Cancelled(cause)), None => Ok(()) }
    }
    fn completed(&mut self, rows: usize) {
        self.stats.completed_projections += 1; self.stats.completed_rows += rows as u64;
    }
}
impl Int4ProjectionExecutor for PartitionExecutor<'_> {
    fn model_identity(&self) -> &Int4ModelIdentity { self.model.identity() }
    fn kernel_kind(&self) -> Int8KernelKind { Int8KernelKind::Scalar }
    fn stats(&self) -> Int4ProjectionStats { self.stats }
    fn project_bf16_into(&mut self, op: Int4ProjectionOp, input: &ActivationBuffer,
        rows: LinearRows<'_>, output: &mut [Bf16], control: &mut dyn DecodeStepControl)
        -> Result<(), StrictInt4Error> {
        self.entered()?;
        let matrix = self.model.projection(op)?; let count = rows.checked_count(matrix.rows())?;
        for range in [0..count / 2, count / 2..count] {
            matrix.project_bf16_range_into(input.values()?, input.scale()?, rows, range.clone(),
                &mut output[range], &mut self.scratch, control)?;
        }
        self.completed(count); Ok(())
    }
    fn project_f32_into(&mut self, op: Int4ProjectionOp, input: &ActivationBuffer,
        rows: LinearRows<'_>, output: &mut [f32], control: &mut dyn DecodeStepControl)
        -> Result<(), StrictInt4Error> {
        self.entered()?;
        let matrix = self.model.projection(op)?; let count = rows.checked_count(matrix.rows())?;
        for range in [0..count / 2, count / 2..count] {
            matrix.project_f32_range_into(input.values()?, input.scale()?, rows, range.clone(),
                &mut output[range], &mut self.scratch, control)?;
        }
        self.completed(count); Ok(())
    }
}

#[test]
fn injected_ranges_match_serial_rails_and_charge_each_complete_projection_once() {
    let (model, input) = projection_fixture(4);
    for rows in [LinearRows::All, LinearRows::Selected(&[1, 6, 9])] {
        let matrix = &model.head; let count = rows.checked_count(matrix.rows()).unwrap();
        let work = matrix.as_linear().planned_work(rows).unwrap();
        let mut expected = vec![0.0_f32; count]; let mut expected_bf16 = vec![Bf16::from_bits(0); count];
        let mut serial = ProjectionLedger::new(work.checked_add(work).unwrap());
        project_f32(&mut None, Int4ProjectionOp::Head, matrix, &input, rows, &mut expected,
            &mut [0; 4], PackedInt4Kernel::scalar(), &mut serial, &mut Continue).unwrap();
        project_bf16(&mut None, Int4ProjectionOp::Head, matrix, &input, rows, &mut expected_bf16,
            &mut [0; 4], PackedInt4Kernel::scalar(), &mut serial, &mut Continue).unwrap();
        let mut executor = PartitionExecutor::new(&model);
        let mut output = vec![0.0_f32; count]; let mut bf16 = vec![Bf16::from_bits(0); count];
        let mut ledger = ProjectionLedger::new(work.checked_add(work).unwrap());
        {
            let mut injected: Option<&mut dyn Int4ProjectionExecutor> = Some(&mut executor);
            project_f32(&mut injected, Int4ProjectionOp::Head, matrix, &input, rows, &mut output,
                &mut [], PackedInt4Kernel::scalar(), &mut ledger, &mut Continue).unwrap();
            project_bf16(&mut injected, Int4ProjectionOp::Head, matrix, &input, rows, &mut bf16,
                &mut [], PackedInt4Kernel::scalar(), &mut ledger, &mut Continue).unwrap();
        }
        assert_eq!(output.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        assert_eq!(bf16, expected_bf16); assert_eq!(ledger.reserved(), serial.reserved());
        assert_eq!(ledger.remaining(), ProjectionWork::default());
        assert_eq!(executor.calls, 2);
        assert_eq!(executor.stats(), Int4ProjectionStats { team_width: 1, completed_projections: 2,
            completed_rows: (count * 2) as u64, completed_worker_rows: 0 });
    }
}

#[test]
fn malformed_or_unaffordable_projection_never_enters_the_executor() {
    let (model, input) = projection_fixture(4);
    let rows = LinearRows::Selected(&[1, 6, 9]); let matrix = &model.head;
    let work = matrix.as_linear().planned_work(rows).unwrap();
    for mode in 0..4 {
        let mut budget = work;
        if mode == 1 { budget.dot_products -= 1; }
        if mode == 2 { budget.multiply_accumulates -= 1; }
        let mut ledger = ProjectionLedger::new(budget);
        let mut executor = PartitionExecutor::new(&model); let mut output = [91.0; 3];
        let invalid = ActivationBuffer::try_new(6).unwrap();
        {
            let mut injected: Option<&mut dyn Int4ProjectionExecutor> = Some(&mut executor);
            let result = project_f32(&mut injected, Int4ProjectionOp::Head, matrix,
                if mode == 3 { &invalid } else { &input }, rows,
                &mut output[..if mode == 0 { 2 } else { 3 }], &mut [], PackedInt4Kernel::scalar(),
                &mut ledger, &mut Continue);
            assert!(result.is_err());
        }
        assert_eq!(executor.calls, 0); assert_eq!(output, [91.0; 3]);
        assert_eq!(ledger.remaining(), budget); assert_eq!(ledger.reserved(), ProjectionWork::default());
    }
}

#[test]
fn failed_injected_execution_does_not_refund_any_precharged_work() {
    let (model, input) = projection_fixture(4);
    let rows = LinearRows::All; let matrix = &model.head;
    let work = matrix.as_linear().planned_work(rows).unwrap();
    let mut ledger = ProjectionLedger::new(work);
    let mut executor = PartitionExecutor::new(&model); executor.stop = Some(DecodeCancellationKind::Deadline);
    let mut output = [91.0; 10];
    {
        let mut injected: Option<&mut dyn Int4ProjectionExecutor> = Some(&mut executor);
        assert_eq!(project_f32(&mut injected, Int4ProjectionOp::Head, matrix, &input, rows,
            &mut output, &mut [], PackedInt4Kernel::scalar(), &mut ledger, &mut Continue),
            Err(StrictInt4Error::Cancelled(DecodeCancellationKind::Deadline)));
    }
    assert_eq!(executor.calls, 1); assert_eq!(executor.stats().completed_projections, 0);
    assert_eq!(output, [91.0; 10]); assert_eq!(ledger.reserved(), work);
    assert_eq!(ledger.remaining(), ProjectionWork::default());
}

#[test]
fn every_physical_projection_role_resolves_against_the_exact_shared_model() {
    let (model, input) = projection_fixture(4);
    for index in [0, 1, 21] {
        let layer = &model.layers[index];
        for (kind, expected) in [(Int4ProjectionKind::Query, &layer.q), (Int4ProjectionKind::Key, &layer.k),
            (Int4ProjectionKind::Value, &layer.v), (Int4ProjectionKind::AttentionOutput, &layer.o),
            (Int4ProjectionKind::Gate, &layer.gate), (Int4ProjectionKind::Up, &layer.up),
            (Int4ProjectionKind::Down, &layer.down)] {
            let actual = model.projection(Int4ProjectionOp::Layer { layer: index, kind }).unwrap();
            let mut serial = [0.0_f32; 6]; let mut resolved = [0.0_f32; 6];
            expected.as_linear().project_f32_into(&input, LinearRows::All, &mut serial, &mut [0; 4],
                &mut ProjectionLedger::new(expected.as_linear().planned_work(LinearRows::All).unwrap()),
                &mut Continue).unwrap();
            actual.project_f32_range_into(input.values().unwrap(), input.scale().unwrap(), LinearRows::All,
                0..6, &mut resolved, &mut [0; 4], &mut Continue).unwrap();
            assert_eq!(serial.map(f32::to_bits), resolved.map(f32::to_bits));
        }
    }
    for layer in [22, usize::MAX] {
        assert!(matches!(model.projection(Int4ProjectionOp::Layer { layer, kind: Int4ProjectionKind::Query }),
            Err(StrictInt4Error::ProjectionExecutor)));
    }
}

#[test]
fn executor_binding_refuses_another_model_or_kernel_before_session_open() {
    let (model, _) = projection_fixture(4); let (other, _) = projection_fixture(3);
    let mut wrong = PartitionExecutor::new(&other);
    let memory = Int4MemoryRequirement::for_context(1, 4).unwrap();
    let budget = Int4MemoryBudget { max_kv_bytes: memory.kv_bytes, max_rope_bytes: memory.rope_bytes,
        max_scratch_payload_bytes: memory.scratch_payload_bound };
    assert!(StrictInt4Engine::new(&model, 1, budget).unwrap().with_projection_executor(&mut wrong).is_err());
    let mut matching = PartitionExecutor::new(&model);
    assert_eq!(check_binding(model.identity(), Int8KernelKind::Avx2Low7HighBit, &matching),
        Err(StrictInt4Error::ProjectionExecutor));
    let mut engine = StrictInt4Engine::new(&model, 1, budget).unwrap()
        .with_projection_executor(&mut matching).unwrap();
    {
        let mut control = Continue;
        let session = engine.session(Int4RunBudget::exact(Int4Work::default()), &mut control).unwrap();
        assert_eq!(session.position().unwrap(), 0);
    }
    assert!(!engine.is_poisoned());
    // A later builder choice cannot leave the executor and reported kernel
    // disagreeing. This branch needs no particular optional ISA to exist.
    for kind in Int8KernelKind::ALL {
        if kind == Int8KernelKind::Scalar { continue; }
        if let Ok(kernel) = PackedInt4Kernel::detect(kind) {
            engine = engine.with_kernel(kernel);
            assert!(matches!(engine.session(Int4RunBudget::exact(Int4Work::default()), &mut Continue),
                Err(StrictInt4Error::ProjectionExecutor)));
            break;
        }
    }
}

#[test]
fn shared_workspace_quote_is_bounded_and_excludes_the_ordinary_serial_route() {
    assert_eq!(int4_team_workspace_bytes(2, 4).unwrap(), 911_112);
    assert_eq!(int4_team_workspace_bytes(64, MAX_MODEL_K).unwrap(), 3_324_568);
    for width in [0, 1, MAX_INT4_TEAM_WIDTH + 1, usize::MAX] {
        assert_eq!(int4_team_workspace_bytes(width, 4), Err(StrictInt4Error::Memory));
    }
    for group in [0, MAX_MODEL_K + 1, usize::MAX] {
        assert_eq!(int4_team_workspace_bytes(2, group), Err(StrictInt4Error::Memory));
    }
}

#[test]
fn central_debit_refuses_exhaustion_instead_of_restarting_a_worker_allowance() {
    let (model, input) = projection_fixture(4); let rows = LinearRows::Selected(&[1, 6, 9]);
    let matrix = &model.head; let work = matrix.as_linear().planned_work(rows).unwrap();
    let mut ledger = ProjectionLedger::new(work); let mut executor = PartitionExecutor::new(&model);
    {
        let mut injected: Option<&mut dyn Int4ProjectionExecutor> = Some(&mut executor);
        project_f32(&mut injected, Int4ProjectionOp::Head, matrix, &input, rows, &mut [0.0; 3],
            &mut [], PackedInt4Kernel::scalar(), &mut ledger, &mut Continue).unwrap();
        assert!(matches!(project_f32(&mut injected, Int4ProjectionOp::Head, matrix, &input, rows,
            &mut [0.0; 3], &mut [], PackedInt4Kernel::scalar(), &mut ledger, &mut Continue),
            Err(StrictInt4Error::Linear(super::super::GroupedInt4Error::Linear(LinearError::WorkBudget)))));
    }
    assert_eq!(executor.calls, 1); assert_eq!(ledger.reserved(), work);
}

/// Controller-selected model-present DSR gate: exercises the actual decoder's
/// two passes, every physical matrix role and incremental head routing. The
/// serial partition adapter proves arithmetic/routing only; actual worker
/// lifecycle tests belong to hosted dispatch. Neither is a BF16 quality gate.
#[test]
#[ignore = "requires admitted pinned full source closure and multi-GB model memory"]
fn model_present_decoder_and_head_match_through_the_injected_executor() {
    use crate::artifact::converter::{ConversionSourceManifest, prepare_nanbeige42_input};
    use crate::native_engine::{kv::KV_SLOT_COUNT, lmhead::NANBEIGE_VOCAB_SIZE,
        strict_int4::{Int4ModelLoadBudget, Int4ModelLoadRequirement}};
    let root = std::env::var_os("FNLP_INT4_TEST_SOURCE_DIR").expect("explicit pinned source directory required");
    let manifest = std::env::var_os("FNLP_INT4_TEST_SOURCE_MANIFEST").expect("explicit pinned manifest path required");
    let source = prepare_nanbeige42_input(std::path::PathBuf::from(root),
        ConversionSourceManifest::load_pinned(std::path::PathBuf::from(manifest)).unwrap(), false, 64 * 1024).unwrap();
    let need = Int4ModelLoadRequirement::for_group_width(128).unwrap();
    let model = Int4ModelWeights::from_prepared(&source, 128, Int4ModelLoadBudget {
        max_payload_bytes: need.peak_payload_bytes().unwrap(), max_source_values: need.source_values,
        max_groups: need.groups, max_shard_revalidation_bytes: need.shard_revalidation_bytes }, &mut Continue).unwrap();
    let memory = Int4MemoryRequirement::for_context(2, 128).unwrap();
    let budget = Int4MemoryBudget { max_kv_bytes: memory.kv_bytes, max_rope_bytes: memory.rope_bytes,
        max_scratch_payload_bytes: memory.scratch_payload_bound };
    let rows = LinearRows::Selected(&[0, 42, (NANBEIGE_VOCAB_SIZE - 1) as u32]);
    let work = Int4Work::for_sequence(128, 0, 2, 6).unwrap();
    let expected = {
        let mut engine = StrictInt4Engine::new(&model, 2, budget).unwrap();
        let mut control = Continue;
        let output;
        {
            let mut session = engine.session(Int4RunBudget::exact(work), &mut control).unwrap();
            let first = session.prefill(&[1], rows).unwrap();
            session.append(2).unwrap(); let second = session.logits(rows).unwrap();
            assert_eq!(session.work(), work); output = (first, second);
        }
        assert!(engine.kv_cache().all_slots_have_len(0)); assert!(!engine.is_poisoned()); output
    };
    let mut executor = PartitionExecutor::new(&model);
    let actual = {
        let mut engine = StrictInt4Engine::new(&model, 2, budget).unwrap()
            .with_projection_executor(&mut executor).unwrap();
        let mut control = Continue;
        let output;
        {
            let mut session = engine.session(Int4RunBudget::exact(work), &mut control).unwrap();
            let first = session.prefill(&[1], rows).unwrap();
            session.append(2).unwrap(); let second = session.logits(rows).unwrap();
            assert_eq!(session.work(), work); output = (first, second);
        }
        assert!(engine.kv_cache().all_slots_have_len(0)); assert!(!engine.is_poisoned()); output
    };
    let bits = |values: &[f32]| values.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&actual.0), bits(&expected.0)); assert_eq!(bits(&actual.1), bits(&expected.1));
    let stats = executor.stats();
    assert_eq!(stats.completed_projections, (2 * KV_SLOT_COUNT * 7 + 2) as u64);
    let decoder_rows: u64 = super::super::PROJECTION_SHAPES.iter().map(|&(rows, _)| rows as u64).sum();
    assert_eq!(stats.completed_rows, 2 * KV_SLOT_COUNT as u64 * decoder_rows + 6);
    assert_eq!(stats.completed_worker_rows, 0);
}
