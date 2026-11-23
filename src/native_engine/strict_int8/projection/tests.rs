//! Model-free projection/admission regression definitions. The private adapter
//! partitions serially; actual persistent workers are tested at hosted dispatch.
use super::*;
use super::fixture::ProjectionFixture;
use crate::native_engine::{decode::DecodeCancellationKind, portable_int8::{LinearError, ProjectionWork},
    simd::Int8Kernel, strict_int8::{Int8MemoryBudget, Int8MemoryRequirement, Int8RunBudget,
        Int8Work, StrictInt8Engine, prefill::Int8PrefillLimits}};

struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}
struct PartitionExecutor<'view, 'weights> {
    model: &'view Int8WeightView<'weights>, calls: usize, stop: Option<DecodeCancellationKind>,
    stats: Int8ProjectionStats,
}
impl<'view, 'weights> PartitionExecutor<'view, 'weights> {
    fn new(model: &'view Int8WeightView<'weights>) -> Self {
        Self { model, calls: 0, stop: None, stats: Int8ProjectionStats { team_width: 1, ..Default::default() } }
    }
    fn completed(&mut self, rows: usize) {
        self.stats.completed_projections += 1; self.stats.completed_rows += rows as u64;
    }
}
impl Int8ProjectionExecutor for PartitionExecutor<'_, '_> {
    fn artifact_identity(&self) -> &ArtifactIdentity { self.model.artifact_identity() }
    fn kernel_kind(&self) -> Int8KernelKind { self.model.kernel_kind() }
    fn stats(&self) -> Int8ProjectionStats { self.stats }
    fn project_bf16_into(&mut self, op: Int8ProjectionOp, input: &ActivationBuffer,
        rows: LinearRows<'_>, output: &mut [Bf16], control: &mut dyn DecodeStepControl)
        -> Result<(), StrictInt8Error> {
        self.calls += 1;
        let matrix = self.model.projection(op)?; let count = rows.checked_count(matrix.rows())?;
        for (index, range) in [0..count / 2, count / 2..count].into_iter().enumerate() {
            matrix.project_bf16_range_into(input.values()?, input.scale()?, rows, range.clone(), &mut output[range], control)?;
            if index == 0 { if let Some(cause) = self.stop { return Err(StrictInt8Error::Cancelled(cause)); } }
        }
        self.completed(count); Ok(())
    }
    fn project_f32_into(&mut self, op: Int8ProjectionOp, input: &ActivationBuffer,
        rows: LinearRows<'_>, output: &mut [f32], control: &mut dyn DecodeStepControl)
        -> Result<(), StrictInt8Error> {
        self.calls += 1;
        let matrix = self.model.projection(op)?; let count = rows.checked_count(matrix.rows())?;
        for (index, range) in [0..count / 2, count / 2..count].into_iter().enumerate() {
            matrix.project_f32_range_into(input.values()?, input.scale()?, rows, range.clone(), &mut output[range], control)?;
            if index == 0 { if let Some(cause) = self.stop { return Err(StrictInt8Error::Cancelled(cause)); } }
        }
        self.completed(count); Ok(())
    }
}
fn memory() -> Int8MemoryBudget {
    let required = Int8MemoryRequirement::for_context(1).unwrap();
    Int8MemoryBudget { max_kv_bytes: required.kv_bytes, max_rope_bytes: required.rope_bytes,
        max_scratch_payload_bytes: required.scratch_payload_bound }
}

#[test]
fn optional_executor_preserves_public_engine_and_immutable_view_auto_traits() {
    fn require_send_sync<T: Send + Sync>() {}
    require_send_sync::<StrictInt8Engine<'static>>();
    require_send_sync::<Int8WeightView<'static>>();
}

#[test]
fn every_physical_projection_role_resolves_its_original_immutable_matrix() {
    let fixture = ProjectionFixture::new(0); let weights = fixture.weights(); let input = fixture.input();
    let mut pairs = vec![(Int8ProjectionOp::Head, weights.head)];
    for (index, layer) in weights.layers.iter().enumerate() {
        for (kind, matrix) in [(Int8ProjectionKind::Query, layer.q), (Int8ProjectionKind::Key, layer.k),
            (Int8ProjectionKind::Value, layer.v), (Int8ProjectionKind::AttentionOutput, layer.o),
            (Int8ProjectionKind::Gate, layer.gate), (Int8ProjectionKind::Up, layer.up),
            (Int8ProjectionKind::Down, layer.down)] {
            pairs.push((Int8ProjectionOp::Layer { layer: index, kind }, matrix));
        }
    }
    for (op, original) in pairs {
        let resolved = weights.projection(op).unwrap();
        assert_eq!((resolved.rows(), resolved.columns()), (original.rows(), original.columns()));
        let work = ProjectionWork::for_shape(original.rows(), original.columns()).unwrap();
        let mut expected = vec![0.0_f32; original.rows()]; let mut actual = expected.clone();
        original.project_f32_into(&input, LinearRows::All, &mut expected, &mut ProjectionLedger::new(work), &mut Continue).unwrap();
        resolved.project_f32_into(&input, LinearRows::All, &mut actual, &mut ProjectionLedger::new(work), &mut Continue).unwrap();
        assert_eq!(actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
    }
    assert!(matches!(weights.projection(Int8ProjectionOp::Layer {
        layer: crate::native_engine::kv::PHYSICAL_LAYER_COUNT, kind: Int8ProjectionKind::Query }),
        Err(StrictInt8Error::ProjectionExecutor)));
}

#[test]
fn installed_ranges_preserve_full_and_selected_bits_with_one_complete_debit() {
    let fixture = ProjectionFixture::new(0); let weights = fixture.weights(); let input = fixture.input();
    for rows in [LinearRows::All, LinearRows::Selected(&[1, 6, 9])] {
        let matrix = weights.head; let count = rows.checked_count(matrix.rows()).unwrap();
        let one = ProjectionWork::for_shape(count, matrix.columns()).unwrap(); let work = one.checked_add(one).unwrap();
        let mut expected = vec![0.0_f32; count]; let mut expected_bf16 = vec![Bf16::default(); count];
        let mut serial = ProjectionLedger::new(work);
        project_f32(&mut None, Int8ProjectionOp::Head, matrix, &input, rows, &mut expected, &mut serial, &mut Continue).unwrap();
        project_bf16(&mut None, Int8ProjectionOp::Head, matrix, &input, rows, &mut expected_bf16, &mut serial, &mut Continue).unwrap();
        let mut actual = vec![0.0_f32; count]; let mut bf16 = vec![Bf16::default(); count];
        let mut ledger = ProjectionLedger::new(work); let mut executor = PartitionExecutor::new(&weights);
        {
            let mut injected: Option<&mut dyn Int8ProjectionExecutor> = Some(&mut executor);
            project_f32(&mut injected, Int8ProjectionOp::Head, matrix, &input, rows, &mut actual, &mut ledger, &mut Continue).unwrap();
            project_bf16(&mut injected, Int8ProjectionOp::Head, matrix, &input, rows, &mut bf16, &mut ledger, &mut Continue).unwrap();
            assert!(matches!(project_f32(&mut injected, Int8ProjectionOp::Head, matrix, &input, rows,
                &mut actual, &mut ledger, &mut Continue), Err(StrictInt8Error::Linear(LinearError::WorkBudget))));
        }
        assert_eq!(actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        assert_eq!(bf16, expected_bf16); assert_eq!(ledger.reserved(), serial.reserved());
        assert_eq!(ledger.remaining(), ProjectionWork::default()); assert_eq!(executor.calls, 2);
        assert_eq!(executor.stats(), Int8ProjectionStats { team_width: 1, completed_projections: 2,
            completed_rows: (count * 2) as u64, completed_worker_rows: 0 });
    }
}

#[test]
fn malformed_or_unaffordable_projection_never_enters_executor_or_spends_work() {
    let fixture = ProjectionFixture::new(0); let weights = fixture.weights(); let input = fixture.input();
    let matrix = weights.head; let rows = LinearRows::Selected(&[1, 6, 9]);
    let work = ProjectionWork::for_shape(3, matrix.columns()).unwrap();
    for mode in 0..5 {
        let mut cap = work;
        if mode == 1 { cap.dot_products -= 1; }
        if mode == 2 { cap.multiply_accumulates -= 1; }
        let invalid = ActivationBuffer::try_new(6).unwrap();
        let mut ledger = ProjectionLedger::new(cap); let mut executor = PartitionExecutor::new(&weights);
        let mut output = [91.0; 3];
        {
            let mut injected: Option<&mut dyn Int8ProjectionExecutor> = Some(&mut executor);
            assert!(project_f32(&mut injected, Int8ProjectionOp::Head, matrix,
                if mode == 3 { &invalid } else { &input },
                if mode == 4 { LinearRows::Selected(&[1, 1, 9]) } else { rows },
                &mut output[..if mode == 0 { 2 } else { 3 }], &mut ledger, &mut Continue).is_err());
        }
        assert_eq!(executor.calls, 0); assert_eq!(output, [91.0; 3]);
        assert_eq!(ledger.remaining(), cap); assert_eq!(ledger.reserved(), ProjectionWork::default());
    }
}

#[test]
fn partial_failure_retains_all_work_and_poisons_the_owning_native_session() {
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let fixture = ProjectionFixture::new(0); let weights = fixture.weights();
        let mut executor = PartitionExecutor::new(&weights); executor.stop = Some(cause);
        let mut engine = StrictInt8Engine::new(fixture.weights(), 1, memory()).unwrap()
            .with_projection_executor(&mut executor).unwrap();
        {
            let mut control = Continue;
            let cap = Int8Work::for_sequence(0, 0, 2).unwrap();
            let mut session = engine.session(Int8RunBudget::exact(cap), &mut control).unwrap();
            // Private head-only fixture: do not pretend the tiny model can
            // execute a real decoder. The public session still owns failure.
            session.hidden = Some(vec![Bf16::from_f32(1.0); 6]);
            assert_eq!(session.logits(LinearRows::Selected(&[0, 3])).unwrap_err(), StrictInt8Error::Cancelled(cause));
            assert_eq!(session.work().projections, ProjectionWork::for_shape(2, 6).unwrap());
            assert_eq!(session.work().projected_logits, 0);
        }
        assert!(engine.is_poisoned()); assert!(engine.kv_cache().all_slots_have_len(0));
        assert!(matches!(engine.session(Int8RunBudget::exact(Int8Work::default()), &mut Continue),
            Err(StrictInt8Error::EngineUnavailable)));
        drop(engine); assert_eq!(executor.calls, 1); assert_eq!(executor.stats().completed_projections, 0);
    }
}

#[test]
fn wrong_model_or_kernel_is_refused_before_admission_and_after_builder_chaining() {
    let fixture = ProjectionFixture::new(0); let other = ProjectionFixture::new(1);
    let weights = fixture.weights(); let foreign = other.weights(); let mut wrong = PartitionExecutor::new(&foreign);
    assert!(matches!(StrictInt8Engine::new(fixture.weights(), 1, memory()).unwrap().with_projection_executor(&mut wrong),
        Err(StrictInt8Error::ProjectionExecutor)));
    let mut matching = PartitionExecutor::new(&weights);
    assert_eq!(check_binding(weights.artifact_identity(), Int8KernelKind::Avx2Low7HighBit, &matching),
        Err(StrictInt8Error::ProjectionExecutor));
    let mut engine = StrictInt8Engine::new(fixture.weights(), 1, memory()).unwrap()
        .with_projection_executor(&mut matching).unwrap();
    {
        let mut control = Continue;
        let session = engine.session(Int8RunBudget::exact(Int8Work::default()), &mut control).unwrap();
        assert_eq!(session.position().unwrap(), 0);
    }
    for kind in Int8KernelKind::ALL {
        if kind == Int8KernelKind::Scalar { continue; }
        if let Ok(kernel) = Int8Kernel::detect(kind) {
            engine = engine.with_kernel(kernel);
            assert!(matches!(engine.session(Int8RunBudget::exact(Int8Work::default()), &mut Continue),
                Err(StrictInt8Error::ProjectionExecutor)));
            break;
        }
    }
}

#[test]
fn layer_major_cannot_bypass_an_installed_single_row_executor() {
    let fixture = ProjectionFixture::new(0); let weights = fixture.weights();
    let mut executor = PartitionExecutor::new(&weights);
    let mut engine = StrictInt8Engine::new(fixture.weights(), 1, memory()).unwrap()
        .with_projection_executor(&mut executor).unwrap();
    {
        let mut control = Continue;
        let mut session = engine.session(Int8RunBudget::exact(Int8Work::default()), &mut control).unwrap();
        for rows in [1, 2, 64] {
            assert_eq!(session.append_layer_major(&[1], Int8PrefillLimits { max_batch_rows: rows,
                max_extra_scratch_bytes: Int8PrefillLimits::required_extra_scratch_bytes(rows).unwrap() }),
                Err(StrictInt8Error::ProjectionExecutor));
            assert_eq!(session.work(), Int8Work::default()); assert_eq!(session.position().unwrap(), 0);
        }
    }
    assert!(!engine.is_poisoned()); assert!(engine.kv_cache().all_slots_have_len(0));
    drop(engine); assert_eq!(executor.calls, 0);
}

#[test]
fn one_pure_workspace_quote_covers_fixed_workers_without_changing_serial_defaults() {
    assert_eq!(int8_team_workspace_bytes(2).unwrap(), 911_104);
    assert_eq!(int8_team_workspace_bytes(64).unwrap(), 2_636_440);
    for width in [0, 1, MAX_INT8_TEAM_WIDTH + 1, usize::MAX] {
        assert_eq!(int8_team_workspace_bytes(width), Err(StrictInt8Error::Memory));
    }
}
