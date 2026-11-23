//! Model-free batch admission and routing definitions. This adapter executes
//! partitions sequentially; hosted tests own real persistent worker lifecycle.
use super::*;
use super::fixture::ProjectionFixture;
use crate::native_engine::{decode::DecodeCancellationKind, portable_int8::{LinearError, ProjectionWork},
    simd::Int8Kernel, strict_int8::{Int8MemoryBudget, Int8MemoryRequirement, Int8RunBudget,
        Int8Work, StrictInt8Engine, prefill::Int8PrefillLimits}};

struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}
struct BatchExecutor<'view, 'weights> {
    weights: &'view Int8WeightView<'weights>, capacity: usize, batch_calls: usize,
    stop: Option<DecodeCancellationKind>, stats: Int8ProjectionStats,
}
impl<'view, 'weights> BatchExecutor<'view, 'weights> {
    fn new(weights: &'view Int8WeightView<'weights>, capacity: usize) -> Self {
        Self { weights, capacity, batch_calls: 0, stop: None,
            stats: Int8ProjectionStats { team_width: 2, ..Default::default() } }
    }
    fn completed(&mut self, rows: usize) {
        self.stats.completed_projections += 1; self.stats.completed_rows += rows as u64;
    }
}
impl Int8ProjectionExecutor for BatchExecutor<'_, '_> {
    fn artifact_identity(&self) -> &ArtifactIdentity { self.weights.artifact_identity() }
    fn kernel_kind(&self) -> Int8KernelKind { self.weights.kernel_kind() }
    fn max_batch_rows(&self) -> usize { self.capacity }
    fn stats(&self) -> Int8ProjectionStats { self.stats }
    fn project_bf16_into(&mut self, op: Int8ProjectionOp, input: &ActivationBuffer,
        rows: LinearRows<'_>, output: &mut [Bf16], control: &mut dyn DecodeStepControl) -> Result<(), StrictInt8Error> {
        let matrix = self.weights.projection(op)?; let count = rows.checked_count(matrix.rows())?;
        matrix.project_bf16_range_into(input.values()?, input.scale()?, rows, 0..count, output, control)?;
        self.completed(count); Ok(())
    }
    fn project_f32_into(&mut self, op: Int8ProjectionOp, input: &ActivationBuffer,
        rows: LinearRows<'_>, output: &mut [f32], control: &mut dyn DecodeStepControl) -> Result<(), StrictInt8Error> {
        let matrix = self.weights.projection(op)?; let count = rows.checked_count(matrix.rows())?;
        matrix.project_f32_range_into(input.values()?, input.scale()?, rows, 0..count, output, control)?;
        self.completed(count); Ok(())
    }
    fn project_batch_bf16_into(&mut self, op: Int8ProjectionOp, inputs: &[ActivationBuffer],
        rows: LinearRows<'_>, output: &mut [Bf16], control: &mut dyn DecodeStepControl) -> Result<(), StrictInt8Error> {
        self.batch_calls += 1;
        let matrix = self.weights.projection(op)?; let count = rows.checked_count(matrix.rows())?;
        let raw: Vec<_> = inputs.iter().map(ActivationBuffer::values).collect::<Result<_, _>>()?;
        let scales: Vec<_> = inputs.iter().map(ActivationBuffer::scale).collect::<Result<_, _>>()?;
        for (index, range) in [0..count / 2, count / 2..count].into_iter().enumerate() {
            let span = if range.is_empty() { 0 } else { (inputs.len() - 1) * count + range.len() };
            matrix.project_batch_bf16_range_into(&raw, &scales, rows, range.clone(),
                &mut output[range.start..range.start + span], count, control)?;
            if index == 0 { if let Some(cause) = self.stop { return Err(StrictInt8Error::Cancelled(cause)); } }
        }
        self.completed(inputs.len() * count); Ok(())
    }
}
fn inputs(rows: usize) -> Vec<ActivationBuffer> {
    (0..rows).map(|index| {
        let mut input = ActivationBuffer::try_new(6).unwrap();
        input.encode_f32(&[index as f32, -127.0, 3.0, 0.0, -2.5, 55.0]).unwrap(); input
    }).collect()
}
fn op() -> Int8ProjectionOp { Int8ProjectionOp::Layer { layer: 0, kind: Int8ProjectionKind::Query } }
fn memory(context: usize) -> Int8MemoryBudget {
    let required = Int8MemoryRequirement::for_context(context).unwrap();
    Int8MemoryBudget { max_kv_bytes: required.kv_bytes, max_rope_bytes: required.rope_bytes,
        max_scratch_payload_bytes: required.scratch_payload_bound }
}
#[test]
fn every_layer_role_keeps_batch_bits_and_one_whole_projection_debit() {
    let fixture = ProjectionFixture::new(0); let weights = fixture.weights();
    for layer in 0..crate::native_engine::kv::PHYSICAL_LAYER_COUNT {
        for kind in [Int8ProjectionKind::Query, Int8ProjectionKind::Key, Int8ProjectionKind::Value,
            Int8ProjectionKind::AttentionOutput, Int8ProjectionKind::Gate, Int8ProjectionKind::Up, Int8ProjectionKind::Down] {
            let op = Int8ProjectionOp::Layer { layer, kind }; let matrix = weights.projection(op).unwrap();
            for count in [1, 3, 5] { for rows in [LinearRows::All, LinearRows::Selected(&[0, 3, 5])] {
                let inputs = inputs(count); let n = rows.checked_count(matrix.rows()).unwrap();
                let work = ProjectionWork::for_shape(count * n, matrix.columns()).unwrap();
                let mut expected = vec![Bf16::default(); count * n]; let mut actual = expected.clone();
                let mut serial = ProjectionLedger::new(work); let mut ledger = ProjectionLedger::new(work);
                project_batch_bf16(&mut None, op, matrix, &inputs, rows, &mut expected, &mut serial, &mut Continue).unwrap();
                let mut executor = BatchExecutor::new(&weights, count);
                {
                    let mut installed: Option<&mut dyn Int8ProjectionExecutor> = Some(&mut executor);
                    project_batch_bf16(&mut installed, op, matrix, &inputs, rows, &mut actual, &mut ledger, &mut Continue).unwrap();
                    assert_eq!(project_batch_bf16(&mut installed, op, matrix, &inputs, rows, &mut actual,
                        &mut ledger, &mut Continue), Err(StrictInt8Error::Linear(LinearError::WorkBudget)));
                }
                assert_eq!(actual, expected); assert_eq!(ledger.reserved(), serial.reserved());
                assert_eq!(ledger.reserved(), work); assert_eq!(executor.batch_calls, 1);
                assert_eq!(executor.stats.completed_rows, (count * n) as u64);
            } }
        }
    }
}
#[test]
fn late_invalid_activation_short_work_and_unsupported_shape_never_dispatch() {
    let fixture = ProjectionFixture::new(0); let weights = fixture.weights(); let matrix = weights.projection(op()).unwrap();
    let work = ProjectionWork::for_shape(18, 6).unwrap();
    for mode in 0..7 {
        let mut inputs = inputs(3); let mut cap = work; let mut length = 18; let mut capacity = 3;
        let mut op = op(); let mut matrix = matrix;
        match mode {
            0 => { assert!(inputs[2].encode_f32(&[f32::NAN; 6]).is_err()); },
            1 => cap.dot_products -= 1,
            2 => cap.multiply_accumulates -= 1,
            3 => length -= 1,
            4 => capacity -= 1,
            5 => { op = Int8ProjectionOp::Head; matrix = weights.head; length = 30; },
            _ => inputs[2] = ActivationBuffer::try_new(5).unwrap(),
        }
        let mut ledger = ProjectionLedger::new(cap); let mut executor = BatchExecutor::new(&weights, capacity);
        let sentinel = Bf16::from_f32(91.0); let mut output = vec![sentinel; length];
        {
            let mut installed: Option<&mut dyn Int8ProjectionExecutor> = Some(&mut executor);
            assert!(project_batch_bf16(&mut installed, op, matrix, &inputs, LinearRows::All,
                &mut output, &mut ledger, &mut Continue).is_err(), "mode {mode}");
        }
        assert_eq!(executor.batch_calls, 0); assert_eq!(ledger.reserved(), ProjectionWork::default());
        assert!(output.iter().all(|value| *value == sentinel));
    }
}
#[test]
fn every_stop_cause_keeps_complete_debit_with_no_successful_batch_rows() {
    use DecodeCancellationKind::*;
    let fixture = ProjectionFixture::new(0); let weights = fixture.weights(); let inputs = inputs(5);
    let matrix = weights.projection(op()).unwrap(); let work = ProjectionWork::for_shape(30, 6).unwrap();
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let mut executor = BatchExecutor::new(&weights, 5); executor.stop = Some(cause);
        let mut ledger = ProjectionLedger::new(work); let mut output = vec![Bf16::default(); 30];
        {
            let mut installed: Option<&mut dyn Int8ProjectionExecutor> = Some(&mut executor);
            assert_eq!(project_batch_bf16(&mut installed, op(), matrix, &inputs, LinearRows::All,
                &mut output, &mut ledger, &mut Continue), Err(StrictInt8Error::Cancelled(cause)));
        }
        assert_eq!(ledger.reserved(), work); assert_eq!(executor.batch_calls, 1);
        assert_eq!(executor.stats.completed_projections, 0); assert_eq!(executor.stats.completed_rows, 0);
    }
}
#[test]
fn too_small_team_cohort_refuses_entire_prompt_before_native_work_or_poison() {
    let fixture = ProjectionFixture::new(0); let weights = fixture.weights();
    let mut executor = BatchExecutor::new(&weights, 1);
    let mut engine = StrictInt8Engine::new(fixture.weights(), 2, memory(2)).unwrap()
        .with_projection_executor(&mut executor).unwrap();
    {
        let mut control = Continue; let mut session = engine.session(Int8RunBudget::exact(Int8Work::default()), &mut control).unwrap();
        assert_eq!(session.append_layer_major(&[1, 2], Int8PrefillLimits { max_batch_rows: 2,
            max_extra_scratch_bytes: Int8PrefillLimits::required_extra_scratch_bytes(2).unwrap() }),
            Err(StrictInt8Error::ProjectionExecutor));
        assert_eq!(session.work(), Int8Work::default()); assert_eq!(session.position().unwrap(), 0);
    }
    assert!(!engine.is_poisoned()); assert!(engine.kv_cache().all_slots_have_len(0));
    drop(engine); assert_eq!(executor.batch_calls, 0);
}
#[test]
fn prefill_workspace_covers_widest_layer_inputs_and_preserves_the_one_row_quote() {
    for width in [2, 3, 4, 8, 16, 64] {
        assert_eq!(int8_team_workspace_bytes_with_prefill(width, 1).unwrap(), int8_team_workspace_bytes(width).unwrap());
        let head = NANBEIGE_VOCAB_SIZE.div_ceil(width);
        let layer = NANBEIGE_INTERMEDIATE_SIZE.div_ceil(width);
        let mut previous = 0;
        for rows in [1, 3, 4, 5, 16, 64] {
            let payload = head * (std::mem::size_of::<u32>() + std::mem::size_of::<f32>())
                + head.max(rows * layer) * std::mem::size_of::<Bf16>() + rows * MAX_MODEL_K;
            let quote = int8_team_workspace_bytes_with_prefill(width, rows).unwrap();
            assert_eq!(quote, (payload as u64 + 4096) * (width - 1) as u64 + 65_536);
            assert!(quote >= previous); previous = quote;
        }
    }
    for rows in [0, 65, usize::MAX] {
        assert_eq!(int8_team_workspace_bytes_with_prefill(2, rows), Err(StrictInt8Error::Memory));
    }
}
#[test]
#[ignore = "requires explicit current-candidate INT8 artifact and multi-GB admission; no team throughput award"]
fn model_present_prompt_morsels_then_decode_match_serial_logits_and_all_44_kv_slots() {
    use crate::native_engine::{artifact_bridge::ArtifactLoadBudget, kv::KV_SLOT_COUNT,
        strict_int8::CurrentCandidateInt8Model};
    let path = std::env::var_os("FNLP_INT8_TEST_ARTIFACT").expect("explicit INT8 candidate artifact required");
    let model = CurrentCandidateInt8Model::load(std::path::PathBuf::from(path),
        ArtifactLoadBudget { max_weight_bytes: 8 * 1024 * 1024 * 1024,
            max_stream_chunk_bytes: 64 * 1024, max_resident_envelope_bytes: 0 }, |_| {}).unwrap();
    let rows = LinearRows::Selected(&[0, 42, (NANBEIGE_VOCAB_SIZE - 1) as u32]);
    let work = Int8Work::for_sequence(0, 6, 6).unwrap();
    let prefill = Int8PrefillLimits { max_batch_rows: 3,
        max_extra_scratch_bytes: Int8PrefillLimits::required_extra_scratch_bytes(3).unwrap() };
    let run = |engine: &mut StrictInt8Engine<'_>, grouped: bool| {
        let mut control = Continue; let output;
        {
            let mut session = engine.session(Int8RunBudget::exact(work), &mut control).unwrap();
            let first = if grouped { session.prefill_layer_major(&[1, 2, 3, 4, 5], rows, prefill).unwrap() }
                else { session.prefill(&[1, 2, 3, 4, 5], rows).unwrap() };
            session.append(6).unwrap(); let next = session.logits(rows).unwrap();
            assert_eq!(session.work(), work);
            let mut kv = Vec::new();
            for slot in 0..KV_SLOT_COUNT { for position in 0..6 {
                kv.extend_from_slice(session.engine.kv_cache().key_at(slot, position).unwrap());
                kv.extend_from_slice(session.engine.kv_cache().value_at(slot, position).unwrap());
            } }
            output = (first.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                next.iter().map(|value| value.to_bits()).collect::<Vec<_>>(), kv);
        }
        assert!(!engine.is_poisoned()); assert!(engine.kv_cache().all_slots_have_len(0)); output
    };
    let weights = model.weight_view().unwrap().with_kernel(Int8Kernel::scalar());
    let expected = run(&mut StrictInt8Engine::new(model.weight_view().unwrap().with_kernel(Int8Kernel::scalar()),
        6, memory(6)).unwrap(), false);
    let serial_batch = run(&mut StrictInt8Engine::new(model.weight_view().unwrap().with_kernel(Int8Kernel::scalar()),
        6, memory(6)).unwrap(), true);
    let mut executor = BatchExecutor::new(&weights, 3);
    let actual = {
        let mut engine = StrictInt8Engine::new(model.weight_view().unwrap().with_kernel(Int8Kernel::scalar()),
            6, memory(6)).unwrap().with_projection_executor(&mut executor).unwrap();
        run(&mut engine, true)
    };
    assert_eq!(serial_batch, expected); assert_eq!(actual, expected);
    assert_eq!(executor.batch_calls, 2 * KV_SLOT_COUNT * 7);
    assert_eq!(executor.stats.completed_projections, (3 * KV_SLOT_COUNT * 7 + 2) as u64);
    assert_eq!(executor.stats.completed_rows, work.projections.dot_products);
}
