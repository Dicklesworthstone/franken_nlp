//! Explicit layer-major prompt execution with the SAME generation cursor.
//!
//! Prompt scheduling never changes semantic identity, seed addressing, output
//! policy, or the per-row numerics program. Opt-in is mandatory until model
//! parity and target-host qualification exist. The host separately reserves
//! Int8PrefillLimits' extra workspace; limits alone are not an admission permit.
use super::*;
use crate::native_engine::strict_int8::prefill::Int8PrefillLimits;

impl Int8GenerationPlan {
    #[allow(clippy::too_many_arguments)]
    pub fn execute_layer_major<D: DecodeByteDecoder, C: DecodeStepControl>(&self,
        admitted: &ExecutionIdentity, engine: &mut StrictInt8Engine<'_>, decoder: &D,
        request_seq: u64, budget: Int8GenerationBudget, prefill: Int8PrefillLimits, control: &mut C)
        -> Result<Int8GenerationRun, Int8GenerationError> {
        self.execute_layer_major_with_sink(admitted, engine, decoder, request_seq, budget, prefill, &mut Discard, control)
    }

    /// Preflight precedes session creation. A failed prompt never emits a token;
    /// a later failure leaves only provisional events and aborts the session.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_layer_major_with_sink<D: DecodeByteDecoder, S: DecodeEventSink, C: DecodeStepControl>(&self,
        admitted: &ExecutionIdentity, engine: &mut StrictInt8Engine<'_>, decoder: &D,
        request_seq: u64, budget: Int8GenerationBudget, prefill: Int8PrefillLimits, sink: &mut S, control: &mut C)
        -> Result<Int8GenerationRun, Int8GenerationError> {
        prefill.validate()?;
        self.preflight(admitted, engine, budget)?;
        let mut session = engine.session(budget.native, control)?;
        self.drive_layer_major(decoder, request_seq, sink, &mut session, prefill)
    }

    fn drive_layer_major<D: DecodeByteDecoder, S: DecodeEventSink, B: PrefillDriver>(&self,
        decoder: &D, request_seq: u64, sink: &mut S, driver: &mut B, limits: Int8PrefillLimits)
        -> Result<Int8GenerationRun, Int8GenerationError> {
        limits.validate()?;
        let result = (|| {
            let mut row = cursor::Cursor::new(&self.plan, request_seq, INT8_GENERATION_VERSION)?;
            row.before_forward(driver.control())?;
            driver.append_prompt(&self.plan.prompt, limits)?;
            // Physical prompt work is already complete. Reconcile each logical
            // cursor position without rerunning the model or drawing randomness.
            for index in 0..self.plan.prompt.len() {
                let last = index + 1 == self.plan.prompt.len();
                if last {
                    let logits = driver.logits()?;
                    check_logits(&logits)?;
                    row.record_forward(true)?;
                    row.emit_next(&logits, decoder, sink, driver.control())?;
                } else { row.record_forward(false)?; }
            }
            while !row.done {
                row.before_forward(driver.control())?;
                let (token, needs_selection) = row.next_token()?;
                if !needs_selection { return Err(GenerationError::Contract("prefill cursor reconciliation").into()); }
                driver.append(token)?;
                let logits = driver.logits()?;
                check_logits(&logits)?;
                row.record_forward(true)?;
                row.emit_next(&logits, decoder, sink, driver.control())?;
            }
            let sequence = row.finish()?;
            let model_work = driver.work();
            check_completed(sequence.native_work, model_work)?;
            Ok(Int8GenerationRun { schema_version: 1, sequence, model_work })
        })();
        if result.is_err() { driver.abort(); }
        result
    }
}
trait PrefillDriver: Driver {
    fn append_prompt(&mut self, tokens: &[u32], limits: Int8PrefillLimits) -> Result<(), Int8GenerationError>;
}
impl<C: DecodeStepControl> PrefillDriver for Int8Session<'_, '_, C> {
    fn append_prompt(&mut self, tokens: &[u32], limits: Int8PrefillLimits) -> Result<(), Int8GenerationError> {
        self.append_layer_major(tokens, limits).map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Synthetic logits exercise cursor integration, NOT native model parity.
    fn plan(options: GenerationOptions) -> Int8GenerationPlan {
        let d = Sha256Digest::of_bytes(b"prefill-driver-fixture");
        let identity = ExecutionIdentity { schema_version: 1, source_revision: "fixture-revision".into(),
            logical_model_digest: d, artifact_format: "synthetic-only".into(), quant_recipe: "fixture-int8".into(),
            packing_set_digest: d, tokenizer_digest: d, template_digest: d, task_spec: "generate-v1".into(),
            taskir_digest: d, prompt_digest: d, grammar_compiler_version: "none".into(), schema_digest: d,
            numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".into(),
            sampler_version: "fixture".into(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
            calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT8_EXECUTION.into(),
            host_class: None, compiler_identity: None };
        Int8GenerationPlan::compile(vec![5, 6, 7, 8, 9], options, identity, "stable-item", 3, GenerationLimits::default()).unwrap()
    }
    fn limits(rows: usize) -> Int8PrefillLimits {
        Int8PrefillLimits { max_batch_rows: rows,
            max_extra_scratch_bytes: Int8PrefillLimits::required_extra_scratch_bytes(rows).unwrap() }
    }
    #[derive(Default)] struct Control { polls: usize, cancel: Option<usize> }
    impl DecodeStepControl for Control {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            self.polls += 1; (self.cancel == Some(self.polls)).then_some(DecodeCancellationKind::Deadline)
        }
    }
    struct Model { control: Control, tokens: Vec<u32>, preferred: Vec<u32>, heads: usize,
        prompt_calls: usize, flat: bool, fail_prompt: bool, forged: bool, aborted: bool }
    impl Model {
        fn new(preferred: &[u32]) -> Self { Self { control: Control::default(), tokens: Vec::new(), preferred: preferred.to_vec(),
            heads: 0, prompt_calls: 0, flat: false, fail_prompt: false, forged: false, aborted: false } }
    }
    impl Driver for Model {
        type Control = Control;
        fn control(&mut self) -> &mut Control { &mut self.control }
        fn append(&mut self, token: u32) -> Result<(), Int8GenerationError> { self.tokens.push(token); Ok(()) }
        fn logits(&mut self) -> Result<Vec<f32>, Int8GenerationError> {
            let mut output = vec![-100.0; NANBEIGE_VOCAB_SIZE];
            let selected = self.preferred[self.heads.min(self.preferred.len() - 1)] as usize; self.heads += 1;
            if self.flat { output[1..5].fill(0.0); } else { output[selected] = 10.0; }
            Ok(output)
        }
        fn work(&self) -> Int8Work {
            let mut work = Int8Work::for_sequence(0, self.tokens.len(), self.heads * NANBEIGE_VOCAB_SIZE).unwrap();
            if self.forged { work.attention_pairs += 1; } work
        }
        fn abort(&mut self) { self.aborted = true; }
    }
    impl PrefillDriver for Model {
        fn append_prompt(&mut self, tokens: &[u32], limits: Int8PrefillLimits) -> Result<(), Int8GenerationError> {
            self.prompt_calls += 1;
            if self.fail_prompt { return Err(StrictInt8Error::Primitive.into()); }
            for chunk in tokens.chunks(limits.max_batch_rows) { for &token in chunk {
                if let Some(cause) = self.control.checkpoint(self.tokens.len()) { return Err(StrictInt8Error::Cancelled(cause).into()); }
                self.append(token)?;
            } }
            Ok(())
        }
    }
    struct Decoder;
    impl DecodeByteDecoder for Decoder {
        type Error = &'static str;
        fn decode_token_ids(&self, tokens: &[u32]) -> Result<Vec<u8>, Self::Error> {
            Ok(tokens.iter().map(|&id| b'a' + id as u8).collect())
        }
    }
    #[derive(Default)] struct Sink { events: Vec<DecodeTokenEvent>, fail: bool }
    impl DecodeEventSink for Sink {
        type Permit = (); type Error = &'static str;
        fn reserve(&mut self, _: &DecodeTokenEvent) -> Result<(), Self::Error> { Ok(()) }
        fn permit(&mut self, _: (), event: DecodeTokenEvent) -> Result<(), Self::Error> {
            if self.fail { return Err("fixture delivery failure"); } self.events.push(event); Ok(())
        }
    }
    #[test] fn prompt_grouping_preserves_eos_stops_byte_limits_scores_and_work() {
        for case in 0..4 {
            let mut options = GenerationOptions::greedy(4, if case == 2 { 1 } else { 100 }, 0);
            options.capture_logprobs = true;
            if case == 1 { options.stop_suffixes = vec![b"bc".to_vec()]; }
            let preferred = if case == 0 { vec![1, 0] } else { vec![1, 2, 3, 4] };
            let p = plan(options); let mut serial = Model::new(&preferred);
            let baseline = p.drive(&Decoder, 7, &mut Sink::default(), &mut serial).unwrap();
            for width in [1, 3, 4, 64] {
                let mut grouped = Model::new(&preferred); let mut sink = Sink::default();
                let actual = p.drive_layer_major(&Decoder, 7, &mut sink, &mut grouped, limits(width)).unwrap();
                assert_eq!(canonjson::canonical_bytes(&baseline).unwrap(), canonjson::canonical_bytes(&actual).unwrap());
                assert_eq!(grouped.tokens, serial.tokens); assert_eq!(grouped.heads, serial.heads);
                assert_eq!(grouped.prompt_calls, 1); assert!(!grouped.aborted);
                assert_eq!(sink.events.iter().flat_map(|e| e.decoded_bytes.iter().copied()).collect::<Vec<_>>(), actual.sequence.content_bytes);
            }
        }
    }
    #[test] fn grouping_does_not_consume_draws_or_change_the_seeded_request_key() {
        let mut options = GenerationOptions::greedy(5, 100, 0);
        options.sampling = GenerationSampling::Seeded { effective_seed: [17; 32], temperature_milli: 800,
            top_k: Some(4), top_p_ppm: 950_000 };
        let p = plan(options); let mut serial = Model::new(&[1]); serial.flat = true;
        let expected = p.drive(&Decoder, 9, &mut Sink::default(), &mut serial).unwrap();
        for width in [1, 4, 64] {
            let mut model = Model::new(&[1]); model.flat = true;
            let actual = p.drive_layer_major(&Decoder, 9, &mut Sink::default(), &mut model, limits(width)).unwrap();
            assert_eq!(canonjson::canonical_bytes(&expected).unwrap(), canonjson::canonical_bytes(&actual).unwrap());
            assert_eq!(actual.sequence.native_work.sampled_steps, 5);
        }
    }
    #[test] fn prompt_failure_and_cancellation_abort_before_any_output_or_head() {
        let p = plan(GenerationOptions::greedy(4, 100, 0));
        for cancelled in [false, true] {
            let mut model = Model::new(&[1]); model.fail_prompt = !cancelled;
            if cancelled { model.control.cancel = Some(2); }
            let mut sink = Sink::default();
            let error = p.drive_layer_major(&Decoder, 1, &mut sink, &mut model, limits(4)).err().unwrap();
            if cancelled { assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline)); }
            assert!(model.aborted); assert!(sink.events.is_empty()); assert_eq!(model.heads, 0);
        }
    }
    #[test] fn delivery_and_forged_work_fail_without_success_or_retry() {
        let p = plan(GenerationOptions::greedy(1, 100, 0));
        for forged in [false, true] {
            let mut model = Model::new(&[1]); model.forged = forged;
            let mut sink = Sink { events: Vec::new(), fail: !forged };
            assert!(p.drive_layer_major(&Decoder, 1, &mut sink, &mut model, limits(4)).is_err());
            assert!(model.aborted); assert_eq!(model.heads, 1); assert_eq!(model.prompt_calls, 1);
        }
    }
    #[test] fn insufficient_extra_scratch_refuses_before_driver_entry() {
        let p = plan(GenerationOptions::greedy(1, 100, 0)); let mut model = Model::new(&[1]);
        let mut limit = limits(4); limit.max_extra_scratch_bytes -= 1;
        assert!(p.drive_layer_major(&Decoder, 1, &mut Sink::default(), &mut model, limit).is_err());
        assert_eq!(model.prompt_calls, 0); assert_eq!(model.heads, 0); assert!(!model.aborted);
    }
}
