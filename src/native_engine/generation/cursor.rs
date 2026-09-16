//! One sequence's resumable decode state; shared by scalar and batch drivers.
//! Physical row index and scheduler tick never enter the addressed sampler.
use super::*;

pub(super) struct Cursor<'a> {
    pub plan: &'a GenerationPlan,
    pub output: GeneratedSequence,
    pub prompt_cursor: usize,
    pub done: bool,
    awaiting_selection: bool,
    workspace: policy::Workspace,
}
impl<'a> Cursor<'a> {
    pub fn new(plan: &'a GenerationPlan, request_seq: u64, execution: &'static str) -> Result<Self, GenerationError> {
        let p = &plan.options;
        let output = GeneratedSequence {
            schema_version: 1, execution: execution.to_owned(), numerics_profile: HF_BF16_EAGER_PROFILE.to_owned(),
            request_seq, sample_index: plan.sample_index, token_ids: reserved(p.max_new_tokens)?, content_bytes: Vec::new(),
            finish_reason: GenerationFinish::TokenLimit,
            effective_seed: match &p.sampling { GenerationSampling::Greedy => None,
                GenerationSampling::Seeded { effective_seed, .. } => Some(Seed256::from(*effective_seed).to_lower_hex()) },
            token_logprobs: if p.capture_logprobs { Some(reserved(p.max_new_tokens)?) } else { None },
            logprob_score_space: p.capture_logprobs.then_some(DecodeScoreSpace::FullVocabularyLogSoftmax),
            native_work: GenerationWork::default(),
        };
        Ok(Self { plan, output, prompt_cursor: 0, done: false, awaiting_selection: false,
            workspace: policy::Workspace::new(&plan.prompt)? })
    }
    /// The boolean is true precisely when this forward must select a token.
    pub fn next_token(&self) -> Result<(u32, bool), GenerationError> {
        if self.done || self.awaiting_selection { return Err(GenerationError::Contract("cursor forward phase")); }
        if self.prompt_cursor < self.plan.prompt.len() {
            return Ok((self.plan.prompt[self.prompt_cursor], self.prompt_cursor + 1 == self.plan.prompt.len()));
        }
        self.output.token_ids.last().copied().map(|id| (id, true))
            .ok_or(GenerationError::Contract("missing feedback token"))
    }
    pub fn before_forward<C: DecodeStepControl>(&self, control: &mut C) -> Result<(), GenerationError> {
        self.next_token()?;
        if self.prompt_cursor < self.plan.prompt.len() {
            if let Some(cause) = control.prefill_checkpoint(self.prompt_cursor) { return Err(GenerationError::Cancelled(cause)); }
            Ok(())
        } else { checkpoint(control, self.output.token_ids.len()) }
    }
    pub fn record_forward(&mut self, projected: bool) -> Result<(), GenerationError> {
        let (_, selection) = self.next_token()?;
        if selection && !projected { return Err(GenerationError::Contract("missing selection projection")); }
        let work = &mut self.output.native_work;
        work.forward_positions = work.forward_positions.checked_add(1).ok_or(GenerationError::Limit("forward work"))?;
        if projected { work.projected_logits = work.projected_logits.checked_add(NANBEIGE_VOCAB_SIZE as u64)
            .ok_or(GenerationError::Limit("projection work"))?; }
        if work.forward_positions > self.plan.bound.forward_positions || work.projected_logits > self.plan.bound.projected_logits {
            return Err(GenerationError::Limit("cursor exceeded admitted work"));
        }
        if self.prompt_cursor < self.plan.prompt.len() { self.prompt_cursor += 1; }
        self.awaiting_selection = selection;
        Ok(())
    }
    pub fn emit_next<D: DecodeByteDecoder, S: DecodeEventSink, C: DecodeStepControl>(&mut self,
        logits: &[f32], decoder: &D, sink: &mut S, control: &mut C) -> Result<(), GenerationError> {
        if self.done || !self.awaiting_selection { return Err(GenerationError::Contract("cursor selection phase")); }
        let p = &self.plan.options;
        let index = self.output.token_ids.len();
        checkpoint(control, index)?;
        let selected = self.workspace.select(logits, p, self.plan.key, self.plan.sample_index, index as u64)?;
        if matches!(&p.sampling, GenerationSampling::Seeded { .. }) { self.output.native_work.sampled_steps += 1; }
        let eos = p.eos_token_ids.binary_search(&selected).is_ok();
        let score = if p.capture_logprobs { Some(raw_logprob(logits, selected)?) } else { None };
        let mut ids = reserved(index + 1)?; ids.extend_from_slice(&self.output.token_ids); ids.push(selected);
        let decoded = if eos {
            let mut bytes = reserved(self.output.content_bytes.len())?; bytes.extend_from_slice(&self.output.content_bytes); bytes
        } else { decoder.decode_token_ids(&ids).map_err(|_| GenerationError::Decoder)? };
        if !decoded.starts_with(&self.output.content_bytes) { return Err(GenerationError::DecoderNotPrefixStable); }
        if decoded.len() > p.max_output_bytes {
            self.output.finish_reason = GenerationFinish::ByteLimit; self.done = true; self.awaiting_selection = false; return Ok(());
        }
        let mut delta = reserved(decoded.len() - self.output.content_bytes.len())?;
        delta.extend_from_slice(&decoded[self.output.content_bytes.len()..]);
        let event = DecodeTokenEvent { schema_version: DECODE_TOKEN_EVENT_SCHEMA_VERSION,
            request_seq: self.output.request_seq, token_index: index, token_id: selected, decoded_bytes: delta, logprob: score };
        checkpoint(control, index)?;
        let permit = sink.reserve(&event).map_err(|_| GenerationError::Stream)?;
        // Cancellation while the sink waited releases the permit, not a token.
        checkpoint(control, index)?;
        sink.permit(permit, event).map_err(|_| GenerationError::Stream)?;
        self.output.token_ids = ids; self.output.content_bytes = decoded;
        if let Some(scores) = &mut self.output.token_logprobs { scores.push(score.ok_or(GenerationError::InvalidLogits)?); }
        self.workspace.commit(selected)?;
        self.awaiting_selection = false;
        let finish = if eos { Some(GenerationFinish::Eos) }
            else if index + 1 >= p.min_new_tokens && p.stop_suffixes.iter().any(|s| self.output.content_bytes.ends_with(s)) {
                Some(GenerationFinish::StopSuffix)
            } else if index + 1 == p.max_new_tokens { Some(GenerationFinish::TokenLimit) } else { None };
        if let Some(finish) = finish { self.output.finish_reason = finish; self.done = true; }
        Ok(())
    }
    pub fn finish(self) -> Result<GeneratedSequence, GenerationError> {
        if !self.done { return Err(GenerationError::Contract("unfinished cursor")); }
        Ok(self.output)
    }
}
