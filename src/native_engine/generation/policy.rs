//! Ordered processor stack and the existing pinned top-k/nucleus sampler.
use super::*;
use crate::native_engine::sampler::{DrawAddress, MAX_TOP_K, Nucleus, NucleusToken, TopK,
    SamplerError, SamplingCheckpoints, addressed_uniform};

pub(super) fn validate(p: &GenerationOptions, limits: GenerationLimits) -> Result<(), GenerationError> {
    if p.max_new_tokens == 0 || p.max_new_tokens > limits.max_new_tokens || p.min_new_tokens > p.max_new_tokens
        || p.max_output_bytes == 0 || p.max_output_bytes > limits.max_output_bytes {
        return Err(GenerationError::Limit("output tokens or bytes"));
    }
    if p.eos_token_ids.is_empty() || p.eos_token_ids.len() > 256 || p.banned_token_ids.len() > NANBEIGE_VOCAB_SIZE
        || p.logit_bias_milli.len() > 4096 || p.stop_suffixes.len() > 64
        || p.stop_suffixes.iter().any(|s| s.is_empty() || s.len() > 4096) {
        return Err(GenerationError::Contract("EOS, banned, bias or stop set"));
    }
    if p.eos_token_ids.iter().chain(&p.banned_token_ids).chain(p.logit_bias_milli.keys())
        .any(|&id| id as usize >= NANBEIGE_VOCAB_SIZE) {
        return Err(GenerationError::Contract("processor token outside vocabulary"));
    }
    if !(100..=10_000).contains(&p.repetition_penalty_milli)
        || p.presence_penalty_milli.unsigned_abs() > 10_000 || p.frequency_penalty_milli.unsigned_abs() > 10_000
        || p.logit_bias_milli.values().any(|bias| bias.unsigned_abs() > 100_000) {
        return Err(GenerationError::Contract("penalty or bias range"));
    }
    if let GenerationSampling::Seeded { temperature_milli, top_k, top_p_ppm, .. } = &p.sampling {
        if !(1..=100_000).contains(temperature_milli) || !(1..=1_000_000).contains(top_p_ppm)
            || top_k.is_some_and(|k| !(1..=MAX_TOP_K).contains(&k)) {
            return Err(GenerationError::Contract("temperature, top-k or top-p"));
        }
    }
    Ok(())
}

/// Counts + compact legal IDs/logits + the largest pinned Nucleus allocation.
/// Stack top-k arrays have an additional conservative allowance. This is a
/// payload bound, not allocator/RSS measurement or a no-hot-allocation claim.
pub(in crate::native_engine) fn workspace_bytes() -> Result<u64, GenerationError> {
    let per_row = 2 * size_of::<u32>() + size_of::<f32>() + size_of::<NucleusToken>();
    (NANBEIGE_VOCAB_SIZE as u64).checked_mul(per_row as u64)
        .and_then(|n| n.checked_add(4096)).ok_or(GenerationError::Limit("sampler accounting"))
}
fn tracks_frequencies(p: &GenerationOptions) -> bool {
    p.repetition_penalty_milli != 1000 || p.presence_penalty_milli != 0 || p.frequency_penalty_milli != 0
}

/// Generation binds the complete processor policy before allocating a cursor.
/// Raw model logits and result storage are accounted for by the native caller.
pub(super) fn generation_workspace_bytes(p: &GenerationOptions) -> Result<u64, GenerationError> {
    let mut per_row = if tracks_frequencies(p) { size_of::<u32>() } else { 0 };
    if let GenerationSampling::Seeded { top_k, .. } = &p.sampling {
        per_row += size_of::<u32>() + size_of::<f32>();
        // With top-k enabled, the nucleus contains at most MAX_TOP_K entries,
        // covered by the fixed 4096-byte allowance (including the top-k heap).
        if top_k.is_none() { per_row += size_of::<NucleusToken>(); }
    }
    (NANBEIGE_VOCAB_SIZE as u64).checked_mul(per_row as u64)
        .and_then(|bytes| bytes.checked_add(4096)).ok_or(GenerationError::Limit("sampler accounting"))
}

pub(in crate::native_engine) struct Workspace {
    counts: Vec<u32>, ids: Vec<u32>, logits: Vec<f32>, last_selected_id: Option<u32>,
    greedy_best: Option<(u32, f32)>,
}
impl Workspace {
    pub fn new(prompt: &[u32]) -> Result<Self, GenerationError> {
        Self::allocate(prompt, NANBEIGE_VOCAB_SIZE, NANBEIGE_VOCAB_SIZE)
    }
    pub(super) fn for_generation(prompt: &[u32], p: &GenerationOptions) -> Result<Self, GenerationError> {
        let frequency_rows = if tracks_frequencies(p) { NANBEIGE_VOCAB_SIZE } else { 0 };
        let sampled_rows = match &p.sampling {
            GenerationSampling::Greedy => 0,
            GenerationSampling::Seeded { .. } => NANBEIGE_VOCAB_SIZE,
        };
        Self::allocate(prompt, frequency_rows, sampled_rows)
    }
    fn allocate(prompt: &[u32], frequency_rows: usize, sampled_rows: usize) -> Result<Self, GenerationError> {
        let mut counts = reserved(frequency_rows)?;
        counts.resize(frequency_rows, 0_u32);
        let mut result = Self { counts, ids: reserved(sampled_rows)?,
            logits: reserved(sampled_rows)?, last_selected_id: None, greedy_best: None };
        for &id in prompt { result.commit(id)?; }
        Ok(result)
    }
    pub fn commit(&mut self, token: u32) -> Result<(), GenerationError> {
        if token as usize >= NANBEIGE_VOCAB_SIZE { return Err(GenerationError::InvalidLogits); }
        let Some(count) = self.counts.get_mut(token as usize) else { return Ok(()); };
        *count = count.checked_add(1).ok_or(GenerationError::Limit("token frequency"))?; Ok(())
    }
    pub fn select_with_control<C: DecodeStepControl + ?Sized>(&mut self, raw: &[f32],
        p: &GenerationOptions, key: StableRequestKey, sample_index: u64, step: usize,
        control: &mut C) -> Result<u32, GenerationError> {
        if raw.len() != NANBEIGE_VOCAB_SIZE { return Err(GenerationError::InvalidLogits); }
        self.select_processed(raw, p, key, sample_index, step, |_| true, control)
    }
    #[cfg(test)]
    fn select(&mut self, raw: &[f32], p: &GenerationOptions, key: StableRequestKey,
        sample_index: u64, step: u64) -> Result<u32, GenerationError> {
        self.select_with_control(raw, p, key, sample_index,
            usize::try_from(step).map_err(|_| GenerationError::Limit("decode step"))?, &mut Continue)
    }
    /// Exact external legality intersects the existing ordered processor
    /// stack before temperature, top-k and nucleus selection. The native
    /// owner checks its full vocabulary; a smaller finite width is admitted
    /// only for the closed decoder's synthetic differential fixtures.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::native_engine) fn select_legal_with_control<F: Fn(u32) -> bool,
        C: DecodeStepControl + ?Sized>(&mut self, raw: &[f32], p: &GenerationOptions,
        key: StableRequestKey, sample_index: u64, step: usize, legal: F, control: &mut C,
    ) -> Result<u32, GenerationError> {
        if raw.is_empty() || raw.len() > NANBEIGE_VOCAB_SIZE { return Err(GenerationError::InvalidLogits); }
        self.select_processed(raw, p, key, sample_index, step, legal, control)
    }
    #[cfg(test)]
    fn select_legal<F: Fn(u32) -> bool>(&mut self, raw: &[f32], p: &GenerationOptions,
        key: StableRequestKey, sample_index: u64, step: u64, legal: F) -> Result<u32, GenerationError> {
        self.select_legal_with_control(raw, p, key, sample_index,
            usize::try_from(step).map_err(|_| GenerationError::Limit("decode step"))?, legal, &mut Continue)
    }
    #[allow(clippy::too_many_arguments)]
    fn select_processed<F: Fn(u32) -> bool, C: DecodeStepControl + ?Sized>(
        &mut self, raw: &[f32], p: &GenerationOptions, key: StableRequestKey,
        sample_index: u64, step: usize, legal: F, control: &mut C,
    ) -> Result<u32, GenerationError> {
        self.begin_legal_rows();
        let scales = processor_scales(p);
        let mut checkpoints = SamplingCheckpoints::new(control, step);
        checkpoints.checkpoint().map_err(sampler_error)?;
        for (index, &raw_logit) in raw.iter().enumerate() {
            checkpoints.visit().map_err(sampler_error)?;
            // Illegal or banned rows must not conceal a corrupt projection.
            if !raw_logit.is_finite() { return Err(GenerationError::InvalidLogits); }
            let id = index as u32;
            if legal(id) { self.push_processed(id, raw_logit, p, step as u64, scales)?; }
        }
        checkpoints.checkpoint().map_err(sampler_error)?;
        self.finish_legal_rows_with_control(p, key, sample_index, step, control)
    }

    /// Start one exact selected-row step using the already admitted sampler
    /// buffers. The caller must supply EVERY legal row, in original ID order,
    /// before finishing; this private seam never invents absent logits.
    pub(in crate::native_engine) fn begin_legal_rows(&mut self) {
        self.ids.clear(); self.logits.clear(); self.last_selected_id = None; self.greedy_best = None;
    }

    /// Append a bounded projection block through the SAME ordered arithmetic
    /// as the full-head path. Ordering is checked across block boundaries, so
    /// compact positions retain the original-token-ID ranking and tie rules.
    pub(in crate::native_engine) fn extend_legal_rows(&mut self, rows: &[u32], raw: &[f32],
        p: &GenerationOptions, step: u64) -> Result<(), GenerationError> {
        if rows.is_empty() || rows.len() != raw.len()
            || rows.windows(2).any(|pair| pair[0] >= pair[1])
            || self.last_selected_id.is_some_and(|last| rows[0] <= last)
            || rows.last().is_some_and(|&id| id as usize >= NANBEIGE_VOCAB_SIZE)
            || raw.iter().any(|value| !value.is_finite()) {
            return Err(GenerationError::InvalidLogits);
        }
        let scales = processor_scales(p);
        for (&id, &logit) in rows.iter().zip(raw) { self.push_processed(id, logit, p, step, scales)?; }
        self.last_selected_id = rows.last().copied();
        Ok(())
    }

    fn push_processed(&mut self, id: u32, raw_logit: f32, p: &GenerationOptions, step: u64,
        (repetition, temperature): (f64, f64))
        -> Result<(), GenerationError> {
        // Static bans cannot be overridden by bias. EOS is additionally
        // absent until the required non-EOS output count has committed.
        if p.banned_token_ids.binary_search(&id).is_ok()
            || (step < p.min_new_tokens as u64 && p.eos_token_ids.binary_search(&id).is_ok()) {
            return Ok(());
        }
        let count = self.counts.get(id as usize).copied().unwrap_or(0);
        let mut value = f64::from(raw_logit);
        if count > 0 {
            value = if value > 0.0 { value / repetition } else { value * repetition };
            value -= f64::from(p.presence_penalty_milli) / 1000.0;
            value -= f64::from(p.frequency_penalty_milli) * f64::from(count) / 1000.0;
        }
        value += f64::from(*p.logit_bias_milli.get(&id).unwrap_or(&0)) / 1000.0;
        let processed = (value / temperature) as f32;
        if !processed.is_finite() { return Err(GenerationError::InvalidLogits); }
        if matches!(&p.sampling, GenerationSampling::Greedy) {
            // Rows arrive in ID order. Exact equality keeps the first legal ID.
            if self.greedy_best.is_none_or(|(_, best)| processed > best) {
                self.greedy_best = Some((id, processed));
            }
            return Ok(());
        }
        // Compact IDs remain increasing, preserving the pinned sampler's
        // lowest-original-ID tie break without feeding artificial -Inf.
        self.ids.push(id); self.logits.push(processed);
        Ok(())
    }

    /// Select only after the complete exact legal set has been projected.
    /// No native block, cap or physical schedule enters the addressed draw.
    pub(in crate::native_engine) fn finish_legal_rows_with_control<C: DecodeStepControl + ?Sized>(
        &self, p: &GenerationOptions, key: StableRequestKey, sample_index: u64, step: usize,
        control: &mut C) -> Result<u32, GenerationError> {
        checkpoint(control, step)?;
        let selected = match &p.sampling {
            GenerationSampling::Greedy => return self.greedy_best.map(|(id, _)| id)
                .ok_or(GenerationError::NoLegalToken),
            GenerationSampling::Seeded { effective_seed, top_k, top_p_ppm, .. } => {
                if self.ids.is_empty() { return Err(GenerationError::NoLegalToken); }
                let uniform = addressed_uniform(Seed256::from(*effective_seed), key,
                    DrawAddress::new(sample_index, step as u64, 0));
                select_uniform_with_control(&self.logits, *top_k,
                    f64::from(*top_p_ppm) / 1_000_000.0, uniform, control, step)?
            }
        };
        self.ids.get(selected).copied().ok_or(GenerationError::InvalidLogits)
    }
    #[cfg(test)]
    fn finish_legal_rows(&self, p: &GenerationOptions, key: StableRequestKey,
        sample_index: u64, step: u64) -> Result<u32, GenerationError> {
        self.finish_legal_rows_with_control(p, key, sample_index,
            usize::try_from(step).map_err(|_| GenerationError::Limit("decode step"))?, &mut Continue)
    }
}
fn processor_scales(p: &GenerationOptions) -> (f64, f64) {
    let repetition = f64::from(p.repetition_penalty_milli) / 1000.0;
    let temperature = match &p.sampling { GenerationSampling::Greedy => 1.0,
        GenerationSampling::Seeded { temperature_milli, .. } => f64::from(*temperature_milli) / 1000.0 };
    (repetition, temperature)
}
fn select_uniform_with_control<C: DecodeStepControl + ?Sized>(logits: &[f32], top_k: Option<usize>,
    p: f64, u: f64, control: &mut C, step: usize) -> Result<usize, GenerationError> {
    match top_k {
        None => Nucleus::exact_top_p_with_control(logits, p, control, step)
            .and_then(|n| n.select_uniform_with_control(u, control, step)).map_err(sampler_error),
        Some(k) => {
            let top = TopK::select_with_control(logits, k, control, step).map_err(sampler_error)?;
            if p == 1.0 { return top.select_uniform(u).map_err(sampler_error); }
            let entries = top.as_slice();
            let mut restricted = [0.0_f32; MAX_TOP_K];
            for (index, token) in entries.iter().enumerate() { restricted[index] = token.logit(); }
            // Nucleus is conditional on the exact top-k set here. Without k,
            // the other branch computes the complete legal-vocabulary normalizer.
            let index = Nucleus::exact_top_p_with_control(&restricted[..entries.len()], p, control, step)
                .and_then(|n| n.select_uniform_with_control(u, control, step)).map_err(sampler_error)?;
            Ok(entries[index].token_id())
        }
    }
}

pub(super) fn sampler_error(error: SamplerError) -> GenerationError {
    match error {
        SamplerError::Allocation => GenerationError::Allocation,
        SamplerError::Cancelled(cause) => GenerationError::Cancelled(cause),
        _ => GenerationError::InvalidLogits,
    }
}
#[cfg(test)]
struct Continue;
#[cfg(test)]
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}
#[cfg(test)]
fn select_uniform(logits: &[f32], top_k: Option<usize>, p: f64, u: f64) -> Result<usize, GenerationError> {
    select_uniform_with_control(logits, top_k, p, u, &mut Continue, 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nucleus_with_k_disabled_does_not_silently_use_twenty_rows() {
        let logits = vec![0.0; 100];
        assert_eq!(select_uniform(&logits, None, 1.0, 0.999).unwrap(), 99);
        assert_eq!(select_uniform(&logits, Some(20), 1.0, 0.999).unwrap(), 19);
    }
    #[test]
    fn k_precedes_conditional_nucleus_and_keeps_the_crossing_token() {
        let logits = [0.0, 0.0, 0.0, 0.0];
        assert_eq!(select_uniform(&logits, Some(2), 0.6, 0.9).unwrap(), 1);
        assert_eq!(select_uniform(&logits, None, 0.6, 0.9).unwrap(), 2);
    }
    #[test]
    fn processor_order_and_minimum_eos_are_not_bypassed_by_large_bias() {
        let mut p = GenerationOptions::greedy(4, 100, 0); p.min_new_tokens = 2;
        p.banned_token_ids = vec![1]; p.logit_bias_milli.insert(1, 100000);
        p.repetition_penalty_milli = 2000; p.frequency_penalty_milli = 1000;
        let mut workspace = Workspace::new(&[2, 2]).unwrap();
        let mut logits = vec![-100.0; NANBEIGE_VOCAB_SIZE]; logits[0] = 100.0; logits[1] = 100.0; logits[2] = 8.0; logits[3] = 3.0;
        let key = StableRequestKey::from_canonical_digest([0; 32]);
        assert_eq!(workspace.select(&logits, &p, key, 0, 0).unwrap(), 3);
        assert_eq!(workspace.select(&logits, &p, key, 0, 2).unwrap(), 0);
    }
    #[test]
    fn nonfinite_raw_rows_are_never_hidden_by_bans() {
        let mut p = GenerationOptions::greedy(1, 100, 0); p.banned_token_ids = vec![1];
        let mut logits = vec![0.0; NANBEIGE_VOCAB_SIZE]; logits[1] = f32::NAN;
        assert!(Workspace::new(&[0]).unwrap().select(&logits, &p, StableRequestKey::from_canonical_digest([0; 32]), 0, 0).is_err());
    }
    #[test]
    fn universal_external_legality_preserves_the_existing_processor_stack() {
        let mut p = GenerationOptions::greedy(4, 100, 0);
        p.banned_token_ids = vec![1]; p.min_new_tokens = 1;
        p.repetition_penalty_milli = 1500; p.presence_penalty_milli = 200;
        p.frequency_penalty_milli = 100; p.logit_bias_milli.insert(3, 750);
        let mut logits = vec![-100.0; NANBEIGE_VOCAB_SIZE];
        logits[..5].copy_from_slice(&[4.0, 100.0, 3.0, 2.0, 1.0]);
        let key = StableRequestKey::from_canonical_digest([7; 32]);
        for sampling in [GenerationSampling::Greedy, GenerationSampling::Seeded {
            effective_seed: [11; 32], temperature_milli: 700, top_k: Some(3), top_p_ppm: 900_000,
        }, GenerationSampling::Seeded {
            effective_seed: [11; 32], temperature_milli: 1200, top_k: None, top_p_ppm: 800_000,
        }] {
            p.sampling = sampling;
            for step in 0..3 {
                let mut ordinary = Workspace::new(&[2, 2]).unwrap();
                let mut masked = Workspace::new(&[2, 2]).unwrap();
                assert_eq!(ordinary.select(&logits, &p, key, 8, step).unwrap(),
                    masked.select_legal(&logits, &p, key, 8, step, |_| true).unwrap());
            }
        }
    }
    #[test]
    fn selected_blocks_reuse_bit_exact_processor_order_and_original_id_draws() {
        let mut p = GenerationOptions::greedy(4, 100, 0);
        p.min_new_tokens = 1; p.banned_token_ids = vec![1, 9];
        p.repetition_penalty_milli = 1357; p.presence_penalty_milli = -217;
        p.frequency_penalty_milli = 399; p.logit_bias_milli.insert(7, 1319);
        let raw = [2.0, 999.0, -0.0, 0.0, -2.75, 1.25, 8.0, -1.0, -17.125, 999.0,
            f32::MIN_POSITIVE, -f32::MIN_POSITIVE, 3.5, 0.75, -4.125, 0.25];
        let rows = [0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 13, 15];
        let key = StableRequestKey::from_canonical_digest([29; 32]);
        for sampling in [GenerationSampling::Greedy, GenerationSampling::Seeded {
            effective_seed: [61; 32], temperature_milli: 719, top_k: Some(4), top_p_ppm: 611_111,
        }, GenerationSampling::Seeded {
            effective_seed: [61; 32], temperature_milli: 1317, top_k: None, top_p_ppm: 1_000_000,
        }, GenerationSampling::Seeded {
            effective_seed: [61; 32], temperature_milli: 100_000, top_k: None, top_p_ppm: 700_001,
        }] {
            p.sampling = sampling;
            for step in 0..3 { for sample in 0..8 { for block in [1, 3, 8, rows.len()] {
                let mut dense = Workspace::new(&[2, 2, 3, 7, 7, 7, 11]).unwrap();
                let expected = dense.select_legal(&raw, &p, key, sample, step,
                    |id| rows.binary_search(&id).is_ok()).unwrap();
                let mut selected = Workspace::new(&[2, 2, 3, 7, 7, 7, 11]).unwrap();
                let capacities = (selected.ids.capacity(), selected.logits.capacity(), selected.counts.capacity());
                selected.begin_legal_rows();
                for ids in rows.chunks(block) {
                    let logits: Vec<_> = ids.iter().map(|&id| raw[id as usize]).collect();
                    selected.extend_legal_rows(ids, &logits, &p, step).unwrap();
                }
                assert_eq!(selected.ids, dense.ids);
                assert_eq!(selected.logits.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    dense.logits.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
                assert_eq!(selected.finish_legal_rows(&p, key, sample, step).unwrap(), expected);
                assert_eq!((selected.ids.capacity(), selected.logits.capacity(), selected.counts.capacity()), capacities);
            } } }
        }
    }
    #[test]
    fn selected_blocks_refuse_duplicates_reordering_shape_and_nonfinite_rows() {
        let p = GenerationOptions::greedy(2, 100, 0);
        let key = StableRequestKey::from_canonical_digest([0; 32]);
        let cases: &[(&[u32], &[f32])] = &[
            (&[], &[]), (&[2], &[]), (&[2, 2], &[1.0, 1.0]), (&[3, 2], &[1.0, 1.0]),
            (&[NANBEIGE_VOCAB_SIZE as u32], &[1.0]), (&[2], &[f32::NAN]),
            (&[2], &[f32::INFINITY]), (&[2], &[f32::NEG_INFINITY]),
        ];
        for &(ids, raw) in cases {
            let mut workspace = Workspace::new(&[]).unwrap();
            workspace.begin_legal_rows();
            assert!(matches!(workspace.extend_legal_rows(ids, raw, &p, 0), Err(GenerationError::InvalidLogits)));
        }
        let mut workspace = Workspace::new(&[]).unwrap();
        workspace.begin_legal_rows();
        assert!(matches!(workspace.finish_legal_rows(&p, key, 0, 0), Err(GenerationError::NoLegalToken)));
        workspace.extend_legal_rows(&[2, 4], &[1.0, 1.0], &p, 0).unwrap();
        assert!(workspace.extend_legal_rows(&[4, 5], &[1.0, 1.0], &p, 0).is_err());
        assert!(workspace.extend_legal_rows(&[3], &[1.0], &p, 0).is_err());
        workspace.begin_legal_rows();
        workspace.extend_legal_rows(&[1], &[1.0], &p, 0).unwrap();
        assert_eq!(workspace.finish_legal_rows(&p, key, 0, 0).unwrap(), 1);
    }

    #[test]
    fn processor_visits_are_bounded_even_when_every_row_is_illegal() {
        use std::cell::Cell;
        use crate::native_engine::sampler::SAMPLING_CHECKPOINT_STRIDE;
        struct Stop<'a> { visits: &'a Cell<usize>, cause: DecodeCancellationKind }
        impl DecodeStepControl for Stop<'_> {
            fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
                (self.visits.get() >= SAMPLING_CHECKPOINT_STRIDE).then_some(self.cause)
            }
        }
        let p = GenerationOptions::greedy(2, 100, 0);
        let key = StableRequestKey::from_canonical_digest([13; 32]);
        for cause in super::super::tests::CANCELLATION_KINDS {
            let visits = Cell::new(0);
            let mut control = Stop { visits: &visits, cause };
            let mut workspace = Workspace::new(&[]).unwrap();
            let result = workspace.select_legal_with_control(&vec![0.0; 1025], &p, key, 0, 7,
                |_| { visits.set(visits.get() + 1); false }, &mut control);
            assert!(matches!(result, Err(GenerationError::Cancelled(actual)) if actual == cause));
            assert_eq!(visits.get(), SAMPLING_CHECKPOINT_STRIDE);
            assert!(workspace.ids.is_empty());
        }
    }

    #[test]
    fn greedy_workspace_tracks_only_active_penalties_and_retains_the_first_legal_maximum() {
        let mut p = GenerationOptions::greedy(2, 100, 0);
        p.banned_token_ids = vec![0];
        let key = StableRequestKey::from_canonical_digest([0; 32]);
        let raw = [100.0, 3.0, 3.0, -1.0];
        for (penalty, expected, count_rows) in [(1000, 1, 0), (2000, 2, NANBEIGE_VOCAB_SIZE)] {
            p.repetition_penalty_milli = penalty;
            let mut workspace = Workspace::for_generation(&[1], &p).unwrap();
            let expected_bytes = (count_rows * size_of::<u32>()) as u64 + 4096;
            assert_eq!(generation_workspace_bytes(&p).unwrap(), expected_bytes);
            assert!(p.validate(GenerationLimits { max_sampler_bytes: expected_bytes, ..Default::default() }).is_ok());
            assert!(p.validate(GenerationLimits { max_sampler_bytes: expected_bytes - 1, ..Default::default() }).is_err());
            assert_eq!(workspace.select_legal(&raw, &p, key, 0, 0, |_| true).unwrap(), expected);
            assert_eq!((workspace.ids.capacity(), workspace.logits.capacity()), (0, 0));
            assert_eq!(workspace.counts.len(), count_rows);
            assert!(workspace.commit(NANBEIGE_VOCAB_SIZE as u32).is_err());
        }
    }

    #[test]
    fn sampler_allocation_and_cancellation_are_not_flattened_to_invalid_logits() {
        assert!(matches!(sampler_error(SamplerError::Allocation), GenerationError::Allocation));
        for cause in super::super::tests::CANCELLATION_KINDS {
            assert!(matches!(sampler_error(SamplerError::Cancelled(cause)),
                GenerationError::Cancelled(actual) if actual == cause));
        }
    }
}
