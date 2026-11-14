//! Ordered processor stack and the existing pinned top-k/nucleus sampler.
use super::*;
use crate::native_engine::sampler::{DrawAddress, MAX_TOP_K, Nucleus, NucleusToken, TopK, addressed_uniform};

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
pub(in crate::native_engine) struct Workspace { counts: Vec<u32>, ids: Vec<u32>, logits: Vec<f32> }
impl Workspace {
    pub fn new(prompt: &[u32]) -> Result<Self, GenerationError> {
        let mut counts = reserved(NANBEIGE_VOCAB_SIZE)?; counts.resize(NANBEIGE_VOCAB_SIZE, 0_u32);
        let mut result = Self { counts, ids: reserved(NANBEIGE_VOCAB_SIZE)?, logits: reserved(NANBEIGE_VOCAB_SIZE)? };
        for &id in prompt { result.commit(id)?; }
        Ok(result)
    }
    pub fn commit(&mut self, token: u32) -> Result<(), GenerationError> {
        let count = self.counts.get_mut(token as usize).ok_or(GenerationError::InvalidLogits)?;
        *count = count.checked_add(1).ok_or(GenerationError::Limit("token frequency"))?; Ok(())
    }
    pub fn select(&mut self, raw: &[f32], p: &GenerationOptions, key: StableRequestKey,
        sample_index: u64, step: u64) -> Result<u32, GenerationError> {
        check_logits(raw)?;
        self.select_processed(raw, p, key, sample_index, step, |_| true)
    }
    /// Exact external legality intersects the existing ordered processor
    /// stack before temperature, top-k and nucleus selection. The native
    /// owner checks its full vocabulary; a smaller finite width is admitted
    /// only for the closed decoder's synthetic differential fixtures.
    pub(in crate::native_engine) fn select_legal<F: Fn(u32) -> bool>(
        &mut self, raw: &[f32], p: &GenerationOptions, key: StableRequestKey,
        sample_index: u64, step: u64, legal: F,
    ) -> Result<u32, GenerationError> {
        if raw.is_empty() || raw.len() > self.counts.len() || raw.iter().any(|value| !value.is_finite()) {
            return Err(GenerationError::InvalidLogits);
        }
        self.select_processed(raw, p, key, sample_index, step, legal)
    }
    fn select_processed<F: Fn(u32) -> bool>(
        &mut self, raw: &[f32], p: &GenerationOptions, key: StableRequestKey,
        sample_index: u64, step: u64, legal: F,
    ) -> Result<u32, GenerationError> {
        self.ids.clear(); self.logits.clear();
        let repetition = f64::from(p.repetition_penalty_milli) / 1000.0;
        let temperature = match &p.sampling { GenerationSampling::Greedy => 1.0,
            GenerationSampling::Seeded { temperature_milli, .. } => f64::from(*temperature_milli) / 1000.0 };
        for (index, &raw_logit) in raw.iter().enumerate() {
            let id = index as u32;
            // Static bans cannot be overridden by bias. EOS is additionally
            // absent until the required non-EOS output count has committed.
            if !legal(id) || p.banned_token_ids.binary_search(&id).is_ok()
                || (step < p.min_new_tokens as u64 && p.eos_token_ids.binary_search(&id).is_ok()) { continue; }
            let count = self.counts[index];
            let mut value = f64::from(raw_logit);
            if count > 0 {
                value = if value > 0.0 { value / repetition } else { value * repetition };
                value -= f64::from(p.presence_penalty_milli) / 1000.0;
                value -= f64::from(p.frequency_penalty_milli) * f64::from(count) / 1000.0;
            }
            value += f64::from(*p.logit_bias_milli.get(&id).unwrap_or(&0)) / 1000.0;
            let processed = (value / temperature) as f32;
            if !processed.is_finite() { return Err(GenerationError::InvalidLogits); }
            // Compact IDs remain increasing, preserving the pinned sampler's
            // lowest-original-ID tie break without feeding artificial -Inf.
            self.ids.push(id); self.logits.push(processed);
        }
        if self.ids.is_empty() { return Err(GenerationError::NoLegalToken); }
        let selected = match &p.sampling {
            GenerationSampling::Greedy => {
                let mut best = 0;
                for index in 1..self.logits.len() {
                    if self.logits[index] > self.logits[best] { best = index; }
                }
                best
            }
            GenerationSampling::Seeded { effective_seed, top_k, top_p_ppm, .. } => {
                let uniform = addressed_uniform(Seed256::from(*effective_seed), key, DrawAddress::new(sample_index, step, 0));
                select_uniform(&self.logits, *top_k, f64::from(*top_p_ppm) / 1_000_000.0, uniform)?
            }
        };
        self.ids.get(selected).copied().ok_or(GenerationError::InvalidLogits)
    }
}
fn select_uniform(logits: &[f32], top_k: Option<usize>, p: f64, u: f64) -> Result<usize, GenerationError> {
    match top_k {
        None => Nucleus::exact_top_p(logits, p).and_then(|n| n.select_uniform(u)).map_err(|_| GenerationError::InvalidLogits),
        Some(k) => {
            let top = TopK::select(logits, k).map_err(|_| GenerationError::InvalidLogits)?;
            if p == 1.0 { return top.select_uniform(u).map_err(|_| GenerationError::InvalidLogits); }
            let entries = top.as_slice();
            let mut restricted = [0.0_f32; MAX_TOP_K];
            for (index, token) in entries.iter().enumerate() { restricted[index] = token.logit(); }
            // Nucleus is conditional on the exact top-k set here. Without k,
            // the other branch computes the complete legal-vocabulary normalizer.
            let index = Nucleus::exact_top_p(&restricted[..entries.len()], p)
                .and_then(|n| n.select_uniform(u)).map_err(|_| GenerationError::InvalidLogits)?;
            Ok(entries[index].token_id())
        }
    }
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
}
