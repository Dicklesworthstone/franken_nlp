#![deny(unsafe_code)]

#[allow(dead_code)]
#[path = "spec_engine/mod.rs"]
mod spec_engine;

use franken_nlp::native_engine::{
    kv::{KV_ELEMENTS_PER_POSITION, KV_SLOT_COUNT, KvCache, KvCacheError},
    looprun::{LayerBinding, LayerExecutor, LoopRunner, PositionContext},
};
use spec_engine::{
    KvCache as SpecKvCache, PHYSICAL_LAYERS, SpecConfig, SpecEngine, SpecError, SpecWeights,
    rms_norm,
};

#[derive(Debug)]
enum CrossCheckError {
    Kv(KvCacheError),
    Spec(SpecError),
}

impl From<KvCacheError> for CrossCheckError {
    fn from(value: KvCacheError) -> Self {
        Self::Kv(value)
    }
}

impl From<SpecError> for CrossCheckError {
    fn from(value: SpecError) -> Self {
        Self::Spec(value)
    }
}

struct ScalarBoundaryExecutor {
    key: [u16; KV_ELEMENTS_PER_POSITION],
    value: [u16; KV_ELEMENTS_PER_POSITION],
    invocations: Vec<(usize, usize, usize)>,
    epsilon: f32,
}

impl ScalarBoundaryExecutor {
    fn new(epsilon: f32) -> Self {
        Self {
            key: [0; KV_ELEMENTS_PER_POSITION],
            value: [0; KV_ELEMENTS_PER_POSITION],
            invocations: Vec::with_capacity(KV_SLOT_COUNT),
            epsilon,
        }
    }
}

impl LayerExecutor<()> for ScalarBoundaryExecutor {
    type Hidden = Vec<f32>;
    type Error = CrossCheckError;

    fn layer_forward(
        &mut self,
        binding: &LayerBinding<'_, ()>,
        _hidden: &mut Self::Hidden,
        positions: PositionContext,
        kv_cache: &mut KvCache,
    ) -> Result<(), Self::Error> {
        kv_cache.append(
            binding.kv_slot(),
            positions.cache_position,
            &self.key,
            &self.value,
        )?;
        self.invocations.push((
            binding.loop_index(),
            binding.layer_index(),
            binding.kv_slot(),
        ));
        Ok(())
    }

    fn final_rms_norm(
        &mut self,
        hidden: &mut Self::Hidden,
        _positions: PositionContext,
    ) -> Result<(), Self::Error> {
        *hidden = rms_norm(
            hidden,
            &vec![1.0; hidden.len()],
            self.epsilon,
            "product_runner_boundary_norm",
        )?;
        Ok(())
    }
}

#[test]
fn runner_schedule_and_boundary_norm_match_independent_scalar_spec() {
    let config = SpecConfig::tiny_for_tests();
    let mut weights = SpecWeights::zeroed(&config).expect("tiny spec weights are valid");
    weights.embeddings.set(1, 0, 3.0).expect("row write");
    weights.embeddings.set(1, 1, 4.0).expect("row write");
    let spec = SpecEngine::new(config.clone(), weights).expect("tiny spec model is valid");
    let mut spec_cache = SpecKvCache::new(&config);
    let expected = spec.decode(1, &mut spec_cache).expect("scalar forward succeeds");

    static PHYSICAL_WEIGHTS: [(); PHYSICAL_LAYERS] = [(); PHYSICAL_LAYERS];
    let runner = LoopRunner::from_layer_weights(&PHYSICAL_WEIGHTS);
    let mut product_cache = KvCache::try_with_capacity(1).expect("one token cache reserves");
    let mut executor = ScalarBoundaryExecutor::new(config.rms_epsilon);
    let mut hidden = vec![3.0, 4.0, 0.0, 0.0];
    runner
        .run_token(
            &mut executor,
            &mut hidden,
            PositionContext::at(0),
            &mut product_cache,
        )
        .expect("runner writes every logical slot");

    assert_eq!(executor.invocations.len(), KV_SLOT_COUNT);
    assert_eq!(expected.taps.layer_taps.len(), KV_SLOT_COUNT);
    for (index, ((loop_index, layer_index, slot), expected_tap)) in executor
        .invocations
        .iter()
        .zip(&expected.taps.layer_taps)
        .enumerate()
    {
        assert_eq!(*loop_index, expected_tap.loop_index, "call={index}");
        assert_eq!(*layer_index, expected_tap.layer_index, "call={index}");
        assert_eq!(*slot, layer_index + loop_index * PHYSICAL_LAYERS, "call={index}");
    }
    assert_eq!(hidden, expected.taps.post_loop_norms[1]);
    assert!(product_cache.all_slots_have_len(1));
    for slot in 0..KV_SLOT_COUNT {
        assert_eq!(spec_cache.slot_len(slot).expect("spec slot"), 1);
        assert_eq!(product_cache.len_for_slot(slot).expect("product slot"), 1);
    }

    eprintln!(
        "LOOPRUN RESULT=PASS spec_engine=scalar_f32 slots={} boundary_norms=2",
        KV_SLOT_COUNT
    );
}
