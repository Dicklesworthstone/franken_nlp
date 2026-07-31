#![deny(unsafe_code)]

use franken_nlp::native_engine::{
    kv::{
        KV_BYTES_PER_TOKEN, KV_ELEMENTS_PER_POSITION, KV_SLOT_COUNT, KvCache, KvCacheError,
        LOOP_COUNT, PHYSICAL_LAYER_COUNT, slot_for,
    },
    looprun::{LayerBinding, LayerExecutor, LoopRunner, PositionContext},
};

#[derive(Default)]
struct RecordingExecutor {
    key: [u16; KV_ELEMENTS_PER_POSITION],
    value: [u16; KV_ELEMENTS_PER_POSITION],
    layer_calls: Vec<(usize, usize, usize, PositionContext)>,
    boundary: Option<(usize, i64, PositionContext)>,
    loop_two_entry: Option<(usize, i64, PositionContext)>,
    norm_calls: usize,
}

impl RecordingExecutor {
    fn reset_for_token(&mut self) {
        self.layer_calls.clear();
        self.boundary = None;
        self.loop_two_entry = None;
        self.norm_calls = 0;
    }
}

impl LayerExecutor<u16> for RecordingExecutor {
    type Hidden = i64;
    type Error = KvCacheError;

    fn layer_forward(
        &mut self,
        binding: &LayerBinding<'_, u16>,
        hidden: &mut Self::Hidden,
        positions: PositionContext,
        kv_cache: &mut KvCache,
    ) -> Result<(), Self::Error> {
        if binding.loop_index() == 1 && binding.layer_index() == 0 {
            self.loop_two_entry = Some((hidden as *mut i64 as usize, *hidden, positions));
        }
        self.key[0] = binding.kv_slot() as u16;
        self.value[0] = positions.cache_position as u16;
        kv_cache.append(
            binding.kv_slot(),
            positions.cache_position,
            &self.key,
            &self.value,
        )?;
        self.layer_calls.push((
            binding.loop_index(),
            binding.layer_index(),
            binding.kv_slot(),
            positions,
        ));
        *hidden += i64::from(*binding.weights()) + 1;
        Ok(())
    }

    fn final_rms_norm(
        &mut self,
        hidden: &mut Self::Hidden,
        positions: PositionContext,
    ) -> Result<(), Self::Error> {
        *hidden += 10_000;
        if self.norm_calls == 0 {
            self.boundary = Some((hidden as *mut i64 as usize, *hidden, positions));
        }
        self.norm_calls += 1;
        Ok(())
    }
}

fn runner() -> LoopRunner<'static, u16> {
    static WEIGHTS: [u16; PHYSICAL_LAYER_COUNT] = [0; PHYSICAL_LAYER_COUNT];
    LoopRunner::from_layer_weights(&WEIGHTS)
}

fn run_positions(cache: &mut KvCache, positions: std::ops::RangeInclusive<usize>) {
    let runner = runner();
    let mut executor = RecordingExecutor::default();
    for position in positions {
        let mut hidden = 7;
        runner
            .run_token(
                &mut executor,
                &mut hidden,
                PositionContext::at(position),
                cache,
            )
            .expect("test executor must append valid K/V values");
        executor.reset_for_token();
    }
}

#[test]
fn looprun_kv_contract() {
    let runner = runner();

    assert_eq!(KV_SLOT_COUNT, 44);
    assert_eq!(KV_BYTES_PER_TOKEN, 180_224);
    assert_eq!(KV_BYTES_PER_TOKEN / 1024, 176);
    for loop_index in 0..LOOP_COUNT {
        for layer_index in 0..PHYSICAL_LAYER_COUNT {
            let expected = layer_index + loop_index * PHYSICAL_LAYER_COUNT;
            assert_eq!(slot_for(loop_index, layer_index), Some(expected));
            let binding = runner
                .binding(loop_index, layer_index)
                .expect("all 44 bindings exist");
            assert_eq!(binding.kv_slot(), expected);
            assert_eq!(binding.loop_index(), loop_index);
            assert_eq!(binding.layer_index(), layer_index);
            assert!(std::ptr::eq(
                binding.weights(),
                &runner.bindings()[layer_index].weights(),
            ));
        }
    }
    assert_eq!(slot_for(LOOP_COUNT, 0), None);
    assert_eq!(slot_for(0, PHYSICAL_LAYER_COUNT), None);

    let positions = PositionContext::at(3);
    let mut hidden = 7;
    let mut cache = KvCache::try_with_capacity(4).expect("small fixed cache reserves");
    let mut executor = RecordingExecutor::default();
    runner
        .run_token(&mut executor, &mut hidden, positions, &mut cache)
        .expect("recording executor succeeds");

    assert_eq!(executor.layer_calls.len(), KV_SLOT_COUNT);
    assert_eq!(executor.norm_calls, LOOP_COUNT);
    assert!(executor.layer_calls.iter().all(|call| call.3 == positions));
    assert_eq!(cache.occupied_slot_positions(), KV_SLOT_COUNT);
    assert!(cache.all_slots_have_len(1));
    for slot in 0..KV_SLOT_COUNT {
        assert_eq!(cache.key_at(slot, 0).expect("written key")[0], slot as u16);
        assert_eq!(cache.value_at(slot, 0).expect("written value")[0], 3);
    }

    let boundary = executor.boundary.expect("loop one final norm is recorded");
    let loop_two_entry = executor
        .loop_two_entry
        .expect("loop two layer zero is recorded");
    assert_eq!(
        loop_two_entry.0, boundary.0,
        "loop two receives the same hidden allocation"
    );
    assert_eq!(
        loop_two_entry.1, boundary.1,
        "loop two receives the post-norm hidden value"
    );
    assert_eq!(
        loop_two_entry.2, boundary.2,
        "position/mask/RoPE coordinates are reused"
    );
    assert_ne!(
        loop_two_entry.1, 7,
        "loop two must not re-inject the embedding"
    );

    let duplicate = cache
        .append(
            0,
            0,
            &[0; KV_ELEMENTS_PER_POSITION],
            &[0; KV_ELEMENTS_PER_POSITION],
        )
        .expect_err("overwrite is not append semantics");
    assert!(matches!(
        duplicate,
        KvCacheError::NonAppendPosition {
            slot: 0,
            expected_position: 1,
            received_position: 0
        }
    ));
    assert!(matches!(
        cache.append(
            0,
            1,
            &[0; KV_ELEMENTS_PER_POSITION - 1],
            &[0; KV_ELEMENTS_PER_POSITION]
        ),
        Err(KvCacheError::InvalidVectorLength { .. })
    ));

    let sampled_prefill_lengths = [1, 2, 3, 5, 8, 13, 21, 34, 64];
    for prefill_len in sampled_prefill_lengths {
        let mut staged =
            KvCache::try_with_capacity(prefill_len + 1).expect("staged cache reserves");
        run_positions(&mut staged, 0..=prefill_len - 1);
        run_positions(&mut staged, prefill_len..=prefill_len);

        let mut one_shot =
            KvCache::try_with_capacity(prefill_len + 1).expect("one-shot cache reserves");
        run_positions(&mut one_shot, 0..=prefill_len);

        assert_eq!(
            staged, one_shot,
            "prefill then decode must equal one-shot prefill"
        );
        assert_eq!(
            staged.occupied_slot_positions(),
            KV_SLOT_COUNT * (prefill_len + 1)
        );
        assert!(staged.all_slots_have_len(prefill_len + 1));
    }

    eprintln!(
        "LOOPRUN RESULT=PASS slots={} norm_calls={} sampled_prefill_lengths={}",
        KV_SLOT_COUNT,
        LOOP_COUNT,
        sampled_prefill_lengths.len()
    );
}
