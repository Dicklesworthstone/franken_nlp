//! G0-05 structural loop-boundary probe against the frozen trace inventory.
//!
//! It makes the 22 -> norm -> 22 -> norm schedule and 44-slot mapping
//! executable without claiming a whole-model numerical implementation.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use serde_json::Value;

use franken_nlp::native_engine::looprun::{
    LayerBinding, LoopRunner, PositionContext, StructuralLayerExecutor,
};

const PHYSICAL_LAYERS: usize = 22;
const LOOPS: usize = 2;
const LOGICAL_SLOTS: usize = PHYSICAL_LAYERS * LOOPS;

fn fixture_root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/reference")
}

fn trace_index() -> Value {
    let bytes = fs::read(fixture_root().join("hf-bf16-eager/prompt-000/trace.json"))
        .expect("eager reference trace must remain readable");
    serde_json::from_slice(&bytes).expect("eager reference trace must remain JSON")
}

fn phase_slots(trace: &Value, phase: &str, tap: &str) -> BTreeSet<(u64, u64)> {
    trace[phase]["records"]
        .as_array()
        .expect("trace phase records")
        .iter()
        .filter(|record| record["tap_name"].as_str() == Some(tap))
        .map(|record| {
            (
                record["loop"].as_u64().expect("trace loop"),
                record["layer"].as_u64().expect("trace layer"),
            )
        })
        .collect()
}

fn expected_slots() -> BTreeSet<(u64, u64)> {
    (0..LOOPS)
        .flat_map(|loop_index| {
            (0..PHYSICAL_LAYERS).map(move |layer_index| (loop_index as u64, layer_index as u64))
        })
        .collect()
}

#[derive(Default)]
struct RecordingExecutor {
    layer_events: Vec<(usize, usize, usize, u32)>,
    norm_inputs: Vec<u32>,
}

impl StructuralLayerExecutor<u8> for RecordingExecutor {
    type Hidden = u32;
    type Error = std::convert::Infallible;

    fn layer_forward(
        &mut self,
        binding: &LayerBinding<'_, u8>,
        hidden: &mut Self::Hidden,
        positions: PositionContext,
    ) -> Result<(), Self::Error> {
        assert_eq!(positions.position, positions.rope_position);
        assert_eq!(positions.position, positions.cache_position);
        self.layer_events.push((
            binding.loop_index(),
            binding.layer_index(),
            binding.kv_slot(),
            *hidden,
        ));
        *hidden += 1;
        Ok(())
    }

    fn final_rms_norm(
        &mut self,
        hidden: &mut Self::Hidden,
        _positions: PositionContext,
    ) -> Result<(), Self::Error> {
        self.norm_inputs.push(*hidden);
        *hidden += 1_000;
        Ok(())
    }
}

#[test]
fn scalar_runner_feeds_the_first_normalized_state_directly_to_loop_two() {
    let physical_weights = [0_u8; PHYSICAL_LAYERS];
    let runner = LoopRunner::from_layer_weights(&physical_weights);
    let mut executor = RecordingExecutor::default();
    let mut hidden = 0_u32;
    runner
        .run_token_structural(&mut executor, &mut hidden, PositionContext::at(7))
        .expect("infallible scalar loop probe");
    assert_eq!(executor.norm_inputs, vec![22, 1_044]);
    assert_eq!(executor.layer_events.len(), LOGICAL_SLOTS);
    assert_eq!(executor.layer_events[0], (0, 0, 0, 0));
    assert_eq!(executor.layer_events[21], (0, 21, 21, 21));
    assert_eq!(executor.layer_events[22], (1, 0, 22, 1_022));
    assert_eq!(executor.layer_events[43], (1, 21, 43, 1_043));
    println!(
        "G0_PROBE5 case=scalar-loop-runner RESULT=PASS layer_slots={LOGICAL_SLOTS} norms={LOOPS} authority=structural-scalar-only"
    );
}

#[test]
fn loop_boundary_trace_has_two_full_passes_and_all_kv_slots() {
    let trace = trace_index();
    let expected = expected_slots();
    for phase in ["prefill", "append"] {
        let layers = phase_slots(&trace, phase, "post_layer");
        let keys = phase_slots(&trace, phase, "kv_key");
        let values = phase_slots(&trace, phase, "kv_value");
        let norms = trace[phase]["records"]
            .as_array()
            .expect("trace phase records")
            .iter()
            .filter(|record| record["tap_name"].as_str() == Some("post_loop_norm"))
            .count();
        assert_eq!(
            layers, expected,
            "phase={phase} must visit every logical layer"
        );
        assert_eq!(keys, expected, "phase={phase} must populate every K slot");
        assert_eq!(values, expected, "phase={phase} must populate every V slot");
        assert_eq!(norms, LOOPS, "phase={phase} must finish both loop norms");
    }
    println!(
        "G0_PROBE5 case=trace-loop-boundary RESULT=PASS layer_slots={LOGICAL_SLOTS} kv_slots={LOGICAL_SLOTS} norms={LOOPS} authority=fixture-structure-only"
    );
    println!("G0_PROBE5 RESULT=PASS cases=1 authority=fixture-structure-only");
}
