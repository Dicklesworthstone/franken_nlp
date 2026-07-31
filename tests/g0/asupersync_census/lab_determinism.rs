//! OQ-35 census — the deterministic LabRuntime is an async-state test
//! authority, not a claim about native `scoped_cpu` thread interleavings. This
//! probe records a competing cooperative workload twice from fresh runtimes:
//! the same seed must reproduce both its completion order and every replay
//! event.

use std::{
    future::Future,
    pin::Pin,
    sync::{Arc, Mutex},
    task::{Context, Poll},
};

use asupersync::{
    Budget,
    lab::{LabConfig, LabRuntime},
    trace::ReplayTrace,
    util::DetRng,
};

/// Cooperatively yield a fixed number of times to expose scheduler choices.
struct YieldTimes {
    remaining: u8,
}

impl Future for YieldTimes {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.remaining == 0 {
            Poll::Ready(())
        } else {
            self.remaining -= 1;
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }
}

fn record_seeded_run(seed: u64) -> (Vec<usize>, ReplayTrace) {
    let mut runtime = LabRuntime::new(
        LabConfig::new(seed)
            .max_steps(10_000)
            .with_default_replay_recording(),
    );
    let region = runtime.state.create_root_region(Budget::INFINITE);
    let completion_order = Arc::new(Mutex::new(Vec::new()));
    let mut task_ids = Vec::new();

    for task_index in 0..6 {
        let completion_order = Arc::clone(&completion_order);
        let yields = u8::try_from((task_index % 3) + 1).expect("bounded yield count fits u8");
        let (task_id, _) = runtime
            .state
            .create_task(region, Budget::INFINITE, async move {
                YieldTimes { remaining: yields }.await;
                completion_order
                    .lock()
                    .expect("completion order lock is not poisoned")
                    .push(task_index);
            })
            .expect("LabRuntime creates the deterministic task");
        task_ids.push(task_id);
    }

    let mut rng = DetRng::new(seed);
    for index in (1..task_ids.len()).rev() {
        let selected = rng.next_usize(index + 1);
        task_ids.swap(index, selected);
    }
    for task_id in task_ids {
        runtime.scheduler.lock().schedule(task_id, 0);
    }

    runtime.run_until_quiescent();
    assert!(runtime.is_quiescent(), "bounded Lab run reaches quiescence");
    let replay_trace = runtime
        .finish_replay_trace()
        .expect("replay recording is enabled by the fixed LabConfig");
    let completion_order = Arc::try_unwrap(completion_order)
        .expect("all deterministic task handles have drained")
        .into_inner()
        .expect("completion order lock is not poisoned");
    (completion_order, replay_trace)
}

#[test]
fn lab_runtime_replays_same_seed_with_identical_completion_and_trace() {
    const SEED: u64 = 0x035C_E15;

    let (first_completion, first_trace) = record_seeded_run(SEED);
    let (second_completion, second_trace) = record_seeded_run(SEED);

    assert_eq!(
        first_completion, second_completion,
        "the same Lab seed must preserve the cooperative completion order"
    );
    assert_eq!(
        first_trace.events, second_trace.events,
        "the same Lab seed must reproduce every recorded replay event"
    );
    assert!(
        !first_trace.events.is_empty(),
        "the probe must retain a non-empty replay artifact"
    );
    println!(
        "G0_CENSUS item=lab-determinism case=same-seed-replay seed={SEED:#x} completion={first_completion:?} replay_events={}",
        first_trace.events.len(),
    );
    println!(
        "G0_CENSUS item=lab-determinism RESULT=RATIFIED evidence=fresh-runtime+cooperative-workload+completion-order+replay-events"
    );
}
