//! G0-02 hostile state model for the blocking-pool to scoped-CPU seam.
//!
//! The model captures the non-negotiable formation ordering around the pinned
//! `ScopedCpu::spawn` drain-latch gap.  It does not substitute for the
//! configured-pool asupersync run required to ratify ADR G0-02.

use std::sync::{Arc, Barrier, Mutex};
use std::thread;
use std::time::Duration;

const SEED: u64 = 0x4730_3032;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FormationError {
    Capacity,
    SpawnPhaseSealed,
    CompletionLatched,
}

#[derive(Debug)]
struct SealedTeam {
    worker_cap: usize,
    spawned_children: usize,
    workers_released: bool,
    completion_latched: bool,
}

impl SealedTeam {
    fn new(worker_cap: usize) -> Self {
        Self {
            worker_cap,
            spawned_children: 0,
            workers_released: false,
            completion_latched: false,
        }
    }

    fn form_child(&mut self) -> Result<(), FormationError> {
        if self.completion_latched {
            return Err(FormationError::CompletionLatched);
        }
        if self.workers_released {
            return Err(FormationError::SpawnPhaseSealed);
        }
        if self.spawned_children == self.worker_cap {
            return Err(FormationError::Capacity);
        }
        self.spawned_children += 1;
        Ok(())
    }

    fn seal_and_release(&mut self) {
        assert_eq!(self.spawned_children, self.worker_cap);
        self.workers_released = true;
    }

    fn latch_completion(&mut self) {
        self.completion_latched = true;
    }
}

fn log_case(id: &str, result: &str) {
    println!("G0_PROBE2 case={id} RESULT={result} seed={SEED}");
}

#[test]
fn sealed_formation_model_rejects_post_start_and_post_latch_spawns() {
    let coordinator_width = 1usize;
    let worker_cap = 3usize;
    let mut team = SealedTeam::new(worker_cap);
    for _ in 0..worker_cap {
        assert_eq!(team.form_child(), Ok(()));
    }
    assert_eq!(team.form_child(), Err(FormationError::Capacity));
    assert_eq!(coordinator_width + team.spawned_children, 4);
    log_case("coordinator-child-width-arithmetic", "PASS");

    team.seal_and_release();
    assert_eq!(team.form_child(), Err(FormationError::SpawnPhaseSealed));
    log_case("post-start-spawn-refused", "PASS");

    let team = Arc::new(Mutex::new(team));
    let start = Arc::new(Barrier::new(9));
    let mut joins = Vec::new();
    for _ in 0..8 {
        let team = Arc::clone(&team);
        let start = Arc::clone(&start);
        joins.push(thread::spawn(move || {
            start.wait();
            team.lock().expect("team lock poisoned").form_child()
        }));
    }
    start.wait();
    for join in joins {
        assert_eq!(
            join.join().expect("hostile interleaving thread panicked"),
            Err(FormationError::SpawnPhaseSealed)
        );
    }
    log_case("hostile-post-start-interleaving", "PASS");

    team.lock().expect("team lock poisoned").latch_completion();
    assert_eq!(
        team.lock().expect("team lock poisoned").form_child(),
        Err(FormationError::CompletionLatched)
    );
    log_case("post-latch-spawn-refused", "PASS");

    let heartbeat = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let heartbeat_worker = Arc::clone(&heartbeat);
    let heartbeat_join = thread::spawn(move || {
        for _ in 0..8 {
            heartbeat_worker.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            thread::sleep(Duration::from_millis(1));
        }
    });
    thread::sleep(Duration::from_millis(3));
    assert!(heartbeat.load(std::sync::atomic::Ordering::SeqCst) > 0);
    heartbeat_join.join().expect("heartbeat thread panicked");
    log_case("coordinator-busy-heartbeat-model", "PASS");

    println!("G0_PROBE2 RESULT=PASS cases=5 seed={SEED} authority=sealed-model-only");
}
