//! G0-03 bounded model for the process-wide EngineResources broker.
//!
//! This proves the once-install and aggregate-reservation rules before the
//! production broker is introduced.  The ADR remains BLOCKED until a real
//! cgroup/job-object-aware discovery run supplies evidence for its host row.

use std::sync::{Arc, Barrier, Mutex};
use std::thread;

const SEED: u64 = 0x4730_3033;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ResourceConfig {
    weight_digest: [u8; 32],
    memory_limit_bytes: u64,
    worker_cap: u16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ResourceConfigConflict {
    field: &'static str,
}

#[derive(Debug)]
struct EngineResources {
    config: ResourceConfig,
}

#[derive(Debug, Default)]
struct Broker {
    installed: Mutex<Option<Arc<EngineResources>>>,
}

impl Broker {
    fn install(
        &self,
        requested: ResourceConfig,
    ) -> Result<Arc<EngineResources>, ResourceConfigConflict> {
        let mut installed = self.installed.lock().expect("broker lock poisoned");
        if let Some(existing) = installed.as_ref() {
            if existing.config.weight_digest != requested.weight_digest {
                return Err(ResourceConfigConflict {
                    field: "weight_digest",
                });
            }
            if existing.config.memory_limit_bytes != requested.memory_limit_bytes {
                return Err(ResourceConfigConflict {
                    field: "memory_limit_bytes",
                });
            }
            if existing.config.worker_cap != requested.worker_cap {
                return Err(ResourceConfigConflict {
                    field: "worker_cap",
                });
            }
            return Ok(Arc::clone(existing));
        }
        let resources = Arc::new(EngineResources { config: requested });
        *installed = Some(Arc::clone(&resources));
        Ok(resources)
    }
}

#[derive(Debug)]
struct MemoryLedger {
    capacity: u64,
    state: Mutex<LedgerState>,
}

#[derive(Debug, Default)]
struct LedgerState {
    reserved: u64,
    committed: u64,
}

#[derive(Debug)]
struct Reservation {
    bytes: u64,
    settled: bool,
}

impl MemoryLedger {
    fn new(capacity: u64) -> Self {
        Self {
            capacity,
            state: Mutex::new(LedgerState::default()),
        }
    }

    fn reserve(&self, bytes: u64) -> Option<Reservation> {
        let mut state = self.state.lock().expect("ledger lock poisoned");
        let total = state
            .reserved
            .checked_add(state.committed)?
            .checked_add(bytes)?;
        if total > self.capacity {
            return None;
        }
        state.reserved += bytes;
        Some(Reservation {
            bytes,
            settled: false,
        })
    }

    fn commit(&self, reservation: &mut Reservation) {
        assert!(!reservation.settled);
        let mut state = self.state.lock().expect("ledger lock poisoned");
        state.reserved -= reservation.bytes;
        state.committed += reservation.bytes;
        reservation.settled = true;
    }

    fn abort(&self, reservation: &mut Reservation) {
        assert!(!reservation.settled);
        let mut state = self.state.lock().expect("ledger lock poisoned");
        state.reserved -= reservation.bytes;
        reservation.settled = true;
    }
}

fn log_case(id: &str, result: &str) {
    println!("G0_PROBE3 case={id} RESULT={result} seed={SEED}");
}

#[test]
fn broker_model_has_one_winner_and_aggregate_memory_rollback() {
    let config = ResourceConfig {
        weight_digest: [0x5a; 32],
        memory_limit_bytes: 100,
        worker_cap: 4,
    };
    let broker = Arc::new(Broker::default());
    let barrier = Arc::new(Barrier::new(9));
    let mut joins = Vec::new();
    for _ in 0..8 {
        let broker = Arc::clone(&broker);
        let barrier = Arc::clone(&barrier);
        joins.push(thread::spawn(move || {
            barrier.wait();
            broker
                .install(config)
                .expect("compatible install must succeed")
        }));
    }
    barrier.wait();
    let winner = joins
        .pop()
        .expect("at least one contender")
        .join()
        .expect("winner thread panicked");
    for join in joins {
        let contender = join.join().expect("contender thread panicked");
        assert!(Arc::ptr_eq(&winner, &contender));
    }
    log_case("racing-first-install-one-arc", "PASS");

    assert_eq!(
        broker
            .install(ResourceConfig {
                memory_limit_bytes: 101,
                ..config
            })
            .expect_err("incompatible configuration must be refused"),
        ResourceConfigConflict {
            field: "memory_limit_bytes"
        }
    );
    log_case("field-level-config-conflict", "PASS");

    let ledger = MemoryLedger::new(100);
    let mut committed = ledger.reserve(60).expect("first reservation fits");
    assert!(
        ledger.reserve(50).is_none(),
        "aggregate reservation must refuse"
    );
    ledger.commit(&mut committed);
    let mut cancelled = ledger.reserve(30).expect("remaining capacity fits");
    ledger.abort(&mut cancelled);
    let state = ledger.state.lock().expect("ledger lock poisoned");
    assert_eq!(state.committed, 60);
    assert_eq!(state.reserved, 0, "cancel must roll reservation back");
    log_case("aggregate-two-phase-reserve-commit-abort", "PASS");

    println!("G0_PROBE3 RESULT=PASS cases=3 seed={SEED} authority=broker-model-only");
}
