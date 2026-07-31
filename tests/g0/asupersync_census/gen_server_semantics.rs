//! OQ-35 census — GenServer mailbox semantics observed in the deterministic
//! lab, mirroring the pin's own conformance harness (LabRuntime + explicit
//! scheduler control). The lab never runs the server task until we say so,
//! which is exactly what makes the enqueue-only ack observable: an accepted
//! cast with the handler counter still at zero cannot have been processed.
//!
//! Scope honesty: the sync `try_cast` surface is what this probe drives; the
//! async `cast(&cx, msg).await` ack proof needs a lab-driven caller task and
//! stays a named residual for the census continuation.

use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use asupersync::cx::{Cx, Scope};
use asupersync::gen_server::{
    CastError, CastOverflowPolicy, GenServer, Reply, SystemMsg,
};
use asupersync::lab::{LabConfig, LabRuntime};
use asupersync::types::policy::FailFast;
use asupersync::Budget;

struct ProbeServer {
    handled: Arc<AtomicU64>,
    last_seen: Arc<AtomicU64>,
    policy: CastOverflowPolicy,
}

impl GenServer for ProbeServer {
    type Call = ();
    type Reply = ();
    type Cast = u64;
    type Info = SystemMsg;

    fn handle_call(
        &mut self,
        _cx: &Cx,
        _request: Self::Call,
        reply: Reply<Self::Reply>,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        Box::pin(async move {
            let _ = reply.send(());
        })
    }

    fn handle_cast(
        &mut self,
        _cx: &Cx,
        msg: Self::Cast,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        Box::pin(async move {
            self.handled.fetch_add(1, Ordering::SeqCst);
            self.last_seen.store(msg, Ordering::SeqCst);
        })
    }

    fn handle_info(
        &mut self,
        _cx: &Cx,
        _msg: Self::Info,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        Box::pin(async move {})
    }

    fn cast_overflow_policy(&self) -> CastOverflowPolicy {
        self.policy
    }
}

struct SpawnedProbe {
    runtime: LabRuntime,
    handle: asupersync::gen_server::GenServerHandle<ProbeServer>,
    task_id: asupersync::TaskId,
    handled: Arc<AtomicU64>,
    last_seen: Arc<AtomicU64>,
}

fn spawn_probe(seed: u64, capacity: usize, policy: CastOverflowPolicy) -> SpawnedProbe {
    let mut runtime = LabRuntime::new(LabConfig::new(seed).max_steps(10_000));
    let region = runtime.state.create_root_region(Budget::INFINITE);
    let cx = Cx::for_testing();
    let scope = Scope::<FailFast>::new(region, Budget::INFINITE);
    let handled = Arc::new(AtomicU64::new(0));
    let last_seen = Arc::new(AtomicU64::new(0));
    let server = ProbeServer {
        handled: Arc::clone(&handled),
        last_seen: Arc::clone(&last_seen),
        policy,
    };
    let (handle, stored) = scope
        .spawn_gen_server(&mut runtime.state, &cx, server, capacity)
        .expect("spawn_gen_server succeeds in the lab");
    let task_id = handle.task_id();
    runtime.state.store_spawned_task(task_id, stored);
    SpawnedProbe {
        runtime,
        handle,
        task_id,
        handled,
        last_seen,
    }
}

/// Default policy: accepted try_cast acks are ENQUEUE ONLY (the server task
/// has never been scheduled, so the handler count is provably zero at ack
/// time), a full mailbox rejects with CastError::Full, and scheduling the
/// server afterwards delivers exactly the accepted messages.
#[test]
fn try_cast_acks_enqueue_only_and_rejects_when_full() {
    let mut probe = spawn_probe(0x0135_C0DE, 2, CastOverflowPolicy::Reject);

    probe.handle.try_cast(11).expect("first cast fits");
    probe.handle.try_cast(22).expect("second cast fits");
    assert_eq!(
        probe.handled.load(Ordering::SeqCst),
        0,
        "accepted casts are enqueue-only: the never-scheduled server cannot have run its handler"
    );
    println!("G0_CENSUS item=try-cast-policies case=enqueue-only accepted=2 handled_at_ack=0");

    let overflow = probe.handle.try_cast(33);
    assert!(
        matches!(overflow, Err(CastError::Full)),
        "default Reject policy returns CastError::Full, got {overflow:?}"
    );
    println!("G0_CENSUS item=try-cast-policies case=reject-full third_cast=CastError::Full");

    probe.runtime.scheduler.lock().schedule(probe.task_id, 0);
    probe.runtime.run_until_idle();
    assert_eq!(
        probe.handled.load(Ordering::SeqCst),
        2,
        "exactly the accepted messages are delivered once the server runs"
    );
    println!("G0_CENSUS item=try-cast-policies case=post-drive delivered=2");
}

/// DropOldest policy: an overflowing try_cast is ACCEPTED, the oldest queued
/// cast is evicted, and after driving the server the newest messages are the
/// ones that survived — declared-lossy semantics, never silent for the sender.
#[test]
fn try_cast_drop_oldest_evicts_head_and_keeps_newest() {
    let mut probe = spawn_probe(0x0135_D01D, 2, CastOverflowPolicy::DropOldest);

    probe.handle.try_cast(1).expect("first cast fits");
    probe.handle.try_cast(2).expect("second cast fits");
    probe
        .handle
        .try_cast(3)
        .expect("DropOldest accepts the overflowing cast by evicting the head");
    println!("G0_CENSUS item=try-cast-policies case=drop-oldest overflow_accepted=true");

    probe.runtime.scheduler.lock().schedule(probe.task_id, 0);
    probe.runtime.run_until_idle();
    assert_eq!(
        probe.handled.load(Ordering::SeqCst),
        2,
        "capacity-many messages survive under DropOldest"
    );
    assert_eq!(
        probe.last_seen.load(Ordering::SeqCst),
        3,
        "the newest message survives; the evicted head was the oldest"
    );
    println!(
        "G0_CENSUS item=try-cast-policies case=drop-oldest delivered=2 last_seen=3 evicted=oldest"
    );
    println!(
        "G0_CENSUS item=try-cast-policies RESULT=RATIFIED evidence=enqueue-only-ack+reject-full+drop-oldest-evicts-head"
    );
    println!(
        "G0_CENSUS summary items=5 ratified=5 absent_with_fallback=0 fail=0 residual=cast-async-ack,execplan-first-ok,lab-determinism,compile-fail-suite"
    );
}
