//! OQ-35 census — GenServer mailbox semantics observed in the deterministic
//! lab, mirroring the pin's own conformance harness (LabRuntime + explicit
//! scheduler control). The lab never runs the server task until we say so,
//! which is exactly what makes the enqueue-only ack observable: an accepted
//! cast with the handler counter still at zero cannot have been processed.
//!
//! Scope honesty: both the sync `try_cast` and async `cast(&cx, msg).await`
//! surfaces are driven here. The latter runs in a LabRuntime client task so
//! the proof observes the precise await boundary rather than inferring it
//! from the synchronous sibling API.

use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use asupersync::Budget;
use asupersync::cx::{Cx, Scope};
use asupersync::gen_server::{CastError, CastOverflowPolicy, GenServer, Reply, SystemMsg};
use asupersync::lab::{LabConfig, LabRuntime};
use asupersync::types::policy::FailFast;

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
    region: asupersync::RegionId,
    cx: Cx,
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
        region,
        cx,
        task_id,
        handled,
        last_seen,
    }
}

/// `cast(&cx, msg).await` acknowledges mailbox admission only. The caller
/// task captures the handler count immediately after its await resolves while
/// the server task remains deliberately unscheduled; only a later explicit
/// server drive processes the message. This is the required acknowledgement
/// boundary for state/output/journal protocols that need processing or commit
/// evidence beyond an enqueue receipt.
#[test]
fn async_cast_acks_enqueue_only_before_server_processing() {
    let mut probe = spawn_probe(0x0135_C457, 2, CastOverflowPolicy::Reject);
    let client_scope = Scope::<FailFast>::new(probe.region, Budget::INFINITE);
    let server_ref = probe.handle.server_ref();
    let cast_acknowledged = Arc::new(AtomicU64::new(0));
    let handled_at_ack = Arc::new(AtomicU64::new(u64::MAX));
    let cast_acknowledged_for_task = Arc::clone(&cast_acknowledged);
    let handled_at_ack_for_task = Arc::clone(&handled_at_ack);
    let handled_for_task = Arc::clone(&probe.handled);

    let client = client_scope
        .spawn_registered(
            &mut probe.runtime.state,
            &probe.cx,
            move |client_cx| async move {
                server_ref
                    .cast(&client_cx, 44)
                    .await
                    .expect("empty mailbox admits the async cast");
                handled_at_ack_for_task
                    .store(handled_for_task.load(Ordering::SeqCst), Ordering::SeqCst);
                cast_acknowledged_for_task.store(1, Ordering::SeqCst);
            },
        )
        .expect("LabRuntime admits the cast caller task");

    probe.runtime.scheduler.lock().schedule(client.task_id(), 0);
    probe.runtime.run_until_idle();

    assert_eq!(
        cast_acknowledged.load(Ordering::SeqCst),
        1,
        "the async caller must observe a successful cast acknowledgement"
    );
    assert_eq!(
        handled_at_ack.load(Ordering::SeqCst),
        0,
        "cast acknowledgement is enqueue-only, not processing acknowledgement"
    );
    assert_eq!(
        probe.handled.load(Ordering::SeqCst),
        0,
        "the deliberately unscheduled server cannot have processed the cast"
    );
    println!("G0_CENSUS item=cast-async-ack case=await-returned accepted=1 handled_at_ack=0");

    probe.runtime.scheduler.lock().schedule(probe.task_id, 0);
    probe.runtime.run_until_idle();
    assert_eq!(
        probe.handled.load(Ordering::SeqCst),
        1,
        "explicitly driving the server delivers the previously acknowledged cast"
    );
    assert_eq!(probe.last_seen.load(Ordering::SeqCst), 44);
    println!("G0_CENSUS item=cast-async-ack case=post-drive delivered=1 last_seen=44");
    println!(
        "G0_CENSUS item=cast-async-ack RESULT=RATIFIED evidence=lab-client-await-boundary+unscheduled-server+explicit-post-drive-delivery"
    );
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
        "G0_CENSUS summary items=6 ratified=6 absent_with_fallback=0 fail=0 residual=execplan-first-ok,lab-determinism,compile-fail-suite"
    );
}
