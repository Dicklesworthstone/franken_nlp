use super::*;
use std::time::Duration;
use crate::native_engine::strict_int8::projection::{fixture::ProjectionFixture, Int8ProjectionKind};

struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}

#[test]
fn mismatched_batch_reply_never_assembles_a_stale_worker_stripe() {
    let fixture = ProjectionFixture::new(61); let weights = fixture.weights();
    let shared = RunShared::new(RunLimits { max_elapsed: Duration::from_secs(60),
        max_checkpoints: 1000, cleanup_reserve_bytes: 65536 }, CancellationToken::default()).unwrap();
    let mut executor = TeamExecutor::new(&weights, Int8Kernel::scalar(), 2, 3,
        Arc::new(SealedCpuTeam::new(1)), Arc::new(EntryGate::new()), shared).unwrap();
    let (command_sender, command_receiver) = capacity_one_lane();
    let (reply_sender, reply_receiver) = capacity_one_lane();
    executor.peers.push(Peer { command: Some(command_sender), reply: reply_receiver,
        buffers: Some(WorkerBuffers::new(2, 3).unwrap()) });
    let operation = Int8ProjectionOp::Layer { layer: 3, kind: Int8ProjectionKind::Gate };
    let mut stale = WorkerBuffers::new(2, 3).unwrap();
    stale.bf16.fill(Bf16::from_f32(99.0));
    assert!(reply_sender.try_send(Reply::Projected(ProjectionReply {
        epoch: 1, operation, kind: OutputKind::Bf16, range: 3..6,
        batch_rows: 2, buffers: stale, result: Ok(()),
    })).is_ok());
    drop(reply_sender);
    let inputs: Vec<_> = (0..3).map(|_| fixture.input()).collect();
    let sentinel = Bf16::from_f32(13.0);
    let mut output = [sentinel; 18];
    assert_eq!(executor.project_batch_bf16_into(operation, &inputs, LinearRows::All,
        &mut output, &mut Continue), Err(StrictInt8Error::ProjectionExecutor));
    // Coordinator cells may have been computed, but no worker stripe from
    // a different activation count can enter any output activation row.
    for row in output.chunks_exact(6) { assert_eq!(&row[3..], &[sentinel; 3]); }
    assert_eq!(executor.stats().completed_projections, 0);
    assert_eq!(executor.stats().completed_rows, 0);
    assert_eq!(executor.project_f32_into(Int8ProjectionOp::Head, &fixture.input(),
        LinearRows::All, &mut [0.0; 10], &mut Continue), Err(StrictInt8Error::ProjectionExecutor));
    drop(command_receiver);
}
