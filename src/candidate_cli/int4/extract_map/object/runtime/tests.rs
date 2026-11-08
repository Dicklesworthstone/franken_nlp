use super::*;
use crate::tasks::extract::object_reconcile::RootObjectError;
use std::cell::Cell;

struct NoIo;
impl Read for NoIo {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("stale object admission reached IO"); }
}
impl Write for NoIo {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("stale object admission published bytes"); }
    fn flush(&mut self) -> io::Result<()> { panic!("stale object admission flushed output"); }
}

#[test]
fn runtime_revalidates_owned_limits_and_paths_instead_of_trusting_an_old_quote() {
    let (options, args) = super::super::tests::fixture(&[]);
    let old = options.validate(&args).unwrap();
    let mut changed = options.clone(); changed.object.max_object_nodes = 0;
    assert_eq!(execute(args.clone(), changed, old, &mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
    let (remote, _) = super::super::tests::fixture(&["--schema", "https://example.invalid/schema.json"]);
    assert_eq!(execute(args.clone(), remote, old, &mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
    let mut changed = args; changed.memory_mib = 1;
    assert_eq!(execute(changed, options, old, &mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
}

#[test]
fn original_deadline_applies_before_schema_derivation_and_retains_cancellation_kind() {
    let (options, args) = super::super::tests::fixture(&[]);
    let limits = options.validate(&args).unwrap();
    let budget = options.map.extraction.budget(&args, limits.map.common);
    let mut expired = DeadlineControl { started: Instant::now(), deadline: Duration::ZERO };
    // A cancelled derivation must not proceed to even malformed/private schema
    // diagnostics, and timeout is not converted to absent source data.
    assert_eq!(options.request("private malformed schema".to_owned(), budget, limits, &mut expired).err(),
        Some(CandidateError::Timeout));
    for cause in [DecodeCancellationKind::Deadline, DecodeCancellationKind::Timeout] {
        assert_eq!(map_planning_error(Int4ObjectExtractionMapError::Reconciliation(RootObjectError::Cancelled(cause))),
            CandidateError::Timeout);
        assert_eq!(execution_error(HostedError::Int4(Int4HostError::ObjectExtractionMap(
            Int4ObjectExtractionMapError::Reconciliation(RootObjectError::Cancelled(cause))))), CandidateError::Timeout);
    }
    assert_eq!(execution_error(HostedError::Int4(Int4HostError::ObjectExtractionMap(
        Int4ObjectExtractionMapError::Reconciliation(RootObjectError::Conflict)))), CandidateError::Execution);
    assert_eq!(execution_error(HostedError::ModelIdentity), CandidateError::Identity);
}

#[test]
fn expiration_after_complete_serialization_never_publishes_a_partial_object() {
    struct Expiring<'a>(&'a Cell<bool>);
    impl Serialize for Expiring<'_> {
        fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let result = "complete-private-object".serialize(serializer);
            self.0.set(true);
            result
        }
    }
    let expired = Cell::new(false);
    let mut output = Vec::new();
    assert_eq!(publish_checked(&Expiring(&expired), 4096, &mut output,
        &|| if expired.get() { Err(CandidateError::Timeout) } else { Ok(()) }), Err(CandidateError::Timeout));
    assert!(output.is_empty());
}
