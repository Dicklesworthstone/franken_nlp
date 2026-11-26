use super::*;

struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}
fn span(a: usize, b: usize, c: usize, d: usize) -> VerifiedSourceSpan {
    VerifiedSourceSpan { byte_start: a, byte_end: b, scalar_start: c, scalar_end: d }
}
fn name(kind: EntityType, spans: Vec<VerifiedSourceSpan>) -> NamedEntity {
    NamedEntity { text: "李".to_owned(), entity_type: kind,
        occurrence: if spans.len() == 1 { SourceOccurrence::Anchored } else { SourceOccurrence::Ambiguous },
        spans }
}

#[test]
fn typed_surface_union_preserves_chunk_local_ambiguity_and_deduplicates_proposals() {
    let mut u = Union::new(Int8EntityLimits::default());
    let first = name(EntityType::Person, vec![span(0, 3, 0, 1), span(3, 6, 1, 2)]);
    let next = name(EntityType::Person, vec![span(0, 3, 0, 1)]);
    let place = name(EntityType::Location, vec![span(0, 3, 0, 1)]);
    let first_chunk = span(0, 8, 0, 4);
    let second_chunk = span(8, 11, 4, 5);
    for _ in 0..2 {
        u.add(0, first_chunk, &first, &first.spans, &mut Continue).unwrap();
    }
    u.add(1, second_chunk, &next, &[second_chunk], &mut Continue).unwrap();
    u.add(1, second_chunk, &place, &[second_chunk], &mut Continue).unwrap();
    assert_eq!((u.proposals, u.spans, u.work), (4, 4, 18));
    assert_eq!(u.rows.len(), 2);
    let people = &u.rows[&(EntityType::Person, "李")];
    assert_eq!(people.len(), 2);
    assert_eq!(people[0].occurrence, SourceOccurrence::Ambiguous);
    assert_eq!(people[0].spans, first.spans);
    assert_eq!(people[1].occurrence, SourceOccurrence::Anchored);
    assert_eq!(people[1].spans, [second_chunk]);
    assert_eq!(people[1].chunk_id, 1);
}

#[test]
fn invalid_coordinates_and_complete_evidence_limits_refuse() {
    let e = name(EntityType::Person, vec![span(0, 3, 0, 1), span(3, 6, 1, 2)]);
    let chunk = span(6, 12, 2, 4);
    let good = [span(6, 9, 2, 3), span(9, 12, 3, 4)];
    for bad in [span(6, 10, 2, 3), span(6, 9, 1, 2), span(7, 10, 2, 3)] {
        assert!(Union::new(Int8EntityLimits::default()).add(1, chunk, &e, &[bad, good[1]], &mut Continue).is_err());
    }
    let defaults = Int8EntityLimits::default();
    for limits in [Int8EntityLimits { max_evidence_spans: 1, ..defaults },
        Int8EntityLimits { max_scan_work: 4, ..defaults }, Int8EntityLimits { max_result_bytes: 2, ..defaults }] {
        assert!(Union::new(limits).add(1, chunk, &e, &good, &mut Continue).is_err());
    }
    let mut u = Union::new(Int8EntityLimits { max_proposals: 1, ..defaults });
    u.add(1, chunk, &e, &good, &mut Continue).unwrap();
    assert!(u.add(1, chunk, &e, &good, &mut Continue).is_err());
}
