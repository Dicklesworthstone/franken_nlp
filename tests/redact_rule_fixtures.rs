//! Versioned, repo-authored detector fixtures. These measure named lexical
//! cases, not population-level PII recall or resistance to Unicode obfuscation.
use std::collections::BTreeSet;
use serde::Deserialize;
use franken_nlp::{canonjson, tasks::redact::{PiiKind, detectors::{RuleSet, RuleBudget, RULE_PROFILE, detect}}};

#[derive(Deserialize)]
struct Fixture { profile: String, cases: Vec<Case> }
#[derive(Deserialize)]
struct Case { id: String, enabled: BTreeSet<PiiKind>, source: String, expected: Vec<(PiiKind, String)> }

#[test]
fn versioned_rule_cases_match_exact_source_bytes() {
    let value = canonjson::parse_str(include_str!("fixtures/redact/rules-v1.json")).unwrap();
    let fixture: Fixture = serde_json::from_value(value).unwrap();
    assert_eq!(fixture.profile, RULE_PROFILE);
    for case in fixture.cases {
        let hits = detect(&case.source, &RuleSet { enabled: case.enabled }, RuleBudget::default()).unwrap();
        let actual: Vec<_> = hits.iter().map(|h| (h.kind, case.source[h.span.byte_start..h.span.byte_end].to_owned())).collect();
        assert_eq!(actual, case.expected, "fixture={}", case.id);
        for h in hits {
            assert_eq!(h.span.scalar_start, case.source[..h.span.byte_start].chars().count());
            assert_eq!(h.span.scalar_end, case.source[..h.span.byte_end].chars().count());
        }
    }
}
