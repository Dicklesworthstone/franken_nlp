//! Versioned rule automata, not a general Unicode or locale-aware PII oracle.
//! ASCII email/phone/card shapes, HTTP(S) URLs, IP literals, and opt-in ISO
//! calendar dates. Unicode text around a match retains exact original offsets.
//! Homoglyph/zero-width obfuscation is NOT normalized or universally detected.

use std::{collections::BTreeSet, net::IpAddr, str::FromStr};
use serde::{Deserialize, Serialize};
use super::{Detection, Detector, PiiKind, RedactError};
use crate::validation::grounded_fields::VerifiedSourceSpan;

pub const RULE_PROFILE: &str = "ascii-contact-ip-luhn-iso-date-v1";
const MAX_DETECTIONS: usize = 16_384;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuleSet { pub enabled: BTreeSet<PiiKind> }
impl Default for RuleSet {
    fn default() -> Self {
        Self { enabled: [PiiKind::Email, PiiKind::Phone, PiiKind::Url,
            PiiKind::IpAddress, PiiKind::CreditCard].into_iter().collect() }
    }
}
impl RuleSet {
    pub fn validate(&self) -> Result<(), RedactError> {
        if self.enabled.iter().any(|kind| !matches!(kind, PiiKind::Email | PiiKind::Phone
            | PiiKind::Url | PiiKind::IpAddress | PiiKind::CreditCard | PiiKind::Date)) {
            return Err(RedactError::InvalidOptions);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuleBudget {
    pub max_input_bytes: usize,
    pub max_detections: usize,
    pub max_candidate_bytes: usize,
    pub max_work: u64,
}
impl Default for RuleBudget {
    fn default() -> Self {
        Self { max_input_bytes: 1024 * 1024, max_detections: 4096,
            max_candidate_bytes: 4096, max_work: 128 * 1024 * 1024 }
    }
}

/// Price all scans, bounded sorting, and scalar-coordinate conversion before
/// running. The conservative work bound is not measured performance telemetry.
pub fn detect(source: &str, rules: &RuleSet, budget: RuleBudget) -> Result<Vec<Detection>, RedactError> {
    rules.validate()?;
    if budget.max_detections == 0 || budget.max_detections > MAX_DETECTIONS
        || !(64..=65_536).contains(&budget.max_candidate_bytes) { return Err(RedactError::InvalidOptions); }
    if source.len() > budget.max_input_bytes { return Err(RedactError::InputBudget); }
    let work = (source.len() as u64).checked_mul(64)
        .and_then(|n| n.checked_add((budget.max_detections as u64) * 64))
        .ok_or(RedactError::WorkBudget)?;
    if work > budget.max_work { return Err(RedactError::WorkBudget); }
    let mut scan = Scan { source, budget, hits: Vec::new() };
    if rules.enabled.contains(&PiiKind::Email) { scan.emails()?; }
    if rules.enabled.contains(&PiiKind::Url) { scan.urls()?; }
    if rules.enabled.contains(&PiiKind::IpAddress) { scan.ips()?; }
    if rules.enabled.contains(&PiiKind::Phone) || rules.enabled.contains(&PiiKind::CreditCard) { scan.numbers(rules)?; }
    if rules.enabled.contains(&PiiKind::Date) { scan.dates()?; }
    scan.finish()
}

struct Scan<'a> { source: &'a str, budget: RuleBudget, hits: Vec<(usize, usize, PiiKind, Detector)> }
impl Scan<'_> {
    fn push(&mut self, start: usize, end: usize, kind: PiiKind, detector: Detector) -> Result<(), RedactError> {
        if start >= end || self.source.get(start..end).is_none() { return Err(RedactError::InvalidSpan); }
        if self.hits.len() == self.budget.max_detections { return Err(RedactError::DetectionBudget); }
        self.hits.try_reserve(1).map_err(|_| RedactError::AllocationRefused)?;
        self.hits.push((start, end, kind, detector)); Ok(())
    }
    fn emails(&mut self) -> Result<(), RedactError> {
        let bytes = self.source.as_bytes();
        for (at, &byte) in bytes.iter().enumerate() {
            if byte != b'@' { continue; }
            let mut start = at;
            while start > 0 && local(bytes[start - 1]) { start -= 1; }
            let mut end = at + 1;
            while end < bytes.len() && domain_byte(bytes[end]) { end += 1; }
            // Sentence-ending punctuation is outside this detector's span.
            while end > at + 1 && bytes[end - 1] == b'.' { end -= 1; }
            while start < at && bytes[start] == b'\'' { start += 1; }
            if end - start > self.budget.max_candidate_bytes { return Err(RedactError::CandidateBudget); }
            let lhs = &bytes[start..at]; let rhs = &bytes[at + 1..end];
            if !lhs.is_empty() && lhs.len() <= 64 && lhs[0] != b'.' && lhs.last() != Some(&b'.')
                && !lhs.windows(2).any(|w| w == b"..") && dns(rhs, true) {
                self.push(start, end, PiiKind::Email, Detector::EmailAsciiV1)?;
            }
        }
        Ok(())
    }
    fn urls(&mut self) -> Result<(), RedactError> {
        let b = self.source.as_bytes(); let mut i = 0;
        while i < b.len() {
            let prefix = if b[i..].get(..8).is_some_and(|s| s.eq_ignore_ascii_case(b"https://")) { 8 }
                else if b[i..].get(..7).is_some_and(|s| s.eq_ignore_ascii_case(b"http://")) { 7 } else { i += 1; continue; };
            if i > 0 && (b[i-1].is_ascii_alphanumeric() || b[i-1] == b'_') { i += prefix; continue; }
            let start = i; i += prefix;
            while i < b.len() && !b[i].is_ascii_whitespace() && !matches!(b[i], 0..=31 | b'<' | b'>' | b'"' | b'\'') {
                // Unicode whitespace terminates a URL too, without normalizing.
                if b[i] >= 128 && self.source.is_char_boundary(i)
                    && self.source[i..].chars().next().is_some_and(char::is_whitespace) { break; }
                i += 1;
                if i - start > self.budget.max_candidate_bytes { return Err(RedactError::CandidateBudget); }
            }
            let mut end = i;
            while end > start + prefix && matches!(b[end-1], b'.' | b',' | b';' | b'!' | b')' | b']' | b'}') {
                // Preserve a closing IPv6 bracket when it is the authority.
                if b[end-1] == b']' && b[start+prefix] == b'[' { break; }
                end -= 1;
            }
            let tail = &self.source[start + prefix..end];
            let authority = tail.split(['/', '?', '#']).next().unwrap_or("");
            if http_authority(authority) { self.push(start, end, PiiKind::Url, Detector::UrlHttpV1)?; }
        }
        Ok(())
    }
    fn ips(&mut self) -> Result<(), RedactError> {
        let b = self.source.as_bytes(); let mut i = 0;
        while i < b.len() {
            if !ip_byte(b[i]) { i += 1; continue; }
            let start = i;
            while i < b.len() && ip_byte(b[i]) { i += 1; }
            let mut end = i;
            while end > start && b[end-1] == b'.' { end -= 1; }
            if end - start > self.budget.max_candidate_bytes {
                if b[start..end].contains(&b'.') || b[start..end].contains(&b':') { return Err(RedactError::CandidateBudget); }
                continue;
            }
            if start > 0 && (b[start-1].is_ascii_alphanumeric() || b[start-1] == b'_') { continue; }
            if i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') { continue; }
            if IpAddr::from_str(&self.source[start..end]).is_ok() {
                self.push(start, end, PiiKind::IpAddress, Detector::IpLiteralV1)?;
            }
        }
        Ok(())
    }
    fn numbers(&mut self, rules: &RuleSet) -> Result<(), RedactError> {
        let b = self.source.as_bytes(); let mut i = 0;
        while i < b.len() {
            if !b[i].is_ascii_digit() && b[i] != b'+' && b[i] != b'(' { i += 1; continue; }
            let start = i;
            while i < b.len() && (b[i].is_ascii_digit() || matches!(b[i], b'+' | b'(' | b')' | b' ' | b'-')) { i += 1; }
            let mut end = i;
            while end > start && matches!(b[end-1], b' ' | b'-' | b'(' | b'+') { end -= 1; }
            let candidate = &b[start..end];
            if candidate.len() > self.budget.max_candidate_bytes { return Err(RedactError::CandidateBudget); }
            if start > 0 && (b[start-1].is_ascii_alphanumeric() || b[start-1] == b'_') { continue; }
            if end < b.len() && (b[end].is_ascii_alphanumeric() || b[end] == b'_') { continue; }
            let count = candidate.iter().filter(|b| b.is_ascii_digit()).count();
            let card_shape = candidate.iter().all(|b| b.is_ascii_digit() || matches!(b, b' ' | b'-'));
            if rules.enabled.contains(&PiiKind::CreditCard) && card_shape && (13..=19).contains(&count) && luhn(candidate) {
                self.push(start, end, PiiKind::CreditCard, Detector::CardLuhnV1)?;
            }
            let plus_ok = candidate.iter().enumerate().all(|(n, b)| *b != b'+' || n == 0);
            let mut parens = 0_i32; let mut parens_ok = true;
            for &b in candidate {
                if b == b'(' { parens += 1; if parens > 1 { parens_ok = false; } }
                if b == b')' { parens -= 1; if parens < 0 { parens_ok = false; } }
            }
            let shaped = candidate.first() == Some(&b'+') || candidate.iter().any(|b| matches!(b, b' ' | b'-' | b'(' | b')'));
            if rules.enabled.contains(&PiiKind::Phone) && (7..=15).contains(&count) && plus_ok && parens_ok && parens == 0
                && (shaped || count >= 10) && !iso_date(candidate) {
                self.push(start, end, PiiKind::Phone, Detector::PhoneAsciiV1)?;
            }
        }
        Ok(())
    }
    fn dates(&mut self) -> Result<(), RedactError> {
        let b = self.source.as_bytes();
        for start in 0..b.len().saturating_sub(9) {
            let end = start + 10;
            if (start == 0 || !b[start-1].is_ascii_alphanumeric())
                && (end == b.len() || !b[end].is_ascii_alphanumeric()) && iso_date(&b[start..end]) {
                self.push(start, end, PiiKind::Date, Detector::DateIsoV1)?;
            }
        }
        Ok(())
    }
    fn finish(mut self) -> Result<Vec<Detection>, RedactError> {
        self.hits.sort_unstable(); self.hits.dedup();
        let mut endpoints = Vec::new();
        endpoints.try_reserve_exact(self.hits.len() * 2).map_err(|_| RedactError::AllocationRefused)?;
        for &(start, end, _, _) in &self.hits { endpoints.push((start, 0)); endpoints.push((end, 0)); }
        endpoints.sort_unstable(); endpoints.dedup();
        let mut next = 0;
        for (scalar, byte) in self.source.char_indices().map(|(b, _)| b).chain(std::iter::once(self.source.len())).enumerate() {
            if next < endpoints.len() && endpoints[next].0 == byte { endpoints[next].1 = scalar; next += 1; }
        }
        if next != endpoints.len() { return Err(RedactError::InvalidSpan); }
        let coordinate = |byte| endpoints.binary_search_by_key(&byte, |&(b, _)| b)
            .map(|i| endpoints[i].1).map_err(|_| RedactError::InvalidSpan);
        let mut detections = Vec::new();
        detections.try_reserve_exact(self.hits.len()).map_err(|_| RedactError::AllocationRefused)?;
        for (start, end, kind, detector) in self.hits {
            detections.push(Detection { kind, detector, span: VerifiedSourceSpan { byte_start: start, byte_end: end,
                scalar_start: coordinate(start)?, scalar_end: coordinate(end)? } });
        }
        Ok(detections)
    }
}
fn local(b: u8) -> bool { b.is_ascii_alphanumeric() || b".!#$%&'*+-/=?^_`{|}~".contains(&b) }
fn domain_byte(b: u8) -> bool { b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-') }
fn ip_byte(b: u8) -> bool { b.is_ascii_hexdigit() || matches!(b, b'.' | b':') }
fn dns(bytes: &[u8], require_dot: bool) -> bool {
    if bytes.is_empty() || bytes.len() > 253 || (require_dot && !bytes.contains(&b'.')) { return false; }
    bytes.split(|b| *b == b'.').all(|label| !label.is_empty() && label.len() <= 63
        && label[0].is_ascii_alphanumeric() && label[label.len()-1].is_ascii_alphanumeric()
        && label.iter().all(|b| b.is_ascii_alphanumeric() || *b == b'-'))
}
fn http_authority(authority: &str) -> bool {
    let host_port = authority.rsplit('@').next().unwrap_or("");
    if host_port.starts_with('[') {
        let Some(close) = host_port.find(']') else { return false; };
        return host_port[1..close].parse::<std::net::Ipv6Addr>().is_ok() && port(&host_port[close+1..]);
    }
    let (host, suffix) = host_port.find(':').map_or((host_port, ""), |i| (&host_port[..i], &host_port[i..]));
    dns(host.as_bytes(), false) && port(suffix)
}
fn port(suffix: &str) -> bool {
    suffix.is_empty() || suffix.strip_prefix(':').is_some_and(|p| !p.is_empty()
        && p.bytes().all(|b| b.is_ascii_digit()) && p.parse::<u16>().is_ok())
}
/// ASCII digits and optional spaces/hyphens; a check, not proof of issuance.
pub fn luhn(bytes: &[u8]) -> bool {
    let mut sum = 0_u32; let mut digits = 0;
    for &b in bytes.iter().rev() {
        if matches!(b, b' ' | b'-') { continue; }
        if !b.is_ascii_digit() || digits == 19 { return false; }
        let mut n = u32::from(b - b'0');
        if digits % 2 == 1 { n *= 2; if n > 9 { n -= 9; } }
        sum += n; digits += 1;
    }
    (13..=19).contains(&digits) && sum > 0 && sum % 10 == 0
}
fn iso_date(b: &[u8]) -> bool {
    if b.len() != 10 || b[4] != b'-' || b[7] != b'-'
        || b.iter().enumerate().any(|(i, b)| i != 4 && i != 7 && !b.is_ascii_digit()) { return false; }
    let year = b[..4].iter().fold(0_u32, |n, b| n * 10 + u32::from(b - b'0'));
    let month = (b[5]-b'0') * 10 + b[6]-b'0'; let day = (b[8]-b'0') * 10 + b[9]-b'0';
    let max = match month { 1 | 3 | 5 | 7 | 8 | 10 | 12 => 31, 4 | 6 | 9 | 11 => 30,
        2 if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) => 29, 2 => 28, _ => 0 };
    year != 0 && day > 0 && day <= max
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_offsets_are_original_unicode_scalar_coordinates() {
        let source = "é上海 a+b@example.org; +1 (212) 555-0199; https://example.org/a; 2001:db8::1; 4111-1111-1111-1111";
        let hits = detect(source, &RuleSet::default(), RuleBudget::default()).unwrap();
        for kind in [PiiKind::Email, PiiKind::Phone, PiiKind::Url, PiiKind::IpAddress, PiiKind::CreditCard] {
            assert!(hits.iter().any(|hit| hit.kind == kind), "{kind:?}");
        }
        for hit in hits { let s = hit.span;
            assert!(source.get(s.byte_start..s.byte_end).is_some());
            assert_eq!(s.scalar_start, source[..s.byte_start].chars().count());
            assert_eq!(s.scalar_end, source[..s.byte_end].chars().count());
        }
    }
    #[test]
    fn calendar_dates_are_validated_and_opt_in() {
        let source = "2024-02-29 2023-02-29 1900-02-29 2000-02-29";
        assert!(detect(source, &RuleSet::default(), RuleBudget::default()).unwrap().iter().all(|h| h.kind != PiiKind::Date));
        let rules = RuleSet { enabled: [PiiKind::Date].into_iter().collect() };
        let hits = detect(source, &rules, RuleBudget::default()).unwrap();
        assert_eq!(hits.len(), 2);
    }
    #[test]
    fn luhn_and_ip_rejections_are_explicit() {
        assert!(luhn(b"4111 1111 1111 1111")); assert!(!luhn(b"4111 1111 1111 1112")); assert!(!luhn(b"0000000000000000"));
        let rules = RuleSet { enabled: [PiiKind::IpAddress].into_iter().collect() };
        let hits = detect("127.0.0.1 999.0.0.1 [2001:db8::1] prefix127.0.0.1", &rules, RuleBudget::default()).unwrap();
        assert_eq!(hits.len(), 2);
    }
    #[test]
    fn no_partial_result_when_any_limit_exhausts() {
        let r = RuleSet::default(); let b = RuleBudget::default();
        assert_eq!(detect("a@b.org c@d.org", &r, RuleBudget { max_detections: 1, ..b }), Err(RedactError::DetectionBudget));
        assert_eq!(detect("hello", &r, RuleBudget { max_input_bytes: 4, ..b }), Err(RedactError::InputBudget));
        assert_eq!(detect("hello", &r, RuleBudget { max_work: 0, ..b }), Err(RedactError::WorkBudget));
        assert_eq!(detect(&format!("https://example.org/{}", "a".repeat(80)), &r,
            RuleBudget { max_candidate_bytes: 64, ..b }), Err(RedactError::CandidateBudget));
    }
    #[test]
    fn urls_include_private_paths_queries_and_credentials() {
        let s = "See HTTPS://u:p@example.org/private?q=secret. [https://[::1]:443/path]";
        let r = RuleSet { enabled: [PiiKind::Url].into_iter().collect() };
        let hits = detect(s, &r, RuleBudget::default()).unwrap();
        assert_eq!(&s[hits[0].span.byte_start..hits[0].span.byte_end], "HTTPS://u:p@example.org/private?q=secret");
        assert_eq!(hits.len(), 2);
    }
    #[test]
    fn unsupported_rule_types_refuse_instead_of_pretending_model_coverage() {
        assert_eq!(detect("Alice", &RuleSet { enabled: [PiiKind::Person].into_iter().collect() }, RuleBudget::default()), Err(RedactError::InvalidOptions));
    }
}
