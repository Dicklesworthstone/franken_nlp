//! G0-01 fixture oracle for range-authenticated HTTPS asset transfers.
//!
//! This is deliberately a transport-neutral fixture matrix: it defines the
//! required request/response and redirect policy before `fnlp pull` selects a
//! concrete asupersync HTTP client path.  It is not an HTTPS compatibility
//! claim; `ADR-G0-01-https-range.md` remains BLOCKED until a configured client
//! is exercised against the same matrix and its raw evidence is archived.

use std::collections::BTreeMap;

const SEED: u64 = 0x4730_3031;

#[derive(Clone, Debug)]
struct RangeResponse {
    status: u16,
    content_range: Option<&'static str>,
    content_encoding: Option<&'static str>,
    body_len: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RangeViolation {
    Status,
    MissingContentRange,
    InvalidContentRange,
    UnexpectedContentEncoding,
    BodyLength,
}

fn parse_content_range(value: &str) -> Option<(u64, u64, u64)> {
    let value = value.strip_prefix("bytes ")?;
    let (range, total) = value.split_once('/')?;
    let (start, end) = range.split_once('-')?;
    let start = start.parse().ok()?;
    let end = end.parse().ok()?;
    let total = total.parse().ok()?;
    (start <= end && end < total).then_some((start, end, total))
}

fn validate_range_response(
    requested_start: u64,
    requested_end: u64,
    response: &RangeResponse,
) -> Result<(), RangeViolation> {
    if response.status != 206 {
        return Err(RangeViolation::Status);
    }
    if response
        .content_encoding
        .is_some_and(|value| value != "identity")
    {
        return Err(RangeViolation::UnexpectedContentEncoding);
    }
    let (start, end, _) = response
        .content_range
        .and_then(parse_content_range)
        .ok_or(RangeViolation::MissingContentRange)?;
    if start != requested_start || end != requested_end {
        return Err(RangeViolation::InvalidContentRange);
    }
    let expected_len = end
        .checked_sub(start)
        .and_then(|length| length.checked_add(1))
        .ok_or(RangeViolation::InvalidContentRange)?;
    if u64::try_from(response.body_len).ok() != Some(expected_len) {
        return Err(RangeViolation::BodyLength);
    }
    Ok(())
}

fn origin(url: &str) -> Option<(&str, &str)> {
    let (scheme, rest) = url.split_once("://")?;
    let host = rest.split('/').next()?;
    (!scheme.is_empty() && !host.is_empty()).then_some((scheme, host))
}

fn permit_redirect(from: &str, to: &str, has_credentials: bool) -> Result<bool, &'static str> {
    let (from_scheme, from_host) = origin(from).ok_or("invalid source URL")?;
    let (to_scheme, to_host) = origin(to).ok_or("invalid redirect URL")?;
    if to_scheme != "https" {
        return Err("redirect downgraded transport");
    }
    let same_origin = from_scheme == to_scheme && from_host == to_host;
    Ok(has_credentials && !same_origin)
}

fn selected_proxy<'a>(url: &str, environment: &'a BTreeMap<&str, &str>) -> Option<&'a str> {
    let (_, host) = origin(url)?;
    let no_proxy = environment.get("NO_PROXY").copied().unwrap_or_default();
    if no_proxy
        .split(',')
        .map(str::trim)
        .any(|entry| entry == "*" || entry == host)
    {
        return None;
    }
    environment
        .get("HTTPS_PROXY")
        .or_else(|| environment.get("HTTP_PROXY"))
        .copied()
}

fn log_case(id: &str, result: &str) {
    println!("G0_PROBE1 case={id} RESULT={result} seed={SEED}");
}

#[test]
fn fixture_http_matrix_rejects_unauthenticated_range_semantics() {
    let valid = RangeResponse {
        status: 206,
        content_range: Some("bytes 10-19/100"),
        content_encoding: Some("identity"),
        body_len: 10,
    };
    assert_eq!(validate_range_response(10, 19, &valid), Ok(()));
    log_case("206-content-range-identity", "PASS");

    let wrong_range = RangeResponse {
        content_range: Some("bytes 11-20/100"),
        ..valid.clone()
    };
    assert_eq!(
        validate_range_response(10, 19, &wrong_range),
        Err(RangeViolation::InvalidContentRange)
    );
    log_case("wrong-content-range", "PASS");

    let gzip = RangeResponse {
        content_encoding: Some("gzip"),
        ..valid.clone()
    };
    assert_eq!(
        validate_range_response(10, 19, &gzip),
        Err(RangeViolation::UnexpectedContentEncoding)
    );
    log_case("gzip-injected-body", "PASS");

    let full_response = RangeResponse {
        status: 200,
        ..valid.clone()
    };
    assert_eq!(
        validate_range_response(10, 19, &full_response),
        Err(RangeViolation::Status)
    );
    log_case("unexpected-200", "PASS");

    assert_eq!(
        permit_redirect(
            "https://origin.example/part",
            "https://mirror.example/part",
            true,
        ),
        Ok(true),
        "cross-origin redirects must strip credentials"
    );
    assert_eq!(
        permit_redirect(
            "https://origin.example/part",
            "http://origin.example/part",
            false,
        ),
        Err("redirect downgraded transport")
    );
    log_case("redirect-credential-strip-and-https-only", "PASS");

    let environment = BTreeMap::from([
        ("HTTPS_PROXY", "http://proxy.invalid:8080"),
        ("NO_PROXY", "direct.example"),
    ]);
    assert_eq!(
        selected_proxy("https://direct.example/model", &environment),
        None
    );
    assert_eq!(
        selected_proxy("https://mirror.example/model", &environment),
        Some("http://proxy.invalid:8080")
    );
    log_case("proxy-no-proxy-selection", "PASS");

    println!("G0_PROBE1 RESULT=PASS cases=6 seed={SEED} authority=fixture-oracle-only");
}
