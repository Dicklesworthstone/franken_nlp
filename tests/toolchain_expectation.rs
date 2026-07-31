#![feature(portable_simd)]

use serde_json::Value;
use std::simd::Simd;

const EXPECTED: &str = include_str!("../ci/toolchain.expected.json");
const TOOLCHAIN: &str = include_str!("../rust-toolchain.toml");
const RELEASE_TARGETS: [&str; 5] = [
    "aarch64-apple-darwin",
    "x86_64-apple-darwin",
    "x86_64-unknown-linux-gnu",
    "aarch64-unknown-linux-gnu",
    "x86_64-pc-windows-msvc",
];

#[test]
fn expectation_and_toolchain_pin_agree() {
    let expectation: Value = serde_json::from_str(EXPECTED).expect("expectation JSON must parse");
    let mut drifts = Vec::new();

    let channel = expectation["channel"]
        .as_str()
        .expect("expectation channel must be a string");
    if !is_dated_nightly(channel) {
        drifts.push(format!("channel expected dated nightly observed={channel}"));
    }
    if toolchain_value("channel") != Some(channel) {
        drifts.push(format!(
            "channel expected={channel} observed={}",
            toolchain_value("channel").unwrap_or("<missing>")
        ));
    }
    if toolchain_value("profile") != Some("minimal") {
        drifts.push("profile expected=minimal".to_owned());
    }

    for component in expectation["required_components"]
        .as_array()
        .expect("required_components must be an array")
        .iter()
        .map(|value| value.as_str().expect("component must be a string"))
    {
        if !toolchain_array("components").contains(&component) {
            drifts.push(format!("missing toolchain component={component}"));
        }
    }

    let expected_targets: Vec<&str> = expectation["release_targets"]
        .as_array()
        .expect("release_targets must be an array")
        .iter()
        .map(|target| {
            target["triple"]
                .as_str()
                .expect("target triple must be a string")
        })
        .collect();
    if expected_targets != RELEASE_TARGETS {
        drifts.push(format!(
            "release target order expected={RELEASE_TARGETS:?} observed={expected_targets:?}"
        ));
    }
    for target in RELEASE_TARGETS {
        if !toolchain_array("targets").contains(&target) {
            drifts.push(format!("missing toolchain target={target}"));
        }
    }

    let compiler = expectation["compiler_identity"]
        .as_object()
        .expect("compiler_identity must be an object");
    for field in ["release", "commit_hash", "commit_date", "llvm_version"] {
        if compiler
            .get(field)
            .and_then(Value::as_str)
            .is_none_or(str::is_empty)
        {
            drifts.push(format!("compiler_identity.{field} must be non-empty"));
        }
    }
    for target in expectation["release_targets"]
        .as_array()
        .expect("release_targets must be an array")
    {
        for field in ["enabled_target_features", "required_compiler_features"] {
            if target[field]
                .as_array()
                .is_none_or(|items| items.is_empty())
            {
                drifts.push(format!(
                    "target {} missing {field}",
                    target["triple"].as_str().unwrap_or("<missing>")
                ));
            }
        }
    }
    if !expectation["required_language_features"]
        .as_array()
        .expect("required_language_features must be an array")
        .iter()
        .any(|feature| feature.as_str() == Some("portable_simd"))
    {
        drifts.push("required_language_features missing portable_simd".to_owned());
    }

    report(drifts);
}

#[test]
fn floating_nightly_channel_is_rejected() {
    assert!(!is_dated_nightly("nightly"));
    assert!(!is_dated_nightly("nightly-2026-7-20"));
    assert!(!is_dated_nightly("nightly-2026-07-2"));
    assert!(!is_dated_nightly("stable-2026-07-20"));
    assert!(is_dated_nightly("nightly-2026-07-20"));
}

#[test]
fn portable_simd_probe_is_compiled() {
    let vector = Simd::<i8, 16>::splat(7);
    assert_eq!(vector.to_array(), [7; 16]);
}

fn toolchain_value(key: &str) -> Option<&str> {
    TOOLCHAIN.lines().find_map(|line| {
        let (observed_key, observed_value) = line.split_once('=')?;
        if observed_key.trim() == key {
            Some(observed_value.trim().trim_matches('"'))
        } else {
            None
        }
    })
}

fn toolchain_array(key: &str) -> Vec<&str> {
    let mut values = Vec::new();
    let mut in_array = false;
    for line in TOOLCHAIN.lines() {
        let trimmed = line.trim();
        if !in_array {
            if trimmed.starts_with(key) && trimmed.ends_with('[') {
                in_array = true;
            }
            continue;
        }
        if trimmed == "]" {
            break;
        }
        if let Some(value) = trimmed.strip_suffix(',') {
            values.push(value.trim().trim_matches('"'));
        }
    }
    values
}

fn is_dated_nightly(channel: &str) -> bool {
    let Some(date) = channel.strip_prefix("nightly-") else {
        return false;
    };
    let mut pieces = date.split('-');
    let Some(year) = pieces.next() else {
        return false;
    };
    let Some(month) = pieces.next() else {
        return false;
    };
    let Some(day) = pieces.next() else {
        return false;
    };
    pieces.next().is_none()
        && year.len() == 4
        && month.len() == 2
        && day.len() == 2
        && year.bytes().all(|byte| byte.is_ascii_digit())
        && month.bytes().all(|byte| byte.is_ascii_digit())
        && day.bytes().all(|byte| byte.is_ascii_digit())
}

fn report(drifts: Vec<String>) {
    if drifts.is_empty() {
        println!("TOOLCHAIN RESULT=PASS drift=none");
        return;
    }
    for drift in &drifts {
        println!("TOOLCHAIN drift={drift}");
    }
    println!("TOOLCHAIN RESULT=FAIL drift={}", drifts[0]);
    panic!("toolchain expectation drift: {}", drifts.join("; "));
}
