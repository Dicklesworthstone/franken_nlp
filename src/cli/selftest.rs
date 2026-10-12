//! Model-free `robot selftest` command. Its namespaced v1 event protocol is
//! separate from the existing v3 task/health schema, which remains unchanged.
use std::{io::{self, Write}, process::ExitCode};
use clap::{Arg, ArgAction, ArgMatches, Command};
use serde::Serialize;
use crate::{canonjson, error::ErrorCode, native_engine::{dispatch::{self, KernelTier}, selftest::{
    self, CaseReport, SelftestError, SelftestReport, SelftestRequest, TierStatus, Verdict,
    CASES_PER_TIER, SELFTEST_SCHEMA_VERSION, SELFTEST_SUITE_VERSION}}};

mod schema;
#[cfg(test)]
mod tests;
const PROTOCOL: &str = "fnlp.cpu-selftest";

pub(super) fn definition() -> Command {
    Command::new("selftest")
        .about("Execute finite synthetic CPU kernel checks against an independent i64 reference; no model required")
        .arg(Arg::new("tier").long("tier").value_name("KERNEL")
            .value_parser(KernelTier::ALL.map(|tier| tier.id()))
            .help("Check only this exact tier; unsupported explicit requests exit nonzero"))
        .arg(Arg::new("details").long("details").action(ArgAction::SetTrue)
            .help("Emit every case checksum, not only tier summaries and failing cases"))
        .arg(Arg::new("schema").long("schema").action(ArgAction::SetTrue)
            .conflicts_with_all(["tier", "details"])
            .help("Emit the selftest event schema without executing kernels"))
}

#[derive(Serialize)]
struct Envelope<'a, T: Serialize> {
    schema_version: u32, protocol: &'static str, event: &'static str, data: &'a T,
}
#[derive(Serialize)]
struct TierSummary {
    tier: KernelTier, status: TierStatus, cases_run: usize, cases_passed: usize,
}
#[derive(Serialize)]
struct CaseData<'a> {
    tier: KernelTier,
    #[serde(flatten)]
    case: &'a CaseReport,
}

fn line(output: &mut impl Write, value: &impl Serialize) -> io::Result<()> {
    let bytes = canonjson::canonical_bytes(value).map_err(io::Error::other)?;
    output.write_all(&bytes)?;
    output.write_all(b"\n")
}
fn event(output: &mut impl Write, name: &'static str, data: &impl Serialize) -> io::Result<()> {
    line(output, &Envelope { schema_version: SELFTEST_SCHEMA_VERSION, protocol: PROTOCOL, event: name, data })
}
fn emit_report(report: &SelftestReport, details: bool, output: &mut impl Write) -> io::Result<()> {
    for tier in &report.tiers {
        for case in &tier.cases {
            if details || case.failure.is_some() {
                event(output, "selftest_case", &CaseData { tier: tier.tier, case })?;
            }
        }
        event(output, "selftest_tier", &TierSummary { tier: tier.tier, status: tier.status,
            cases_run: tier.cases_run, cases_passed: tier.cases_passed })?;
    }
    event(output, "selftest_complete", &serde_json::json!({
        "suite_version": report.suite_version, "scope": report.scope,
        "requested_tier": report.requested_tier, "verdict": report.verdict,
        "tiers_considered": report.tiers.len(), "tiers_skipped": report.tiers_skipped,
        "cases_run": report.cases_run, "cases_passed": report.cases_passed,
        "cases_per_tier": report.cases_per_tier,
    }))?;
    output.flush()
}
fn verdict_exit(verdict: Verdict) -> ExitCode {
    match verdict {
        Verdict::Pass => ExitCode::SUCCESS,
        Verdict::Fail => ErrorCode::Generic.as_process_exit(),
        Verdict::Unsupported => ErrorCode::Usage.as_process_exit(),
    }
}

pub(super) fn run(matches: &ArgMatches, output: &mut impl Write, errors: &mut impl Write) -> ExitCode {
    if matches.get_flag("schema") {
        return match line(output, &schema::document()).and_then(|()| output.flush()) {
            Ok(()) => ExitCode::SUCCESS,
            Err(_) => ErrorCode::Generic.as_process_exit(),
        };
    }
    let tier = match matches.get_one::<String>("tier") {
        Some(value) => match value.parse::<KernelTier>() {
            Ok(tier) => Some(tier),
            Err(_) => return ErrorCode::Usage.as_process_exit(),
        },
        None => None,
    };
    // A disconnected consumer at startup prevents the diagnostic work itself.
    if event(output, "selftest_start", &serde_json::json!({
        "suite_version": SELFTEST_SUITE_VERSION, "requested_tier": tier,
        "cases_per_tier": CASES_PER_TIER, "detected_features": dispatch::host_features(),
    })).and_then(|()| output.flush()).is_err() {
        let _ = writeln!(errors, "fnlp robot selftest: output unavailable");
        return ErrorCode::Generic.as_process_exit();
    }
    match selftest::run(SelftestRequest { tier }) {
        Ok(report) => {
            if emit_report(&report, matches.get_flag("details"), output).is_err() {
                let _ = writeln!(errors, "fnlp robot selftest: output incomplete");
                return ErrorCode::Generic.as_process_exit();
            }
            verdict_exit(report.verdict)
        }
        Err(error) => {
            let code = match error {
                SelftestError::Allocation => "allocation_refused",
                SelftestError::OracleOverflow => "oracle_overflow",
            };
            let _ = event(output, "selftest_error", &serde_json::json!({"code": code}));
            let _ = output.flush();
            let _ = writeln!(errors, "fnlp robot selftest: {error}");
            ErrorCode::Generic.as_process_exit()
        }
    }
}
