use std::{env, process::Command};

use franken_nlp::error::{
    Cancellation, CancellationKind, EXIT_CODE_TABLE, ErrorCode, FnlpError, RunTerminal,
    StructuredTaskStatus, exit_code_table_json,
};
use serde_json::Value;

fn category_error(code: ErrorCode) -> FnlpError {
    match code {
        ErrorCode::Ok => panic!("success is represented by StructuredTaskStatus, not FnlpError"),
        ErrorCode::Generic => FnlpError::Generic { category: "test" },
        ErrorCode::Usage => FnlpError::Usage { category: "test" },
        ErrorCode::ModelNotFound => FnlpError::ModelNotFound { category: "test" },
        ErrorCode::InputDecodeOrParse => FnlpError::InputDecodeOrParse { category: "test" },
        ErrorCode::BudgetOrTimeout => FnlpError::BudgetOrTimeout { category: "test" },
        ErrorCode::Cancelled => FnlpError::Cancelled(Cancellation::new(CancellationKind::User)),
        ErrorCode::ArtifactIntegrityOrFormatOrVersion => {
            FnlpError::ArtifactIntegrityOrFormatOrVersion { category: "test" }
        }
        ErrorCode::SchemaOrRecipeCompile => FnlpError::SchemaOrRecipeCompile { category: "test" },
        ErrorCode::AdmissionOrResourceLimit => {
            FnlpError::AdmissionOrResourceLimit { category: "test" }
        }
        ErrorCode::StructuredTaskNoResult => FnlpError::StructuredTaskNoResult { category: "test" },
    }
}

#[test]
fn frozen_table_and_typed_variants_cover_every_exit_code() {
    assert_eq!(EXIT_CODE_TABLE.len(), ErrorCode::ALL.len());
    let encoded: Value = serde_json::from_slice(
        &exit_code_table_json().expect("canonical exit-code table must serialize"),
    )
    .expect("canonical exit-code table must be machine-readable JSON");
    assert_eq!(
        encoded
            .as_array()
            .expect("canonical exit-code JSON must be an array")
            .len(),
        ErrorCode::ALL.len()
    );
    for (expected, row) in ErrorCode::ALL.into_iter().zip(EXIT_CODE_TABLE) {
        assert_eq!(row.code, expected, "table must stay numerically ordered");
        assert!(!row.name.is_empty(), "every code needs a machine name");
        assert!(
            !row.description.is_empty(),
            "every code needs a machine description"
        );

        let actual = match expected {
            ErrorCode::Ok => StructuredTaskStatus::Completed.exit_code(),
            _ => category_error(expected).exit_code(),
        };
        assert_eq!(
            actual, expected,
            "typed construction must map to {expected:?}"
        );
        assert_eq!(actual.as_u8(), expected as u8);
    }

    assert_eq!(
        FnlpError::InvariantViolation { category: "test" }.exit_code(),
        ErrorCode::Generic,
        "invariant violations must not consume an application error code"
    );
}

#[test]
fn cancellation_matrix_is_exhaustive_and_preserves_supervision_policy() {
    let matrix = [
        (CancellationKind::User, ErrorCode::Cancelled),
        (CancellationKind::Timeout, ErrorCode::BudgetOrTimeout),
        (CancellationKind::Deadline, ErrorCode::BudgetOrTimeout),
        (CancellationKind::PollQuota, ErrorCode::BudgetOrTimeout),
        (CancellationKind::CostBudget, ErrorCode::BudgetOrTimeout),
        (CancellationKind::FailFast, ErrorCode::Generic),
        (CancellationKind::RaceLost, ErrorCode::Generic),
        (CancellationKind::ParentCancelled, ErrorCode::Cancelled),
        (
            CancellationKind::ResourceUnavailable,
            ErrorCode::AdmissionOrResourceLimit,
        ),
        (CancellationKind::Shutdown, ErrorCode::Cancelled),
        (CancellationKind::LinkedExit, ErrorCode::Generic),
    ];

    assert_eq!(matrix.len(), CancellationKind::ALL.len());
    for (kind, expected) in matrix {
        let cancellation = Cancellation::new(kind);
        assert_eq!(
            cancellation.exit_code(),
            expected,
            "wrong policy for {kind:?}"
        );
        assert_eq!(cancellation.kind(), kind);
    }

    for kind in [CancellationKind::FailFast, CancellationKind::LinkedExit] {
        let cancellation =
            Cancellation::with_underlying(kind, FnlpError::ModelNotFound { category: "test" });
        assert_eq!(cancellation.exit_code(), ErrorCode::ModelNotFound);
    }

    let race_lost = Cancellation::new(CancellationKind::RaceLost);
    assert!(race_lost.crashpack_required());
    assert!(
        race_lost
            .diagnostic_marker()
            .contains("crashpack_required=true"),
        "root RaceLost must leave an explicit crashpack diagnostic marker"
    );
    eprintln!(
        "EXIT_CODES RESULT=PASS cancelkinds_covered={}/11",
        matrix.len()
    );
}

#[test]
fn abstention_is_success_but_no_result_is_not() {
    let abstained = RunTerminal::Success(StructuredTaskStatus::Abstained);
    assert_eq!(abstained.exit_code(), ErrorCode::Ok);
    assert_eq!(StructuredTaskStatus::Abstained.status(), "abstained");
    assert_eq!(abstained.robot_event_name(), "run_complete");

    let no_result = RunTerminal::Error(FnlpError::StructuredTaskNoResult {
        category: "missing required source field",
    });
    assert_eq!(no_result.exit_code(), ErrorCode::StructuredTaskNoResult);
    assert_eq!(no_result.robot_event_name(), "run_error");
}

const PANIC_CHILD_ENV: &str = "FNLP_EXIT_CODE_PANIC_CHILD";

#[test]
fn injected_panic_is_a_run_error_with_a_real_nonzero_process_status() {
    if env::var_os(PANIC_CHILD_ENV).is_some() {
        let payload = std::panic::catch_unwind(|| panic!("injected panic for exit-code harness"))
            .expect_err("the child harness must inject a panic");
        let category = if payload.is::<&str>() {
            "injected_panic"
        } else {
            "non_string_panic"
        };
        let terminal = RunTerminal::Panicked(franken_nlp::error::PanicFailure::new(category));
        eprintln!(
            "event={} crashpack_required={}",
            terminal.robot_event_name(),
            terminal.crashpack_required()
        );
        std::process::exit(terminal.exit_code().as_u8().into());
    }

    let output = Command::new(env::current_exe().expect("test executable path must exist"))
        .args([
            "--exact",
            "injected_panic_is_a_run_error_with_a_real_nonzero_process_status",
            "--nocapture",
        ])
        .env(PANIC_CHILD_ENV, "1")
        .output()
        .expect("panic child process must start");

    assert_eq!(
        output.status.code(),
        Some(ErrorCode::Generic.as_u8().into())
    );
    let stderr = String::from_utf8(output.stderr).expect("test diagnostics must be UTF-8");
    assert!(stderr.contains("event=run_error"));
    assert!(!stderr.contains("event=doc_error"));
    assert!(stderr.contains("crashpack_required=true"));
}
