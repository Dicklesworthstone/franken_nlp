use std::{env, ffi::OsStr, fs, path::PathBuf};

use franken_nlp::robot::{self, RobotCommand, RobotEvent, RobotEventType};
use serde_json::Value;

const GOLDEN_PATH: &str = "tests/fixtures/robot_schema.golden.json";

fn frozen_schema() -> Value {
    serde_json::from_slice(include_bytes!("fixtures/robot_schema.golden.json"))
        .expect("frozen robot schema must remain valid JSON")
}

fn update_golden_if_explicitly_requested(generated: &[u8]) {
    if env::var_os("UPDATE_GOLDENS").as_deref() != Some(OsStr::new("1")) {
        return;
    }
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(GOLDEN_PATH);
    let existing = fs::read(&path).expect("frozen schema golden must be readable");
    if existing != generated {
        fs::write(&path, generated).expect("UPDATE_GOLDENS=1 must update the requested golden");
        panic!(
            "UPDATE_GOLDENS=1 updated {GOLDEN_PATH}; inspect the mandatory diff, then rerun without UPDATE_GOLDENS"
        );
    }
}

fn validate_emitted_event(schema: &Value, event: &Value) {
    let properties = schema["properties"]
        .as_object()
        .expect("frozen schema properties must be an object");
    let event_object = event
        .as_object()
        .expect("emitted robot line must be a JSON object");
    for field in event_object.keys() {
        assert!(
            properties.contains_key(field),
            "schema validation failure event={event} violating_field=/{field}"
        );
    }

    assert_eq!(
        event["schema_version"],
        Value::from(robot::ROBOT_SCHEMA_VERSION)
    );
    let event_name = event["event"]
        .as_str()
        .expect("robot event must carry its event name");
    let supported = schema["properties"]["event"]["enum"]
        .as_array()
        .expect("frozen event enum must be an array");
    assert!(
        supported
            .iter()
            .any(|candidate| candidate.as_str() == Some(event_name)),
        "schema validation failure event={event} violating_field=/event"
    );

    let request_seq_events = schema["x_fnlp_robot"]["request_seq_events"]
        .as_array()
        .expect("frozen request-sequence event list must be an array");
    if request_seq_events
        .iter()
        .any(|candidate| candidate.as_str() == Some(event_name))
    {
        assert!(
            event.get("request_seq").is_some(),
            "schema validation failure event={event} violating_field=/request_seq"
        );
    }
    if event_name == "doc_error" {
        assert!(
            event.get("input_line").is_some(),
            "schema validation failure event={event} violating_field=/input_line"
        );
    }
}

#[test]
fn emitted_events_match_the_frozen_contract() {
    let schema = frozen_schema();
    let mut events = 0;
    for event_type in RobotEventType::ALL {
        let event = RobotEvent::skeleton(event_type);
        let mut stdout = Vec::new();
        robot::write_event(&mut stdout, &event).expect("typed event must emit");
        assert!(
            stdout.ends_with(b"\n"),
            "robot output must be line-oriented"
        );
        assert_eq!(
            stdout[..stdout.len() - 1]
                .iter()
                .filter(|&&byte| byte == b'\n')
                .count(),
            0,
            "one event must occupy exactly one NDJSON line"
        );
        let emitted: Value = serde_json::from_slice(&stdout[..stdout.len() - 1])
            .expect("typed event emission must be valid JSON");
        validate_emitted_event(&schema, &emitted);
        events += 1;
    }
    eprintln!("ROBOT_CONTRACT RESULT=PASS events={events}");
}

#[test]
fn request_sequences_and_error_locations_are_present_without_raw_input_echo() {
    for event_type in [
        RobotEventType::Doc,
        RobotEventType::DocError,
        RobotEventType::Token,
        RobotEventType::RunComplete,
    ] {
        let event = RobotEvent::skeleton(event_type);
        let mut stdout = Vec::new();
        robot::write_event(&mut stdout, &event).expect("per-request event must emit");
        let emitted: Value = serde_json::from_slice(&stdout).expect("event must be valid JSON");
        assert!(
            emitted.get("request_seq").is_some(),
            "{} must echo request_seq",
            event_type.as_str()
        );
        if event_type == RobotEventType::DocError {
            assert_eq!(emitted["input_line"], Value::from(1));
            assert_eq!(emitted["byte_offset"], Value::from(0));
            assert_eq!(emitted["json_path"], Value::from("/synthetic"));
            assert!(emitted.get("raw_input").is_none());
        }
    }

    let run_error = RobotEvent::new(RobotEventType::RunError).with_input_location(
        robot::InputLocation::new(7, Some(11), Some("/request/document".to_owned())),
    );
    let mut stdout = Vec::new();
    robot::write_event(&mut stdout, &run_error).expect("located run_error must emit");
    let emitted: Value = serde_json::from_slice(&stdout).expect("run_error must be valid JSON");
    assert_eq!(emitted["input_line"], Value::from(7));
    assert_eq!(emitted["byte_offset"], Value::from(11));
    assert_eq!(emitted["json_path"], Value::from("/request/document"));
}

#[test]
fn schema_and_unpopulated_commands_are_data_only_and_golden_frozen() {
    let generated = robot::schema_json_bytes();
    update_golden_if_explicitly_requested(&generated);
    assert_eq!(
        generated.as_slice(),
        include_bytes!("fixtures/robot_schema.golden.json"),
        "robot schema changed; use UPDATE_GOLDENS=1, inspect the diff, and commit the reviewed golden"
    );

    for command in [
        RobotCommand::Schema,
        RobotCommand::Health,
        RobotCommand::Backends,
    ] {
        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        robot::write_command(&mut stdout, &mut stderr, command).expect("robot command must emit");
        assert!(
            stderr.is_empty(),
            "successful robot command must not decorate stderr"
        );
        assert!(
            !stdout.windows(2).any(|window| window == b"\x1b["),
            "robot data must not contain terminal decoration"
        );
        let document: Value =
            serde_json::from_slice(&stdout).expect("robot command must emit JSON");
        if command == RobotCommand::Schema {
            assert_eq!(document, frozen_schema());
        } else {
            assert_eq!(
                document["schema_version"],
                Value::from(robot::ROBOT_SCHEMA_VERSION)
            );
            if command == RobotCommand::Backends {
                assert_eq!(document["status"], Value::from("populated"));
                assert_eq!(document["kind"], Value::from("robot_backends"));
                assert!(document["backends"]["detected_features"].is_object());
                assert!(document["backends"]["registry"].is_array());
                let selections = document["backends"]["selections"]
                    .as_array()
                    .expect("backends report must enumerate every fixed dispatch key");
                assert_eq!(selections.len(), 3 * 3 * 3 * 5);
                for selection in selections {
                    assert_eq!(selection["tier"], Value::from("scalar"));
                    assert_eq!(
                        selection["provenance"]["detail"],
                        Value::from("no measurement — conservative default")
                    );
                }
            } else {
                assert_eq!(document["kind"], Value::from("robot_health"));
                assert_eq!(
                    document["capabilities"]["status"],
                    Value::from("unpopulated")
                );
                let inventory = &document["thread_inventory"];
                let status = inventory["status"]
                    .as_str()
                    .expect("health inventory must name its state");
                assert!(matches!(status, "not_installed" | "configured"));
                if status == "configured" {
                    for field in [
                        "runtime_preset",
                        "runtime_workers",
                        "blocking_coordinators",
                        "scoped_cpu_children_per_coordinator",
                        "helper_threads",
                        "total_runnable_threads",
                        "thread_ceiling",
                        "runtime_binding",
                        "active_engine_leases",
                        "outstanding_pool_closures",
                        "cancelled_wrapper_closures",
                        "deadline_check_interval_millis",
                        "checkpoint_timeout_millis",
                        "cancel_attribution_max_depth",
                        "cancel_attribution_max_memory_bytes",
                    ] {
                        assert!(
                            inventory.get(field).is_some(),
                            "configured health must include thread inventory field {field}"
                        );
                    }
                    assert!(
                        inventory["total_runnable_threads"].as_u64()
                            <= inventory["thread_ceiling"].as_u64(),
                        "health must not report an envelope above its fixed ceiling"
                    );
                    assert!(
                        inventory["cancelled_wrapper_closures"].as_u64()
                            <= inventory["outstanding_pool_closures"].as_u64(),
                        "cancelled wrapper count is a subset of outstanding pool closures"
                    );
                    for field in [
                        "deadline_check_interval_millis",
                        "checkpoint_timeout_millis",
                        "cancel_attribution_max_depth",
                        "cancel_attribution_max_memory_bytes",
                    ] {
                        assert!(
                            inventory[field].as_u64().is_some_and(|value| value > 0),
                            "configured health must report a finite nonzero guardrail in {field}"
                        );
                    }
                }
            }
        }
    }
}
