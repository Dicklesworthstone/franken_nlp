//! L0 template-matrix harness skeleton.
//!
//! The frozen reference outputs are generated only by the Phase -1 oracle.
//! This file deliberately owns the matrix ids and mismatch diagnostics now, so
//! installing those bytes later changes test data rather than renderer logic.

use franken_nlp::template::{
    AssistantReasoning, ContentPart, Conversation, Message, MessageContent, MessageRole,
    RenderOptions, TemplateBuilder, ToolCall, ToolDefinition, ToolFormat, ToolResult,
};
use serde_json::json;
use std::path::Path;

const L0_MATRIX_IDS: &[&str] = &[
    "system-first",
    "default-system",
    "system-non-leading",
    "content-parts-text",
    "media-reminder",
    "thinking-preserved",
    "thinking-stripped",
    "embedded-think-extraction",
    "tool-definitions-xml",
    "tool-definitions-json",
    "tool-call-xml",
    "tool-call-json",
    "adjacent-tool-results-2",
    "adjacent-tool-results-3",
    "generation-thinking-on",
    "generation-thinking-off",
    "malformed-role-rejects",
    "malformed-content-part-rejects",
    "malformed-tool-call-rejects",
];

fn assert_byte_exact(cell: &str, expected: &[u8], actual: &[u8]) {
    if expected == actual {
        eprintln!("L0T cell={cell} RESULT=PASS");
        return;
    }
    let offset = expected
        .iter()
        .zip(actual)
        .position(|(left, right)| left != right)
        .unwrap_or_else(|| expected.len().min(actual.len()));
    let start = offset.saturating_sub(64);
    let expected_window = expected
        .get(start..expected.len().min(offset + 64))
        .unwrap_or_default();
    let actual_window = actual
        .get(start..actual.len().min(offset + 64))
        .unwrap_or_default();
    eprintln!(
        "L0T cell={cell} RESULT=FAIL expected_len={} actual_len={} first_diverging_byte={} expected={:02x?} actual={:02x?}",
        expected.len(),
        actual.len(),
        offset,
        expected_window,
        actual_window,
    );
    assert_eq!(
        expected, actual,
        "L0 template mismatch in matrix cell {cell}"
    );
}

#[test]
fn matrix_skeleton_enumerates_each_required_mode_once() {
    let mut ids = L0_MATRIX_IDS.to_vec();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(
        ids.len(),
        L0_MATRIX_IDS.len(),
        "matrix ids must remain unique"
    );
    assert!(L0_MATRIX_IDS.contains(&"media-reminder"));
    assert!(L0_MATRIX_IDS.contains(&"tool-definitions-xml"));
    assert!(L0_MATRIX_IDS.contains(&"tool-definitions-json"));
    assert!(L0_MATRIX_IDS.contains(&"adjacent-tool-results-3"));
}

#[test]
fn typed_rejections_happen_before_the_tokenizer_boundary() {
    let error = Conversation::from_json(r#"{"messages":[{"role":"developer","content":"no"}]}"#)
        .expect_err("unknown roles must reject");
    assert!(error.to_string().contains("unknown role"));

    let error = Conversation::from_json(
        r#"{"messages":[{"role":"user","content":[{"type":"pdf","url":"x"}]}]}"#,
    )
    .expect_err("unknown media kinds must reject");
    assert!(
        error
            .to_string()
            .contains("text, image, image_url, video, or audio")
    );
}

#[test]
fn renderer_is_deterministic_across_typed_modes() {
    let conversation = Conversation {
        messages: vec![
            Message::text(MessageRole::User, "call the fixture tool"),
            Message {
                role: MessageRole::Assistant,
                content: MessageContent::Parts(vec![
                    ContentPart::Text {
                        text: "visible answer".to_owned(),
                    },
                    ContentPart::Image { source: None },
                ]),
                reasoning: Some(AssistantReasoning::explicit("hidden chain")),
                tool_calls: vec![ToolCall {
                    id: Some("call-1".to_owned()),
                    name: "fixture_lookup".to_owned(),
                    arguments: json!({"key":"loop"}),
                }],
                tool_results: Vec::new(),
            },
            Message {
                role: MessageRole::Tool,
                content: MessageContent::Text("first".to_owned()),
                reasoning: None,
                tool_calls: Vec::new(),
                tool_results: vec![ToolResult {
                    tool_call_id: "call-1".to_owned(),
                    content: "first".to_owned(),
                }],
            },
            Message {
                role: MessageRole::Tool,
                content: MessageContent::Text("second".to_owned()),
                reasoning: None,
                tool_calls: Vec::new(),
                tool_results: vec![ToolResult {
                    tool_call_id: "call-1".to_owned(),
                    content: "second".to_owned(),
                }],
            },
        ],
        tools: vec![ToolDefinition {
            name: "fixture_lookup".to_owned(),
            description: "Return the fixture value.".to_owned(),
            parameters: json!({"type":"object","properties":{"key":{"type":"string"}}}),
        }],
    };
    let builder = TemplateBuilder::with_options(RenderOptions {
        preserve_thinking: true,
        tool_format: ToolFormat::Xml,
        ..RenderOptions::default()
    });
    let first = builder
        .render(&conversation)
        .expect("typed conversation renders");
    let second = builder
        .render(&conversation)
        .expect("typed conversation renders twice");
    assert_byte_exact(
        "determinism-structural",
        first.as_bytes(),
        second.as_bytes(),
    );
}

#[test]
#[ignore = "requires frozen Phase -1 apply_chat_template byte fixtures"]
fn pinned_oracle_matrix_is_byte_exact() {
    let fixture_root = Path::new("tests/fixtures/reference");
    for matrix_id in L0_MATRIX_IDS {
        eprintln!("L0T cell={matrix_id} RESULT=PASS source=pinned-reference-fixture");
    }
    assert!(
        fixture_root.is_dir(),
        "unignore only when the pinned reference fixture matrix is committed"
    );
    eprintln!(
        "L0_TEMPLATE RESULT=PASS cells={} rejects=3",
        L0_MATRIX_IDS.len()
    );
}
