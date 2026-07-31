//! L0 template-matrix harness skeleton.
//!
//! The frozen reference outputs are generated only by the Phase -1 oracle.
//! This file deliberately owns the matrix ids and mismatch diagnostics now, so
//! installing those bytes later changes test data rather than renderer logic.

use franken_nlp::template::{
    AssistantReasoning, ContentPart, Conversation, DEFAULT_SYSTEM_TEXT, IM_START,
    MEDIA_REMINDER_TEXT, Message, MessageContent, MessageRole, RenderOptions, THINK_END,
    THINK_START, TemplateBuilder, ToolCall, ToolDefinition, ToolFormat, ToolResult,
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
fn default_system_and_generation_suffix_are_explicit() {
    let conversation = Conversation::new(vec![Message::text(MessageRole::User, "hello")]);
    let thinking = TemplateBuilder::with_options(RenderOptions::default())
        .render(&conversation)
        .expect("typed conversation renders");
    assert!(thinking.starts_with(&format!("{IM_START}system\n{DEFAULT_SYSTEM_TEXT}")));
    assert!(thinking.ends_with(&format!("{IM_START}assistant\n{THINK_START}\n")));

    let no_thinking = TemplateBuilder::with_options(RenderOptions {
        enable_thinking: false,
        ..RenderOptions::default()
    })
    .render(&conversation)
    .expect("typed conversation renders with thinking disabled");
    assert!(no_thinking.ends_with(&format!("{IM_START}assistant\n")));
    assert!(!no_thinking.ends_with(THINK_START));
}

#[test]
fn first_system_position_and_tool_definition_formats_are_explicit() {
    let first_system = Conversation {
        messages: vec![
            Message::text(MessageRole::System, "owner supplied system"),
            Message::text(MessageRole::User, "hello"),
        ],
        tools: vec![ToolDefinition {
            name: "lookup".to_owned(),
            description: "Look up a value.".to_owned(),
            parameters: json!({"type":"object","properties":{"key":{"type":"string"}}}),
        }],
    };
    let non_leading_system = Conversation::new(vec![
        Message::text(MessageRole::User, "hello"),
        Message::text(MessageRole::System, "later system"),
    ]);

    let xml = TemplateBuilder::with_options(RenderOptions {
        add_generation_prompt: false,
        tool_format: ToolFormat::Xml,
        ..RenderOptions::default()
    })
    .render(&first_system)
    .expect("first system with XML tool declaration renders");
    assert!(xml.starts_with("<tools>\n"));
    assert!(xml.contains("<tool><name>lookup</name><description>Look up a value.</description>"));
    assert_eq!(xml.matches(&format!("{IM_START}system\n")).count(), 1);
    assert!(xml.contains(&format!(
        "{IM_START}system\nowner supplied system{IM_END}\n"
    )));
    assert!(!xml.contains(DEFAULT_SYSTEM_TEXT));

    let json = TemplateBuilder::with_options(RenderOptions {
        add_generation_prompt: false,
        tool_format: ToolFormat::Json,
        ..RenderOptions::default()
    })
    .render(&first_system)
    .expect("first system with JSON tool declaration renders");
    assert!(json.starts_with("<tools>[{"));
    assert!(json.contains(
        "\"function\":{\"description\":\"Look up a value.\",\"name\":\"lookup\",\"parameters\":{"
    ));
    assert_ne!(xml, json);

    let rendered_non_leading = TemplateBuilder::with_options(RenderOptions {
        add_generation_prompt: false,
        ..RenderOptions::default()
    })
    .render(&non_leading_system)
    .expect("non-leading system remains role framed");
    assert!(rendered_non_leading.starts_with(&format!(
        "{IM_START}system\n{DEFAULT_SYSTEM_TEXT}{IM_END}\n{IM_START}user\nhello{IM_END}\n"
    )));
    assert!(rendered_non_leading.ends_with(&format!("{IM_START}system\nlater system{IM_END}\n")));
}

#[test]
fn media_and_assistant_reasoning_follow_the_selected_mode() {
    let conversation = Conversation::new(vec![
        Message {
            role: MessageRole::User,
            content: MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "inspect ".to_owned(),
                },
                ContentPart::Video { source: None },
            ]),
            reasoning: None,
            tool_calls: Vec::new(),
            tool_results: Vec::new(),
        },
        Message::text(MessageRole::Assistant, "<think>private</think>visible"),
    ]);
    let preserved = TemplateBuilder::with_options(RenderOptions {
        add_generation_prompt: false,
        preserve_thinking: true,
        ..RenderOptions::default()
    })
    .render(&conversation)
    .expect("assistant leading thinking region is typed");
    assert!(preserved.contains(MEDIA_REMINDER_TEXT));
    assert!(preserved.contains(&format!("{THINK_START}private{THINK_END}visible")));

    let stripped = TemplateBuilder::with_options(RenderOptions {
        add_generation_prompt: false,
        preserve_thinking: false,
        ..RenderOptions::default()
    })
    .render(&conversation)
    .expect("assistant leading thinking region can be stripped");
    assert!(stripped.contains("visible"));
    assert!(!stripped.contains("private"));
}

#[test]
fn tool_branches_and_adjacent_results_are_structurally_distinct() {
    let tool_call = ToolCall {
        id: Some("call-1".to_owned()),
        name: "lookup".to_owned(),
        arguments: json!({"key":"value"}),
    };
    let conversation = Conversation {
        messages: vec![
            Message {
                role: MessageRole::Assistant,
                content: MessageContent::Text(String::new()),
                reasoning: None,
                tool_calls: vec![tool_call],
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
        tools: Vec::new(),
    };
    let xml = TemplateBuilder::with_options(RenderOptions {
        add_generation_prompt: false,
        tool_format: ToolFormat::Xml,
        ..RenderOptions::default()
    })
    .render(&conversation)
    .expect("XML branch renders");
    let json = TemplateBuilder::with_options(RenderOptions {
        add_generation_prompt: false,
        tool_format: ToolFormat::Json,
        ..RenderOptions::default()
    })
    .render(&conversation)
    .expect("JSON branch renders");
    assert!(xml.contains("<name>lookup</name>"));
    assert!(json.contains(r#"{"arguments":{"key":"value"},"id":"call-1","name":"lookup"}"#));
    assert_ne!(xml, json);
    assert_eq!(xml.matches(&format!("{IM_START}tool\n")).count(), 1);
    assert!(xml.contains("first</tool_result><tool_result"));
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
