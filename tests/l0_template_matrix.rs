//! L0 template-matrix harness skeleton.
//!
//! The frozen reference outputs are generated only by the Phase -1 oracle.
//! This file deliberately owns the matrix ids and mismatch diagnostics now, so
//! installing those bytes later changes test data rather than renderer logic.

use std::collections::BTreeMap;

use franken_nlp::{
    template::{
        AssistantReasoning, ContentPart, Conversation, Message, MessageContent, MessageRole,
        RenderOptions, TemplateBuilder, ToolCall, ToolDefinition, ToolFormat, ToolResult,
        DEFAULT_SYSTEM_TEXT, IM_END, IM_START, MEDIA_REMINDER_TEXT, THINK_END, THINK_START,
    },
    tokenizer::{
        bpe::{AddedToken, EncodeOptions, SpBpeTokenizer},
        sp_model::parse_spm_model,
    },
};
use serde_json::json;
use sha2::{Digest, Sha256};

const REFERENCE_INPUTS: &str = include_str!("fixtures/reference_inputs.json");
const REFERENCE_AUXILIARY: &str = include_str!("fixtures/reference/auxiliary.json");
const PINNED_TOKENIZER_MODEL: &[u8] = include_bytes!("fixtures/reference/tokenizer.model");
const PINNED_ADDED_TOKENS: &[u8] = include_bytes!("fixtures/reference/added_tokens.json");

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
    assert!(error
        .to_string()
        .contains("text, image, image_url, video, video_url, audio, audio_url, or input_audio"));

    let error = Conversation::from_json(
        r#"{"messages":[{"role":"assistant","content":"","tool_calls":[{"type":"function","function":{"name":"lookup","arguments":"[]"}}]}]}"#,
    )
    .expect_err("non-object tool-call arguments must reject");
    assert!(error
        .to_string()
        .contains("tool-call arguments must be a JSON object"));

    let conversation = Conversation::from_json(
        r#"{"messages":[{"role":"user","content":["bare text",{"type":"image"}]}]}"#,
    )
    .expect("the reference content-array branch accepts bare string items");
    let rendered = TemplateBuilder::with_options(RenderOptions {
        add_generation_prompt: false,
        ..RenderOptions::default()
    })
    .render(&conversation)
    .expect("bare content-array strings render before tokenization");
    assert!(rendered.contains("bare text"));
    assert!(rendered.contains(MEDIA_REMINDER_TEXT));
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
                    arguments_as_supplied: None,
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
    assert!(no_thinking.ends_with(&format!(
        "{IM_START}assistant\n{THINK_START}\n\n{THINK_END}\n\n"
    )));
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
    assert!(xml.starts_with(&format!(
        "{IM_START}system\nowner supplied system\n\n# Tools"
    )));
    assert!(xml.contains("<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"lookup\""));
    assert_eq!(xml.matches(&format!("{IM_START}system\n")).count(), 1);
    assert!(!xml.contains(DEFAULT_SYSTEM_TEXT));

    let json = TemplateBuilder::with_options(RenderOptions {
        add_generation_prompt: false,
        tool_format: ToolFormat::Json,
        ..RenderOptions::default()
    })
    .render(&first_system)
    .expect("first system with JSON tool declaration renders");
    assert!(json.starts_with(&format!(
        "{IM_START}system\nowner supplied system\n\n# Tools"
    )));
    assert!(json.contains(
        "\"function\": {\"name\": \"lookup\", \"description\": \"Look up a value.\", \"parameters\": {"
    ));
    assert_ne!(xml, json);

    let error = TemplateBuilder::with_options(RenderOptions {
        add_generation_prompt: false,
        ..RenderOptions::default()
    })
    .render(&non_leading_system)
    .expect_err("non-leading systems must reject before rendering");
    assert!(error
        .to_string()
        .contains("system message must be at the beginning"));
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
                ContentPart::Image { source: None },
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
    assert!(preserved.contains(&format!("{THINK_START}\nprivate\n{THINK_END}\n\nvisible")));

    let historical = Conversation::new(vec![
        Message::text(MessageRole::Assistant, "<think>private</think>visible"),
        Message::text(MessageRole::User, "follow-up"),
    ]);
    let stripped = TemplateBuilder::with_options(RenderOptions {
        add_generation_prompt: false,
        preserve_thinking: false,
        ..RenderOptions::default()
    })
    .render(&historical)
    .expect("assistant leading thinking region can be stripped");
    assert!(stripped.contains("visible"));
    assert!(!stripped.contains("private"));
}

#[test]
fn every_pinned_media_kind_keeps_its_template_derived_reminder_name() {
    let image = MEDIA_REMINDER_TEXT;
    let video = "<reminder>You are unable to process this video because you don't have multi-modal input ability. Try different methods.</reminder>";
    let audio = "<reminder>You are unable to process this audio because you don't have multi-modal input ability. Try different methods.</reminder>";
    let cases = vec![
        ("image", image),
        ("image_url", image),
        ("video", video),
        ("video_url", video),
        ("audio", audio),
        ("audio_url", audio),
        ("input_audio", audio),
    ];

    for (kind, reminder) in cases {
        let input = json!({
            "messages": [{
                "role": "user",
                "content": [{"type": kind}],
            }],
        })
        .to_string();
        let conversation = Conversation::from_json(&input)
            .expect("each pinned media kind is accepted by the typed boundary");
        let rendered = TemplateBuilder::with_options(RenderOptions {
            add_generation_prompt: false,
            ..RenderOptions::default()
        })
        .render(&conversation)
        .expect("every pinned media kind renders a trusted reminder");
        assert_eq!(
            rendered,
            format!(
                "{IM_START}system\n{DEFAULT_SYSTEM_TEXT}{IM_END}\n{IM_START}user\n{reminder}{IM_END}\n"
            ),
            "the {kind} part must retain the pinned template media type"
        );
    }
}

#[test]
fn tool_branches_and_adjacent_results_are_structurally_distinct() {
    let tool_call = ToolCall {
        id: Some("call-1".to_owned()),
        name: "lookup".to_owned(),
        arguments: json!({"key":"value"}),
        arguments_as_supplied: None,
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
    assert!(xml.contains("<function=lookup>"));
    assert!(json.contains(r#"{"name": "lookup", "arguments": {"key": "value"}}"#));
    assert_ne!(xml, json);
    assert_eq!(xml.matches(&format!("{IM_START}user")).count(), 1);
    assert!(xml.contains("first\n</tool_response>\n<tool_response>"));
}

#[test]
fn json_tool_call_preserves_validated_argument_text_spelling() {
    let supplied_arguments = "{\n  \"beta\": 2,\n  \"alpha\": [1, 2]\n}";
    let input = json!({
        "messages": [{
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "type": "function",
                "function": {
                    "name": "lookup",
                    "arguments": supplied_arguments,
                },
            }],
        }],
    })
    .to_string();
    let conversation = Conversation::from_json(&input)
        .expect("a valid JSON-object argument string crosses the typed boundary");

    let rendered = TemplateBuilder::with_options(RenderOptions {
        add_generation_prompt: false,
        tool_format: ToolFormat::Json,
        ..RenderOptions::default()
    })
    .render(&conversation)
    .expect("the JSON tool-call branch renders");

    let expected_call = [
        "<tool_call>\n{\"name\": \"lookup\", \"arguments\": ",
        supplied_arguments,
        "}\n</tool_call>",
    ]
    .concat();
    assert!(
        rendered.contains(&expected_call),
        "the pinned string-arguments branch must preserve supplied JSON spelling"
    );
    assert!(
        !rendered.contains(r#"{"name": "lookup", "arguments": {"alpha": [1, 2], "beta": 2}}"#),
        "the string-arguments branch must not replace supplied JSON with a serializer form"
    );
}

#[test]
fn xml_tool_call_uses_trimmed_content_for_first_call_spacing() {
    let conversation = Conversation {
        messages: vec![Message {
            role: MessageRole::Assistant,
            content: MessageContent::Text(" \n\t".to_owned()),
            reasoning: None,
            tool_calls: vec![ToolCall {
                id: None,
                name: "lookup".to_owned(),
                arguments: json!({"key":"value"}),
                arguments_as_supplied: None,
            }],
            tool_results: Vec::new(),
        }],
        tools: Vec::new(),
    };

    let rendered = TemplateBuilder::with_options(RenderOptions {
        add_generation_prompt: false,
        tool_format: ToolFormat::Xml,
        ..RenderOptions::default()
    })
    .render(&conversation)
    .expect("the XML tool-call branch renders");

    assert!(
        rendered.contains("<think>\n\n</think>\n\n \n\t<tool_call>\n<function=lookup>"),
        "whitespace-only content joins the first XML tool call without a separator"
    );
    assert!(
        !rendered.contains(" \n\t\n\n<tool_call>"),
        "the XML branch must not treat whitespace-only content as visible text"
    );
}

fn pinned_tokenizer() -> SpBpeTokenizer {
    let model = parse_spm_model(PINNED_TOKENIZER_MODEL)
        .expect("the frozen tokenizer.model must remain a valid SentencePiece BPE model");
    let added: BTreeMap<String, u32> = serde_json::from_slice(PINNED_ADDED_TOKENS)
        .expect("the frozen added-token registry must remain valid JSON");
    SpBpeTokenizer::with_added_tokens(
        model,
        added
            .into_iter()
            .map(|(surface, id)| AddedToken::new(surface, id)),
    )
    .expect("the frozen SentencePiece and added-token registries must compose")
}

fn render_options(value: &serde_json::Map<String, serde_json::Value>) -> RenderOptions {
    let tool_format = match value
        .get("tool_call_format")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("xml")
    {
        "xml" => ToolFormat::Xml,
        "json" => ToolFormat::Json,
        other => panic!("fixture supplied unknown pinned tool-call format {other:?}"),
    };
    RenderOptions {
        add_generation_prompt: value
            .get("add_generation_prompt")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
        enable_thinking: value
            .get("enable_thinking")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(true),
        preserve_thinking: value
            .get("preserve_thinking")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
        tool_format,
    }
}

#[test]
fn pinned_oracle_matrix_is_byte_and_token_id_exact() {
    let inputs: serde_json::Value =
        serde_json::from_str(REFERENCE_INPUTS).expect("reference input corpus is valid JSON");
    let auxiliary: serde_json::Value = serde_json::from_str(REFERENCE_AUXILIARY)
        .expect("reference auxiliary fixture is valid JSON");
    let input_cases = inputs["template_cases"]
        .as_array()
        .expect("reference inputs contain template cases");
    let expected_cases = auxiliary["template_cases"]
        .as_array()
        .expect("reference auxiliary fixture contains template cases");
    let tokenizer = pinned_tokenizer();
    let mut rendered_cases = 0_usize;

    for input_case in input_cases {
        let id = input_case["id"]
            .as_str()
            .expect("template fixture id is a string");
        let options = input_case["options"]
            .as_object()
            .expect("template fixture options are an object");
        let expected = expected_cases
            .iter()
            .find(|candidate| candidate["id"].as_str() == Some(id))
            .unwrap_or_else(|| panic!("missing frozen template output for cell {id}"));
        let conversation_value = json!({
            "messages": input_case["messages"].clone(),
            "tools": options.get("tools").cloned().unwrap_or_else(|| json!([])),
        });
        let conversation = Conversation::from_value(&conversation_value)
            .unwrap_or_else(|error| panic!("fixture cell {id} rejected before rendering: {error}"));
        let actual = TemplateBuilder::with_options(render_options(options))
            .render(&conversation)
            .unwrap_or_else(|error| panic!("fixture cell {id} did not render: {error}"));
        let expected_bytes = expected["rendered"]
            .as_str()
            .expect("frozen rendered prompt is a string")
            .as_bytes();
        assert_byte_exact(id, expected_bytes, actual.as_bytes());
        let expected_sha256 = expected["rendered_sha256"]
            .as_str()
            .expect("frozen rendered-prompt digest is a string");
        assert_eq!(
            format!("{:x}", Sha256::digest(actual.as_bytes())),
            expected_sha256,
            "frozen rendered-prompt digest drifted for cell {id}"
        );
        let expected_ids = expected["token_ids"]
            .as_array()
            .expect("frozen template token ids are an array")
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .and_then(|id| u32::try_from(id).ok())
                    .expect("frozen template token id fits u32")
            })
            .collect::<Vec<_>>();
        let actual_ids = tokenizer
            .encode_ids_with_options(
                &actual,
                EncodeOptions {
                    add_bos: false,
                    add_eos: false,
                },
            )
            .unwrap_or_else(|error| panic!("fixture cell {id} did not tokenize: {error}"));
        assert_eq!(
            actual_ids, expected_ids,
            "L0 template token-id mismatch for fixture cell {id}"
        );
        eprintln!("L0T cell={id} RESULT=PASS");
        rendered_cases += 1;
    }

    assert_eq!(rendered_cases, expected_cases.len());
    assert_eq!(
        rendered_cases, 5,
        "the frozen Phase -1 corpus must retain its complete template matrix"
    );
    eprintln!("L0_TEMPLATE RESULT=PASS cells={rendered_cases} rejects=3");
}
