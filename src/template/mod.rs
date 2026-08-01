//! Typed, fixed-program chat-template surface for the pinned Nanbeige model.
//!
//! This module deliberately implements one program rather than a template
//! language.  It owns trusted emission of role, thinking, and tool controls;
//! callers pass a typed [`Conversation`] and receive bytes to give to the
//! tokenizer.  Parsing arbitrary JSON-shaped input is strict and happens
//! before that tokenizer boundary.
//!
//! | L0 matrix cell | Typed surface exercised |
//! | --- | --- |
//! | first/default/non-leading system | [`MessageRole::System`] |
//! | text and media parts | [`MessageContent`], [`ContentPart`] |
//! | thinking and preservation | [`AssistantReasoning`], [`RenderOptions`] |
//! | XML and JSON tools | [`ToolFormat`], [`ToolDefinition`], [`ToolCall`] |
//! | adjacent tool replies | [`ToolResult`] grouping |
//! | generation tails | [`RenderOptions::add_generation_prompt`] |
//!
//! The byte-exact literals and reference cases are frozen by the Phase -1
//! fixture bead. The executable matrix below is the authority for L0 prompt
//! bytes and token ids.

use std::collections::{BTreeSet, HashSet};
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// The role delimiter emitted only by trusted template code.
pub const IM_START: &str = "<|im_start|>";
/// The role delimiter emitted only by trusted template code.
pub const IM_END: &str = "<|im_end|>";
/// The thinking opening delimiter emitted only by trusted template code.
pub const THINK_START: &str = "<think>";
/// The thinking closing delimiter emitted only by trusted template code.
pub const THINK_END: &str = "</think>";
/// The XML tool-call opening delimiter emitted only by trusted template code.
pub const TOOL_CALL_START: &str = "<tool_call>";
/// The XML tool-call closing delimiter emitted only by trusted template code.
pub const TOOL_CALL_END: &str = "</tool_call>";

/// The fixed fallback used when the first message is not a system message and
/// no tools are declared.
///
/// Its bytes are the pinned slow-tokenizer template's default, not a
/// project-authored substitute.
pub const DEFAULT_SYSTEM_TEXT: &str = "你是南北阁，一款由BOSS直聘自主研发并训练的专业大语言模型。";

/// The pinned fallback system instruction for a tools-enabled conversation.
const TOOL_SYSTEM_TEXT: &str = "你是一位工具函数调用专家，你会得到一个问题和一组可能的工具函数。根据问题，你需要进行一个或多个函数/工具调用以实现目的，请尽量尝试探索通过工具解决问题。\n如果没有一个函数可以使用，请直接使用自然语言回复用户。\n如果给定的问题缺少函数所需的参数，请使用自然语言进行提问，向用户询问必要信息。\n如果调用结果已经足够回答用户问题，请对历史结果进行总结，使用自然语言回复用户。";

/// The pinned reminder substituted for an image content part.
///
/// The template is text-only; original media payloads are never rendered as
/// model input. Video and audio parts use their own type-named literals, exactly
/// as the pinned template derives them.
pub const MEDIA_REMINDER_TEXT: &str =
    "<reminder>You are unable to process this image because you don't have multi-modal input ability. Try different methods.</reminder>";

/// A complete conversation passed to the fixed chat-template builder.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Conversation {
    /// Ordered messages; a system message is special only in position zero.
    pub messages: Vec<Message>,
    /// Function declarations made available to the conversation.
    pub tools: Vec<ToolDefinition>,
}

impl Conversation {
    /// Construct a conversation with no tool declarations.
    pub fn new(messages: Vec<Message>) -> Self {
        Self {
            messages,
            tools: Vec::new(),
        }
    }

    /// Decode only the pinned accepted JSON-shaped conversation surface.
    ///
    /// This uses the repository's duplicate-key-rejecting JSON boundary, then
    /// rejects unknown mapping shapes before any call to a tokenizer can occur.
    pub fn from_json(input: &str) -> Result<Self, TemplateError> {
        let value =
            crate::canonjson::parse_str(input).map_err(|error| TemplateError::InvalidShape {
                path: "$".to_owned(),
                expected: format!("duplicate-key-free conversation JSON ({error})"),
            })?;
        Self::from_value(&value)
    }

    /// Decode the same strict accepted surface from an already parsed value.
    pub fn from_value(value: &Value) -> Result<Self, TemplateError> {
        let object = object_at(value, "$")?;
        reject_unknown_keys(object, &["messages", "tools"], "$")?;
        let messages_value = required_value(object, "messages", "$")?;
        let messages = messages_value
            .as_array()
            .ok_or_else(|| TemplateError::InvalidShape {
                path: "$.messages".to_owned(),
                expected: "array".to_owned(),
            })?
            .iter()
            .enumerate()
            .map(|(index, message)| parse_message(message, &format!("$.messages[{index}]")))
            .collect::<Result<Vec<_>, _>>()?;
        let tools = match object.get("tools") {
            Some(value) => parse_tool_definitions(value, "$.tools")?,
            None => Vec::new(),
        };
        let conversation = Self { messages, tools };
        conversation.validate()?;
        Ok(conversation)
    }

    /// Reject malformed typed values before rendering or tokenization.
    pub fn validate(&self) -> Result<(), TemplateError> {
        if self.messages.is_empty() {
            return Err(TemplateError::EmptyConversation);
        }
        let mut tool_names = BTreeSet::new();
        for tool in &self.tools {
            tool.validate()?;
            if !tool_names.insert(tool.name.as_str()) {
                return Err(TemplateError::DuplicateToolName {
                    name: tool.name.clone(),
                });
            }
        }
        for (index, message) in self.messages.iter().enumerate() {
            if message.role == MessageRole::System && index != 0 {
                return Err(TemplateError::InvalidMessage {
                    index,
                    reason: "system message must be at the beginning".to_owned(),
                });
            }
            message.validate(index)?;
        }
        Ok(())
    }
}

/// One role-framed conversation message.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Message {
    /// The pinned role vocabulary; an untrusted role string is never coerced.
    pub role: MessageRole,
    /// String content or the pinned iterable content-part form.
    pub content: MessageContent,
    /// Assistant-only reasoning, supplied directly or extracted from a leading
    /// `<think>…</think>` region of assistant text.
    pub reasoning: Option<AssistantReasoning>,
    /// Assistant-only calls emitted in the selected tool format.
    pub tool_calls: Vec<ToolCall>,
    /// Tool-only results; consecutive tool messages are rendered as one group.
    pub tool_results: Vec<ToolResult>,
}

impl Message {
    /// Construct a plain textual message.
    pub fn text(role: MessageRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: MessageContent::Text(content.into()),
            reasoning: None,
            tool_calls: Vec::new(),
            tool_results: Vec::new(),
        }
    }

    fn validate(&self, index: usize) -> Result<(), TemplateError> {
        self.content.validate(index)?;
        if self.reasoning.is_some() && self.role != MessageRole::Assistant {
            return Err(TemplateError::InvalidMessage {
                index,
                reason: "reasoning is valid only on assistant messages".to_owned(),
            });
        }
        if !self.tool_calls.is_empty() && self.role != MessageRole::Assistant {
            return Err(TemplateError::InvalidMessage {
                index,
                reason: "tool_calls are valid only on assistant messages".to_owned(),
            });
        }
        if !self.tool_results.is_empty() && self.role != MessageRole::Tool {
            return Err(TemplateError::InvalidMessage {
                index,
                reason: "tool_results are valid only on tool messages".to_owned(),
            });
        }
        for call in &self.tool_calls {
            call.validate(index)?;
        }
        for result in &self.tool_results {
            result.validate(index)?;
        }
        Ok(())
    }
}

/// The role names accepted by the pinned template.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    /// Trusted system instruction framing.
    System,
    /// User-provided turn framing.
    User,
    /// Model response framing.
    Assistant,
    /// Tool-response framing.
    Tool,
}

impl MessageRole {
    fn parse(value: &str, path: &str) -> Result<Self, TemplateError> {
        match value {
            "system" => Ok(Self::System),
            "user" => Ok(Self::User),
            "assistant" => Ok(Self::Assistant),
            "tool" => Ok(Self::Tool),
            _ => Err(TemplateError::UnknownRole {
                path: path.to_owned(),
                role: value.to_owned(),
            }),
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }
}

/// Message content accepted by the pinned template.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// The ordinary text form.
    Text(String),
    /// The iterable content-part form used for text plus media reminders.
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    fn validate(&self, index: usize) -> Result<(), TemplateError> {
        match self {
            Self::Text(_) => Ok(()),
            Self::Parts(parts) if parts.is_empty() => Err(TemplateError::InvalidMessage {
                index,
                reason: "content-parts arrays must not be empty".to_owned(),
            }),
            Self::Parts(parts) => {
                for part in parts {
                    part.validate(index)?;
                }
                Ok(())
            }
        }
    }
}

/// A content part from the pinned text/media accepted surface.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    /// Text is rendered verbatim by the trusted builder.
    Text { text: String },
    /// Pinned image part; its payload becomes [`MEDIA_REMINDER_TEXT`].
    Image { source: Option<String> },
    /// Pinned image-url part; its payload becomes [`MEDIA_REMINDER_TEXT`].
    ImageUrl { source: Option<String> },
    /// Pinned video part; its payload becomes the type-named video reminder.
    Video { source: Option<String> },
    /// Pinned video-url part; its payload becomes the type-named video reminder.
    VideoUrl { source: Option<String> },
    /// Pinned audio part; its payload becomes the type-named audio reminder.
    Audio { source: Option<String> },
    /// Pinned audio-url part; its payload becomes the type-named audio reminder.
    AudioUrl { source: Option<String> },
    /// Pinned input-audio part; its payload becomes the type-named audio reminder.
    InputAudio { source: Option<String> },
}

impl ContentPart {
    fn validate(&self, index: usize) -> Result<(), TemplateError> {
        match self {
            Self::Text { text } if text.is_empty() => Err(TemplateError::InvalidMessage {
                index,
                reason: "text content parts must not be empty".to_owned(),
            }),
            _ => Ok(()),
        }
    }
}

/// Typed assistant reasoning and the source from which it was obtained.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct AssistantReasoning {
    /// Text between the pinned thinking delimiters.
    pub reasoning_content: String,
    /// Whether callers supplied the field or it was safely extracted.
    pub source: ReasoningSource,
}

impl AssistantReasoning {
    /// Create an explicit `reasoning_content` value.
    pub fn explicit(reasoning_content: impl Into<String>) -> Self {
        Self {
            reasoning_content: reasoning_content.into(),
            source: ReasoningSource::ExplicitField,
        }
    }

    /// Extract the same assistant reasoning region as the pinned fixed template.
    ///
    /// Think-looking text in user/system/tool content never reaches this method
    /// and is therefore ordinary text, not trusted control markup.
    pub fn extract_leading(content: &str) -> Option<(Self, String)> {
        let close = content.find(THINK_END)?;
        let before_close = content[..close].trim_end_matches('\n');
        let reasoning_content = before_close
            .rsplit(THINK_START)
            .next()
            .expect("rsplit always yields at least one segment")
            .trim_start_matches('\n')
            .to_owned();
        // The reference Jinja expression uses `content.split('</think>')[-1]`:
        // reasoning comes from before the first close, but visible content is
        // the segment after the last close.
        let visible_content = content
            .rsplit_once(THINK_END)
            .expect("the first close guarantees a last close")
            .1
            .trim_start_matches('\n')
            .trim_end_matches('\n')
            .to_owned();
        Some((
            Self {
                reasoning_content,
                source: ReasoningSource::EmbeddedThinkTags,
            },
            visible_content,
        ))
    }
}

/// Provenance of [`AssistantReasoning`].
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningSource {
    /// The structured `reasoning_content` field was supplied.
    ExplicitField,
    /// The builder extracted a leading complete `<think>…</think>` region.
    EmbeddedThinkTags,
}

/// A function declaration made available to the template.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ToolDefinition {
    /// The fixed template accepts only function definitions.
    pub name: String,
    /// Human-readable purpose shown to the model.
    pub description: String,
    /// JSON-schema-shaped function parameters.
    pub parameters: Value,
}

impl ToolDefinition {
    fn validate(&self) -> Result<(), TemplateError> {
        if self.name.is_empty() || self.name.chars().any(char::is_whitespace) {
            return Err(TemplateError::InvalidTool {
                name: self.name.clone(),
                reason: "tool names must be non-empty and whitespace-free".to_owned(),
            });
        }
        if !self.parameters.is_object() {
            return Err(TemplateError::InvalidTool {
                name: self.name.clone(),
                reason: "parameters must be a JSON object".to_owned(),
            });
        }
        Ok(())
    }
}

/// One assistant function call.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ToolCall {
    /// Stable call identity when supplied by the caller.
    pub id: Option<String>,
    /// Declared function name.
    pub name: String,
    /// Validated structured function arguments.
    pub arguments: Value,
    /// Original JSON-object text supplied by an OpenAI-style function wrapper.
    ///
    /// The pinned JSON tool-call branch emits a string `arguments` value
    /// verbatim. This field retains that spelling after it has been validated
    /// into [`Self::arguments`]; typed callers normally leave it as `None`.
    #[serde(skip)]
    pub arguments_as_supplied: Option<String>,
}

impl ToolCall {
    fn validate(&self, index: usize) -> Result<(), TemplateError> {
        if self.name.is_empty() || self.name.chars().any(char::is_whitespace) {
            return Err(TemplateError::InvalidMessage {
                index,
                reason: "tool-call name must be non-empty and whitespace-free".to_owned(),
            });
        }
        if !self.arguments.is_object() {
            return Err(TemplateError::InvalidMessage {
                index,
                reason: "tool-call arguments must be a JSON object".to_owned(),
            });
        }
        if let Some(original) = &self.arguments_as_supplied {
            let parsed = crate::canonjson::parse_str(original).map_err(|error| {
                TemplateError::InvalidMessage {
                    index,
                    reason: format!(
                        "tool-call original arguments must be duplicate-key-free JSON ({error})"
                    ),
                }
            })?;
            if parsed != self.arguments {
                return Err(TemplateError::InvalidMessage {
                    index,
                    reason: "tool-call original arguments must match structured arguments"
                        .to_owned(),
                });
            }
        }
        Ok(())
    }
}

/// One result returned by a tool call.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ToolResult {
    /// Identity of the call receiving this result.
    pub tool_call_id: String,
    /// Textual tool output; it is data, never template control markup.
    pub content: String,
}

impl ToolResult {
    fn validate(&self, index: usize) -> Result<(), TemplateError> {
        if self.tool_call_id.is_empty() {
            return Err(TemplateError::InvalidMessage {
                index,
                reason: "tool-result tool_call_id must not be empty".to_owned(),
            });
        }
        Ok(())
    }
}

/// The two pinned serialization branches for tool declarations and calls.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolFormat {
    /// The native XML branch.
    #[default]
    Xml,
    /// The JSON branch selected by `tool_call_format="json"`.
    Json,
}

/// Explicit rendering options; no ambient mode changes the prompt bytes.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RenderOptions {
    /// Include the assistant generation tail after historical messages.
    pub add_generation_prompt: bool,
    /// Emit the thinking marker in the generation tail.
    pub enable_thinking: bool,
    /// Retain prior assistant reasoning regions instead of stripping them.
    pub preserve_thinking: bool,
    /// Tool serialization branch.
    pub tool_format: ToolFormat,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            add_generation_prompt: true,
            enable_thinking: true,
            preserve_thinking: false,
            tool_format: ToolFormat::Xml,
        }
    }
}

/// Deterministic builder for the one pinned chat-template program.
#[derive(Clone, Copy, Debug, Default)]
pub struct TemplateBuilder {
    options: RenderOptions,
}

impl TemplateBuilder {
    /// Select all prompt-affecting template options explicitly.
    pub const fn with_options(options: RenderOptions) -> Self {
        Self { options }
    }

    /// Render the typed conversation into prompt bytes represented as UTF-8.
    pub fn render(&self, conversation: &Conversation) -> Result<String, TemplateError> {
        conversation.validate()?;
        let mut output = String::new();
        let first_system = conversation
            .messages
            .first()
            .filter(|message| message.role == MessageRole::System);
        if conversation.tools.is_empty() {
            render_frame(
                &mut output,
                MessageRole::System,
                &first_system.map_or_else(
                    || DEFAULT_SYSTEM_TEXT.to_owned(),
                    |message| render_content(&message.content),
                ),
            );
        } else {
            render_tool_preamble(
                &mut output,
                &conversation.tools,
                first_system.map(|message| render_content(&message.content)),
                self.options.tool_format,
            )?;
        }

        let last_query_index = conversation
            .messages
            .iter()
            .enumerate()
            .rev()
            .find_map(|(index, message)| {
                (message.role == MessageRole::User
                    && !is_tool_response(&render_content(&message.content)))
                .then_some(index)
            })
            .unwrap_or(conversation.messages.len() - 1);

        let mut index = 0;
        while index < conversation.messages.len() {
            let message = &conversation.messages[index];
            if message.role == MessageRole::System {
                index += 1;
                continue;
            }
            if message.role == MessageRole::Tool {
                let group_end = conversation.messages[index..]
                    .iter()
                    .take_while(|candidate| candidate.role == MessageRole::Tool)
                    .count()
                    + index;
                render_tool_result_group(&mut output, &conversation.messages[index..group_end]);
                index = group_end;
                continue;
            }
            render_message(&mut output, message, index, last_query_index, self.options)?;
            index += 1;
        }
        if self.options.add_generation_prompt {
            output.push_str(IM_START);
            output.push_str("assistant\n");
            if self.options.enable_thinking {
                output.push_str(THINK_START);
                output.push('\n');
            } else {
                output.push_str(THINK_START);
                output.push_str("\n\n");
                output.push_str(THINK_END);
                output.push_str("\n\n");
            }
        }
        Ok(output)
    }
}

/// Typed failures at the template/tokenizer boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TemplateError {
    /// An arbitrary string was not one of the four pinned roles.
    UnknownRole { path: String, role: String },
    /// An input value had an unsupported shape before any tokenization.
    InvalidShape { path: String, expected: String },
    /// A typed conversation contained no messages.
    EmptyConversation,
    /// A typed message was invalid for its role.
    InvalidMessage { index: usize, reason: String },
    /// A function definition was malformed.
    InvalidTool { name: String, reason: String },
    /// Function declarations duplicated a name.
    DuplicateToolName { name: String },
    /// A structured JSON value could not be canonically rendered.
    CanonicalJson {
        context: &'static str,
        message: String,
    },
}

impl fmt::Display for TemplateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownRole { path, role } => {
                write!(formatter, "unknown role {role:?} at {path}")
            }
            Self::InvalidShape { path, expected } => {
                write!(formatter, "invalid shape at {path}; expected {expected}")
            }
            Self::EmptyConversation => {
                formatter.write_str("conversation requires at least one message")
            }
            Self::InvalidMessage { index, reason } => {
                write!(formatter, "invalid message {index}: {reason}")
            }
            Self::InvalidTool { name, reason } => {
                write!(formatter, "invalid tool {name:?}: {reason}")
            }
            Self::DuplicateToolName { name } => {
                write!(formatter, "duplicate tool definition {name:?}")
            }
            Self::CanonicalJson { context, message } => {
                write!(formatter, "cannot render {context}: {message}")
            }
        }
    }
}

impl Error for TemplateError {}

fn render_message(
    output: &mut String,
    message: &Message,
    index: usize,
    last_query_index: usize,
    options: RenderOptions,
) -> Result<(), TemplateError> {
    let mut content = render_content(&message.content);
    let reasoning = if message.role == MessageRole::Assistant {
        message.reasoning.clone().or_else(|| {
            AssistantReasoning::extract_leading(&content).map(|(reasoning, visible)| {
                content = visible;
                reasoning
            })
        })
    } else {
        None
    };
    let mut body = String::new();
    if message.role == MessageRole::Assistant {
        let strip_historical_reasoning = !options.preserve_thinking && index < last_query_index;
        body.push_str(THINK_START);
        body.push('\n');
        if !strip_historical_reasoning {
            if let Some(reasoning) = reasoning {
                body.push_str(reasoning.reasoning_content.trim());
            }
        }
        body.push('\n');
        body.push_str(THINK_END);
        body.push_str("\n\n");
    }
    body.push_str(&content);
    if !message.tool_calls.is_empty() {
        render_tool_calls(
            &mut body,
            &message.tool_calls,
            options.tool_format,
            !content.is_empty(),
        )?;
    }
    render_frame(output, message.role, &body);
    Ok(())
}

fn render_content(content: &MessageContent) -> String {
    match content {
        MessageContent::Text(text) => text.clone(),
        MessageContent::Parts(parts) => {
            let mut output = String::new();
            for part in parts {
                match part {
                    ContentPart::Text { text } => output.push_str(text),
                    ContentPart::Image { .. }
                    | ContentPart::ImageUrl { .. }
                    | ContentPart::Video { .. }
                    | ContentPart::VideoUrl { .. }
                    | ContentPart::Audio { .. }
                    | ContentPart::AudioUrl { .. }
                    | ContentPart::InputAudio { .. } => {
                        output.push_str(media_reminder(part));
                    }
                }
            }
            output
        }
    }
}

fn is_tool_response(content: &str) -> bool {
    content.starts_with("<tool_response>") && content.ends_with("</tool_response>")
}

fn media_reminder(part: &ContentPart) -> &'static str {
    match part {
        ContentPart::Image { .. } | ContentPart::ImageUrl { .. } => MEDIA_REMINDER_TEXT,
        ContentPart::Video { .. } | ContentPart::VideoUrl { .. } => {
            "<reminder>You are unable to process this video because you don't have multi-modal input ability. Try different methods.</reminder>"
        }
        ContentPart::Audio { .. }
        | ContentPart::AudioUrl { .. }
        | ContentPart::InputAudio { .. } => {
            "<reminder>You are unable to process this audio because you don't have multi-modal input ability. Try different methods.</reminder>"
        }
        ContentPart::Text { .. } => unreachable!("text parts do not have media reminders"),
    }
}

fn render_frame(output: &mut String, role: MessageRole, body: &str) {
    output.push_str(IM_START);
    output.push_str(role.as_str());
    output.push('\n');
    output.push_str(body);
    output.push_str(IM_END);
    output.push('\n');
}

fn render_tool_preamble(
    output: &mut String,
    tools: &[ToolDefinition],
    first_system: Option<String>,
    format: ToolFormat,
) -> Result<(), TemplateError> {
    output.push_str(IM_START);
    output.push_str("system\n");
    if let Some(system) = first_system {
        output.push_str(&system);
        output.push_str("\n\n");
    } else {
        output.push_str(TOOL_SYSTEM_TEXT);
    }
    output.push_str("# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>");
    for tool in tools {
        output.push('\n');
        render_tool_definition_json(output, tool)?;
    }
    match format {
        ToolFormat::Xml => {
            output.push_str("\n</tools>\n\nFor each function call, output the function name and arguments within the following XML format:\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>");
        }
        ToolFormat::Json => {
            output.push_str("\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>");
        }
    }
    output.push_str(IM_END);
    output.push('\n');
    Ok(())
}

fn render_tool_definition_json(
    output: &mut String,
    tool: &ToolDefinition,
) -> Result<(), TemplateError> {
    output.push_str("{\"type\": \"function\", \"function\": {\"name\": ");
    render_json_string(output, &tool.name)?;
    output.push_str(", \"description\": ");
    render_json_string(output, &tool.description)?;
    output.push_str(", \"parameters\": ");
    render_jinja_json(output, &tool.parameters)?;
    output.push_str("}}");
    Ok(())
}

fn render_tool_argument_value(output: &mut String, value: &Value) -> Result<(), TemplateError> {
    if let Some(text) = value.as_str() {
        output.push_str(text);
        return Ok(());
    }
    render_jinja_json(output, value)
}

fn render_jinja_json(output: &mut String, value: &Value) -> Result<(), TemplateError> {
    match value {
        Value::Null => output.push_str("null"),
        Value::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
        Value::Number(value) => output.push_str(&value.to_string()),
        Value::String(value) => render_json_string(output, value)?,
        Value::Array(values) => {
            output.push('[');
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    output.push_str(", ");
                }
                render_jinja_json(output, value)?;
            }
            output.push(']');
        }
        Value::Object(values) => {
            output.push('{');
            let mut first = true;
            for key in ["type", "properties", "required"] {
                if let Some(value) = values.get(key) {
                    render_jinja_json_entry(output, key, value, &mut first)?;
                }
            }
            for (key, value) in values {
                if !matches!(key.as_str(), "type" | "properties" | "required") {
                    render_jinja_json_entry(output, key, value, &mut first)?;
                }
            }
            output.push('}');
        }
    }
    Ok(())
}

fn render_jinja_json_entry(
    output: &mut String,
    key: &str,
    value: &Value,
    first: &mut bool,
) -> Result<(), TemplateError> {
    if !*first {
        output.push_str(", ");
    }
    *first = false;
    render_json_string(output, key)?;
    output.push_str(": ");
    render_jinja_json(output, value)
}

fn render_json_string(output: &mut String, value: &str) -> Result<(), TemplateError> {
    let encoded = serde_json::to_string(value).map_err(|error| TemplateError::CanonicalJson {
        context: "template JSON string",
        message: error.to_string(),
    })?;
    output.push_str(&encoded);
    Ok(())
}

fn render_tool_calls(
    output: &mut String,
    calls: &[ToolCall],
    format: ToolFormat,
    content_is_present: bool,
) -> Result<(), TemplateError> {
    for (call_index, call) in calls.iter().enumerate() {
        match format {
            ToolFormat::Xml => {
                if call_index == 0 && content_is_present {
                    output.push_str("\n\n");
                } else if call_index > 0 {
                    output.push('\n');
                }
                output.push_str("<tool_call>\n<function=");
                output.push_str(&call.name);
                output.push_str(">\n");
                let arguments =
                    call.arguments
                        .as_object()
                        .ok_or_else(|| TemplateError::InvalidMessage {
                            index: 0,
                            reason: "tool-call arguments must be a JSON object".to_owned(),
                        })?;
                for (name, value) in arguments {
                    output.push_str("<parameter=");
                    output.push_str(name);
                    output.push_str(">\n");
                    render_tool_argument_value(output, value)?;
                    output.push_str("\n</parameter>\n");
                }
                output.push_str("</function>\n</tool_call>");
            }
            ToolFormat::Json => {
                if (call_index == 0 && content_is_present) || call_index > 0 {
                    output.push('\n');
                }
                output.push_str("<tool_call>\n{\"name\": ");
                render_json_string(output, &call.name)?;
                output.push_str(", \"arguments\": ");
                if let Some(original) = &call.arguments_as_supplied {
                    output.push_str(original);
                } else {
                    render_jinja_json(output, &call.arguments)?;
                }
                output.push_str("}\n</tool_call>");
            }
        }
    }
    Ok(())
}

fn render_tool_result_group(output: &mut String, messages: &[Message]) {
    output.push_str(IM_START);
    output.push_str("user");
    for message in messages {
        output.push_str("\n<tool_response>\n");
        output.push_str(&render_content(&message.content));
        output.push_str("\n</tool_response>");
    }
    output.push_str(IM_END);
    output.push('\n');
}

fn object_at<'a>(value: &'a Value, path: &str) -> Result<&'a Map<String, Value>, TemplateError> {
    value
        .as_object()
        .ok_or_else(|| TemplateError::InvalidShape {
            path: path.to_owned(),
            expected: "object".to_owned(),
        })
}

fn required_value<'a>(
    object: &'a Map<String, Value>,
    key: &str,
    path: &str,
) -> Result<&'a Value, TemplateError> {
    object.get(key).ok_or_else(|| TemplateError::InvalidShape {
        path: format!("{path}.{key}"),
        expected: "required field".to_owned(),
    })
}

fn required_string(
    object: &Map<String, Value>,
    key: &str,
    path: &str,
) -> Result<String, TemplateError> {
    required_value(object, key, path)?
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| TemplateError::InvalidShape {
            path: format!("{path}.{key}"),
            expected: "string".to_owned(),
        })
}

fn reject_unknown_keys(
    object: &Map<String, Value>,
    allowed: &[&str],
    path: &str,
) -> Result<(), TemplateError> {
    let allowed = allowed.iter().copied().collect::<HashSet<_>>();
    if let Some(key) = object.keys().find(|key| !allowed.contains(key.as_str())) {
        return Err(TemplateError::InvalidShape {
            path: format!("{path}.{key}"),
            expected: "known mapping field".to_owned(),
        });
    }
    Ok(())
}

fn parse_message(value: &Value, path: &str) -> Result<Message, TemplateError> {
    let object = object_at(value, path)?;
    reject_unknown_keys(
        object,
        &[
            "role",
            "content",
            "reasoning_content",
            "tool_calls",
            "tool_call_id",
        ],
        path,
    )?;
    let role = MessageRole::parse(
        &required_string(object, "role", path)?,
        &format!("{path}.role"),
    )?;
    if role != MessageRole::Tool && object.contains_key("tool_call_id") {
        return Err(TemplateError::InvalidShape {
            path: format!("{path}.tool_call_id"),
            expected: "field valid only for a tool role".to_owned(),
        });
    }
    let content = parse_content(
        required_value(object, "content", path)?,
        &format!("{path}.content"),
    )?;
    let reasoning = object
        .get("reasoning_content")
        .map(|value| {
            value
                .as_str()
                .map(AssistantReasoning::explicit)
                .ok_or_else(|| TemplateError::InvalidShape {
                    path: format!("{path}.reasoning_content"),
                    expected: "string".to_owned(),
                })
        })
        .transpose()?;
    let tool_calls = match object.get("tool_calls") {
        Some(value) => parse_tool_calls(value, &format!("{path}.tool_calls"))?,
        None => Vec::new(),
    };
    let tool_results = if role == MessageRole::Tool {
        vec![ToolResult {
            tool_call_id: required_string(object, "tool_call_id", path)?,
            content: render_content(&content),
        }]
    } else {
        Vec::new()
    };
    let message = Message {
        role,
        content,
        reasoning,
        tool_calls,
        tool_results,
    };
    message.validate(0)?;
    Ok(message)
}

fn parse_content(value: &Value, path: &str) -> Result<MessageContent, TemplateError> {
    if let Some(text) = value.as_str() {
        return Ok(MessageContent::Text(text.to_owned()));
    }
    let parts = value
        .as_array()
        .ok_or_else(|| TemplateError::InvalidShape {
            path: path.to_owned(),
            expected: "string or content-parts array".to_owned(),
        })?;
    parts
        .iter()
        .enumerate()
        .map(|(index, part)| parse_content_part(part, &format!("{path}[{index}]")))
        .collect::<Result<Vec<_>, _>>()
        .map(MessageContent::Parts)
}

fn parse_content_part(value: &Value, path: &str) -> Result<ContentPart, TemplateError> {
    if let Some(text) = value.as_str() {
        return Ok(ContentPart::Text {
            text: text.to_owned(),
        });
    }
    let object = object_at(value, path)?;
    let kind = required_string(object, "type", path)?;
    match kind.as_str() {
        "text" => {
            reject_unknown_keys(object, &["type", "text"], path)?;
            Ok(ContentPart::Text {
                text: required_string(object, "text", path)?,
            })
        }
        "image" => parse_media_part(object, path, "image", |source| ContentPart::Image {
            source,
        }),
        "image_url" => parse_media_part(object, path, "image_url", |source| {
            ContentPart::ImageUrl { source }
        }),
        "video" => parse_media_part(object, path, "video", |source| ContentPart::Video {
            source,
        }),
        "video_url" => parse_media_part(object, path, "video_url", |source| {
            ContentPart::VideoUrl { source }
        }),
        "audio" => parse_media_part(object, path, "audio", |source| ContentPart::Audio {
            source,
        }),
        "audio_url" => parse_media_part(object, path, "audio_url", |source| {
            ContentPart::AudioUrl { source }
        }),
        "input_audio" => parse_media_part(object, path, "input_audio", |source| {
            ContentPart::InputAudio { source }
        }),
        _ => Err(TemplateError::InvalidShape {
            path: format!("{path}.type"),
            expected: "text, image, image_url, video, video_url, audio, audio_url, or input_audio"
                .to_owned(),
        }),
    }
}

fn parse_media_part(
    object: &Map<String, Value>,
    path: &str,
    key: &str,
    constructor: impl FnOnce(Option<String>) -> ContentPart,
) -> Result<ContentPart, TemplateError> {
    reject_unknown_keys(object, &["type", key], path)?;
    let source = object
        .get(key)
        .map(|value| media_source(value, &format!("{path}.{key}")))
        .transpose()?;
    Ok(constructor(source))
}

fn media_source(value: &Value, path: &str) -> Result<String, TemplateError> {
    if let Some(source) = value.as_str() {
        return Ok(source.to_owned());
    }
    let object = object_at(value, path)?;
    reject_unknown_keys(object, &["url"], path)?;
    required_string(object, "url", path)
}

fn parse_tool_definitions(value: &Value, path: &str) -> Result<Vec<ToolDefinition>, TemplateError> {
    value
        .as_array()
        .ok_or_else(|| TemplateError::InvalidShape {
            path: path.to_owned(),
            expected: "array".to_owned(),
        })?
        .iter()
        .enumerate()
        .map(|(index, definition)| {
            let definition_path = format!("{path}[{index}]");
            let object = object_at(definition, &definition_path)?;
            reject_unknown_keys(object, &["type", "function"], &definition_path)?;
            if required_string(object, "type", &definition_path)? != "function" {
                return Err(TemplateError::InvalidShape {
                    path: format!("{definition_path}.type"),
                    expected: "function".to_owned(),
                });
            }
            let function_path = format!("{definition_path}.function");
            let function = object_at(
                required_value(object, "function", &definition_path)?,
                &function_path,
            )?;
            reject_unknown_keys(
                function,
                &["name", "description", "parameters"],
                &function_path,
            )?;
            let tool = ToolDefinition {
                name: required_string(function, "name", &function_path)?,
                description: required_string(function, "description", &function_path)?,
                parameters: required_value(function, "parameters", &function_path)?.clone(),
            };
            tool.validate()?;
            Ok(tool)
        })
        .collect()
}

fn parse_tool_calls(value: &Value, path: &str) -> Result<Vec<ToolCall>, TemplateError> {
    value
        .as_array()
        .ok_or_else(|| TemplateError::InvalidShape {
            path: path.to_owned(),
            expected: "array".to_owned(),
        })?
        .iter()
        .enumerate()
        .map(|(index, call)| {
            let call_path = format!("{path}[{index}]");
            let object = object_at(call, &call_path)?;
            reject_unknown_keys(object, &["id", "type", "function"], &call_path)?;
            if required_string(object, "type", &call_path)? != "function" {
                return Err(TemplateError::InvalidShape {
                    path: format!("{call_path}.type"),
                    expected: "function".to_owned(),
                });
            }
            let function_path = format!("{call_path}.function");
            let function = object_at(
                required_value(object, "function", &call_path)?,
                &function_path,
            )?;
            reject_unknown_keys(function, &["name", "arguments"], &function_path)?;
            let arguments = required_value(function, "arguments", &function_path)?;
            let (arguments, arguments_as_supplied) = match arguments {
                Value::String(encoded) => {
                    let parsed = crate::canonjson::parse_str(encoded).map_err(|error| {
                        TemplateError::InvalidShape {
                            path: format!("{function_path}.arguments"),
                            expected: format!("duplicate-key-free JSON object ({error})"),
                        }
                    })?;
                    (parsed, Some(encoded.clone()))
                }
                value => (value.clone(), None),
            };
            let call = ToolCall {
                id: object
                    .get("id")
                    .map(|value| {
                        value.as_str().map(str::to_owned).ok_or_else(|| {
                            TemplateError::InvalidShape {
                                path: format!("{call_path}.id"),
                                expected: "string".to_owned(),
                            }
                        })
                    })
                    .transpose()?,
                name: required_string(function, "name", &function_path)?,
                arguments,
                arguments_as_supplied,
            };
            call.validate(index)?;
            Ok(call)
        })
        .collect()
}
