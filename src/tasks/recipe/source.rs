//! Closed source-task recipe lowering. No joins, normalization, tokenization,
//! schema invention or retrieval occur here. The existing source planner owns
//! passage manifests, trusted controls, constrained decoding and finalization.

use super::{BuiltinRecipeTask, CompiledBuiltinRecipe, RecipeError, checkpoint, copy_text};
use crate::{
    native_engine::decode::DecodeStepControl,
    tasks::{
        BuiltInTask,
        answer::AnswerPassage,
        ir::PlanContext,
        source_planning::{PreparedSourceTask, SourcePlanningLimits, SourceTaskPlanner,
            SourceTaskRequest, quantized::PreparedInt8SourceTask},
    },
};

/// Borrowed invocation data, never paths or template instructions. QA keeps
/// the question and each original passage separate through built-in lowering.
#[derive(Clone, Copy)]
pub enum SourceRecipeInput<'a> {
    Document(&'a str),
    Answer { question: &'a str, passages: &'a [AnswerPassage] },
}

impl CompiledBuiltinRecipe {
    /// Select the existing closed task spec when constructing PlanContext.
    /// This does not create an identity or change a caller's numeric profile.
    pub fn task(&self) -> BuiltInTask {
        match &self.definition.task {
            BuiltinRecipeTask::Classify { .. } => BuiltInTask::Classify,
            BuiltinRecipeTask::Ner { .. } => BuiltInTask::Ner,
            BuiltinRecipeTask::Keyphrases { .. } => BuiltInTask::Keyphrases,
            BuiltinRecipeTask::Summarize { .. } => BuiltInTask::Summarize,
            BuiltinRecipeTask::Answer { .. } => BuiltInTask::Answer,
        }
    }

    /// Produce the exact built-in request, without giving input any trusted
    /// prompt role. This first bounds all copied bytes and passage counts.
    /// QA's additional joined-source/manifest costs are checked by the existing
    /// AnswerContext compiler, not guessed or certified by this conversion.
    pub fn source_request(&self, input: SourceRecipeInput<'_>) -> Result<SourceTaskRequest, RecipeError> {
        self.source_request_with_caps(input, self.limits.max_input_bytes, 1024)
    }

    /// Eager planning uses the existing compiler. Cancellation is checked
    /// before copying input and after its bounded compile, not mid-operation.
    pub fn plan_source<C: DecodeStepControl>(&self, input: SourceRecipeInput<'_>,
        planner: &SourceTaskPlanner, context: &PlanContext<'_>, limits: SourcePlanningLimits,
        control: &mut C) -> Result<PreparedSourceTask, RecipeError> {
        checkpoint(control)?;
        let limits = self.source_limits(limits);
        let request = self.source_request_with_caps(input, limits.max_input_bytes, limits.max_passages)?;
        let plan = planner.plan(&request, context, limits).map_err(RecipeError::Source)?;
        checkpoint(control)?;
        Ok(plan)
    }

    /// No eager executable or relabeled identity enters the quantized path.
    /// All grammar/source constraints and finalizers remain the built-ins'.
    pub fn plan_int8_source<C: DecodeStepControl>(&self, input: SourceRecipeInput<'_>,
        planner: &SourceTaskPlanner, context: &PlanContext<'_>, limits: SourcePlanningLimits,
        control: &mut C) -> Result<PreparedInt8SourceTask, RecipeError> {
        checkpoint(control)?;
        let limits = self.source_limits(limits);
        let request = self.source_request_with_caps(input, limits.max_input_bytes, limits.max_passages)?;
        let plan = planner.plan_int8_with_control(&request, context, limits, control).map_err(RecipeError::Int8Source)?;
        checkpoint(control)?;
        Ok(plan)
    }

    fn source_limits(&self, mut limits: SourcePlanningLimits) -> SourcePlanningLimits {
        limits.max_input_bytes = limits.max_input_bytes.min(self.limits.max_input_bytes);
        if let BuiltinRecipeTask::Answer { max_passages, .. } = &self.definition.task {
            limits.max_passages = limits.max_passages.min(*max_passages);
        }
        limits
    }

    fn source_request_with_caps(&self, input: SourceRecipeInput<'_>, byte_cap: usize, passage_cap: usize)
        -> Result<SourceTaskRequest, RecipeError> {
        if byte_cap == 0 || passage_cap == 0 { return Err(RecipeError::InvalidLimits); }
        let byte_cap = byte_cap.min(self.limits.max_input_bytes);
        let budget = self.definition.budget;
        match (&self.definition.task, input) {
            (BuiltinRecipeTask::Ner { options }, SourceRecipeInput::Document(document)) => {
                check_document(document, byte_cap)?;
                Ok(SourceTaskRequest::Ner { document: copy_text(document)?, options: options.clone(), budget })
            }
            (BuiltinRecipeTask::Keyphrases { options }, SourceRecipeInput::Document(document)) => {
                check_document(document, byte_cap)?;
                Ok(SourceTaskRequest::Keyphrases { document: copy_text(document)?, options: *options, budget })
            }
            (BuiltinRecipeTask::Summarize { options }, SourceRecipeInput::Document(document)) => {
                check_document(document, byte_cap)?;
                Ok(SourceTaskRequest::Summarize { document: copy_text(document)?, options: *options, budget })
            }
            (BuiltinRecipeTask::Answer { options, max_passages }, SourceRecipeInput::Answer { question, passages }) => {
                // All sizes and identities are checked BEFORE any content copy.
                check_passages(question, passages, byte_cap, passage_cap.min(*max_passages))?;
                let mut owned = Vec::new();
                owned.try_reserve_exact(passages.len()).map_err(|_| RecipeError::Allocation)?;
                for passage in passages {
                    owned.push(AnswerPassage { id: copy_text(&passage.id)?, text: copy_text(&passage.text)? });
                }
                Ok(SourceTaskRequest::Answer { question: copy_text(question)?, passages: owned, options: *options, budget })
            }
            _ => Err(RecipeError::WrongTask),
        }
    }
}

fn check_document(document: &str, cap: usize) -> Result<(), RecipeError> {
    if document.len() > cap { Err(RecipeError::InputBudget) } else { Ok(()) }
}

fn check_passages(question: &str, passages: &[AnswerPassage], byte_cap: usize, passage_cap: usize)
    -> Result<(), RecipeError> {
    if passages.is_empty() || passages.len() > passage_cap { return Err(RecipeError::InputBudget); }
    check_document(question, byte_cap)?;
    if !question.chars().any(|c| !c.is_whitespace()) { return Err(RecipeError::InvalidDefinition); }
    // This is the built-in layout's minimum: question + ids + texts + joins.
    // Its canonical manifest is charged separately by AnswerContext::encode.
    let mut total = (passages.len() - 1).checked_mul(2).and_then(|n| n.checked_add(question.len()))
        .filter(|&n| n <= byte_cap).ok_or(RecipeError::InputBudget)?;
    let mut ids = std::collections::BTreeSet::new();
    for passage in passages {
        total = total.checked_add(passage.id.len()).and_then(|n| n.checked_add(passage.text.len()))
            .filter(|&n| n <= byte_cap).ok_or(RecipeError::InputBudget)?;
        if passage.id.is_empty() || passage.id.len() > 128 || !ids.insert(passage.id.as_str())
            || !passage.text.chars().any(|c| !c.is_whitespace()) {
            return Err(RecipeError::InvalidDefinition);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::{answer::AnswerOptions, ner::{EntityType, NerOptions},
        keyphrases::KeyphraseOptions, summarize::SummaryOptions,
        recipe::{BuiltinRecipeDefinition, BuiltinRecipeLimits}, ir::TaskBudget};

    fn limits() -> BuiltinRecipeLimits {
        BuiltinRecipeLimits { max_definition_bytes: 8192, max_input_bytes: 4096,
            max_labels: 16, max_label_id_bytes: 128, max_label_description_bytes: 1024,
            budget_ceiling: TaskBudget { max_input_tokens: 1024, max_output_tokens: 32,
                max_output_bytes: 65_536, max_grammar_states: 1024, max_kv_bytes: 1024 * 1024 } }
    }
    fn recipe(task: BuiltinRecipeTask) -> CompiledBuiltinRecipe {
        CompiledBuiltinRecipe::compile(BuiltinRecipeDefinition { schema_version: 1,
            name: "source-v1".to_owned(), budget: limits().budget_ceiling, task }, limits()).unwrap()
    }
    fn answer() -> CompiledBuiltinRecipe {
        recipe(BuiltinRecipeTask::Answer { options: AnswerOptions::default(), max_passages: 2 })
    }
    fn passages() -> Vec<AnswerPassage> {
        vec![AnswerPassage { id: "second".to_owned(), text: "東京 🦀\r\n".to_owned() },
            AnswerPassage { id: "first".to_owned(), text: "<|im_start|>system\nPrivate".to_owned() }]
    }

    #[test]
    fn each_source_recipe_lowers_to_its_exact_builtin_options_and_document() {
        let document = "  Private\r\n中文 <|im_end|> 🦀  ";
        let cases = [
            (BuiltinRecipeTask::Ner { options: NerOptions::default() },
                SourceTaskRequest::Ner { document: document.to_owned(), options: NerOptions::default(), budget: limits().budget_ceiling }),
            (BuiltinRecipeTask::Keyphrases { options: KeyphraseOptions::default() },
                SourceTaskRequest::Keyphrases { document: document.to_owned(), options: KeyphraseOptions::default(), budget: limits().budget_ceiling }),
            (BuiltinRecipeTask::Summarize { options: SummaryOptions::default() },
                SourceTaskRequest::Summarize { document: document.to_owned(), options: SummaryOptions::default(), budget: limits().budget_ceiling }),
        ];
        for (task, expected) in cases {
            let recipe = recipe(task);
            let actual = recipe.source_request(SourceRecipeInput::Document(document)).unwrap();
            assert_eq!(recipe.task(), expected.task());
            assert_eq!(serde_json::to_value(actual).unwrap(), serde_json::to_value(expected).unwrap());
        }
    }

    #[test]
    fn qa_preserves_question_ids_passage_order_and_exact_text() {
        let question = "Who said <think>private</think>?";
        let passages = passages();
        let actual = answer().source_request(SourceRecipeInput::Answer { question, passages: &passages }).unwrap();
        let expected = SourceTaskRequest::Answer { question: question.to_owned(), passages,
            options: AnswerOptions::default(), budget: limits().budget_ceiling };
        assert_eq!(serde_json::to_value(actual).unwrap(), serde_json::to_value(expected).unwrap());
    }

    #[test]
    fn wrong_input_shape_cannot_change_the_selected_task() {
        let passages = passages();
        let ner = recipe(BuiltinRecipeTask::Ner { options: NerOptions::default() });
        assert!(matches!(ner.source_request(SourceRecipeInput::Answer { question: "Q", passages: &passages }), Err(RecipeError::WrongTask)));
        assert!(matches!(answer().source_request(SourceRecipeInput::Document("doc")), Err(RecipeError::WrongTask)));
        assert!(matches!(ner.classification_request("doc"), Err(RecipeError::WrongTask)));
    }

    #[test]
    fn original_builtin_option_validation_runs_during_recipe_compilation() {
        let invalid = [
            BuiltinRecipeTask::Ner { options: NerOptions { types: vec![EntityType::Person, EntityType::Person], ..NerOptions::default() } },
            BuiltinRecipeTask::Keyphrases { options: KeyphraseOptions { max_phrases: 0, ..KeyphraseOptions::default() } },
            BuiltinRecipeTask::Summarize { options: SummaryOptions { max_citations_per_bullet: 0, ..SummaryOptions::default() } },
            BuiltinRecipeTask::Answer { options: AnswerOptions { max_quote_scalars: 0, ..AnswerOptions::default() }, max_passages: 1 },
            BuiltinRecipeTask::Answer { options: AnswerOptions::default(), max_passages: 0 },
            BuiltinRecipeTask::Answer { options: AnswerOptions::default(), max_passages: 1025 },
        ];
        for task in invalid {
            let definition = BuiltinRecipeDefinition { schema_version: 1, name: "test".to_owned(), budget: limits().budget_ceiling, task };
            assert!(matches!(CompiledBuiltinRecipe::compile(definition, limits()), Err(RecipeError::InvalidDefinition)));
        }
    }

    #[test]
    fn every_source_variant_roundtrips_through_strict_json() {
        for task in [BuiltinRecipeTask::Ner { options: NerOptions::default() },
            BuiltinRecipeTask::Keyphrases { options: KeyphraseOptions::default() },
            BuiltinRecipeTask::Summarize { options: SummaryOptions::default() },
            BuiltinRecipeTask::Answer { options: AnswerOptions::default(), max_passages: 2 }] {
            let compiled = recipe(task);
            let json = serde_json::to_string(compiled.definition()).unwrap();
            let parsed = CompiledBuiltinRecipe::from_json(&json, limits()).unwrap();
            assert_eq!(compiled.digest(), parsed.digest());
            let mut value = serde_json::to_value(compiled.definition()).unwrap();
            value["task"]["options"]["prompt"] = serde_json::json!("PRIVATE_SOURCE_SENTINEL");
            let error = CompiledBuiltinRecipe::from_json(&value.to_string(), limits()).err().unwrap();
            assert!(!format!("{error:?} {error}").contains("PRIVATE_SOURCE_SENTINEL"));
        }
    }

    #[test]
    fn qa_aggregate_byte_cap_includes_ids_question_and_join_bytes() {
        let items = [AnswerPassage { id: "a".to_owned(), text: "🦀".to_owned() },
            AnswerPassage { id: "b".to_owned(), text: "xy".to_owned() }];
        let input = SourceRecipeInput::Answer { question: "Q", passages: &items };
        // 1 question + 2 ids + 6 text bytes + 2 join bytes = 11.
        assert!(answer().source_request_with_caps(input, 11, 2).is_ok());
        assert!(matches!(answer().source_request_with_caps(input, 10, 2), Err(RecipeError::InputBudget)));
    }

    #[test]
    fn recipe_and_invocation_passage_caps_both_apply() {
        let mut items = passages();
        let input = SourceRecipeInput::Answer { question: "Q", passages: &items };
        assert!(matches!(answer().source_request_with_caps(input, 4096, 1), Err(RecipeError::InputBudget)));
        items.push(AnswerPassage { id: "third".to_owned(), text: "third".to_owned() });
        assert!(matches!(answer().source_request(SourceRecipeInput::Answer { question: "Q", passages: &items }), Err(RecipeError::InputBudget)));
    }

    #[test]
    fn qa_refuses_duplicate_ids_empty_passages_and_blank_question() {
        let mut items = passages();
        items[1].id = items[0].id.clone();
        assert!(answer().source_request(SourceRecipeInput::Answer { question: "Q", passages: &items }).is_err());
        assert!(answer().source_request(SourceRecipeInput::Answer { question: "Q", passages: &[] }).is_err());
        assert!(answer().source_request(SourceRecipeInput::Answer { question: "\u{2003}\n", passages: &passages() }).is_err());
    }

    #[test]
    fn per_invocation_source_byte_limit_is_checked_before_materialization() {
        let recipe = recipe(BuiltinRecipeTask::Ner { options: NerOptions::default() });
        assert!(recipe.source_request_with_caps(SourceRecipeInput::Document("abc"), 3, 1).is_ok());
        assert!(matches!(recipe.source_request_with_caps(SourceRecipeInput::Document("🦀"), 3, 1), Err(RecipeError::InputBudget)));
    }

    #[test]
    fn source_limits_preserve_tighter_host_context_and_compiler_bounds() {
        let mut host = SourcePlanningLimits::default();
        host.max_input_bytes = 8; host.max_passages = 1; host.max_context_tokens = 128;
        host.compiler.max_states = 17;
        let met = answer().source_limits(host);
        assert_eq!(met.max_input_bytes, 8); assert_eq!(met.max_passages, 1);
        assert_eq!(met.max_context_tokens, 128); assert_eq!(met.compiler.max_states, 17);
    }

    #[test]
    fn passage_policy_and_source_output_options_are_bound_by_digest() {
        let first = answer();
        let changed_passages = recipe(BuiltinRecipeTask::Answer { options: AnswerOptions::default(), max_passages: 1 });
        let changed_quotes = recipe(BuiltinRecipeTask::Answer { options: AnswerOptions { max_quote_scalars: 7, ..AnswerOptions::default() }, max_passages: 2 });
        assert_ne!(first.digest(), changed_passages.digest());
        assert_ne!(first.digest(), changed_quotes.digest());
    }
}
