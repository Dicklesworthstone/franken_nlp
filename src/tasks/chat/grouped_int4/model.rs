//! Model-derived pinned planners for the source-candidate INT4 backend.
//! No weight cloning/loading, runtime, artifact activation or profile fallback.
use super::*;
use crate::{native_engine::{
    generation::grouped_int4::bind_candidate_model,
    strict_int4::{Int4ModelIdentity, STRICT_INT4_EXECUTION},
}, tokenizer::pinned_controls};

impl Int4ChatPlanner {
    /// Construct from the identity of actual loaded INT4 weights, including a
    /// hosted ResidentInt4. Callers need not invent source/content/packing
    /// fields or a tokenizer/control census. The model constructor owns the
    /// content fingerprint; the ordinary compiler owns prompt/task/policy facts.
    ///
    /// The caller admits pinned tokenizer/planner storage separately, before
    /// entering here. This factory does not reserve memory or create a runtime.
    /// A planner does not retain weights or confer permission to activate them.
    pub fn for_model(model: &Int4ModelIdentity, ceiling: TaskBudget, limits: ChatLimits)
        -> Result<Self, Int4ChatError> {
        ceiling.validate().map_err(|_| ChatError::Limit("host task ceiling"))?;
        let identity = bind_candidate_model(uncompiled_identity(), model)?;
        let controls = pinned_controls::pinned().map_err(|_| ChatError::Identity)?;
        let eos = pinned_eos(controls.template_controls())?;
        Self::pinned(controls.template_controls(), eos, identity, model.group_width(), ceiling, limits)
    }

    /// Terminal EOS verified against the actual pinned tokenizer at creation.
    /// Generation options must retain this ID; it cannot be changed or banned.
    pub fn eos_token_id(&self) -> u32 { self.compiler.eos }
}
fn pinned_eos(controls: &TemplateControlIds) -> Result<u32, ChatError> {
    let mut matches = controls.entries().iter().filter(|entry| entry.special && entry.surface == IM_END);
    let eos = matches.next().ok_or(ChatError::Identity)?.id;
    if matches.next().is_some() { return Err(ChatError::Identity); }
    Ok(eos)
}
fn uncompiled_identity() -> ExecutionIdentity {
    let absent = Sha256Digest::of_bytes(b"null");
    // This value is private and never escapes unbound. bind_candidate_model
    // replaces every model field; the shared task compiler replaces task,
    // tokenizer/template, prompt, sampler and policy fields before admission.
    ExecutionIdentity { schema_version: 1, source_revision: "unbound".to_owned(),
        logical_model_digest: absent, artifact_format: "unbound".to_owned(), quant_recipe: "unbound".to_owned(),
        packing_set_digest: absent, tokenizer_digest: absent, template_digest: absent,
        task_spec: "uncompiled-source-int4-text".to_owned(), taskir_digest: absent, prompt_digest: absent,
        grammar_compiler_version: "none".to_owned(), schema_digest: absent,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "uncompiled".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: absent, decision_policy_digest: absent,
        backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(), host_class: None, compiler_identity: None }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn complete_pinned_registry_agrees_with_configured_tokenizer_eos() {
        let registry = pinned_controls::pinned().unwrap();
        let eos = pinned_eos(registry.template_controls()).unwrap();
        assert_eq!(Some(eos), EmbeddedTokenizer::pinned().unwrap().eos_token_id());
        assert!(registry.tokenizer_special_ids().ids().contains(&eos));
    }
    #[test]
    fn private_unbound_model_fields_cannot_become_executable_plans() {
        let registry = pinned_controls::pinned().unwrap();
        let eos = pinned_eos(registry.template_controls()).unwrap();
        for width in [1, 128, MAX_MODEL_K] {
            assert!(Int4GenerationPlan::compile(vec![1], GenerationOptions::greedy(1, 64, eos),
                uncompiled_identity(), width, "unbound-unit-fixture", 0, GenerationLimits::default()).is_err());
        }
        let identity = uncompiled_identity();
        assert_eq!(identity.numerics_profile, NumericsProfile::StrictQuantized { version: 2 });
        assert_eq!(identity.backend_semantic_version, STRICT_INT4_EXECUTION);
        assert_eq!(identity.thinking_mode, ThinkingMode::Disabled);
        assert_eq!(identity.tool_mode, ToolMode::None);
    }
    #[test]
    #[ignore = "requires controller-admitted pinned source and full-model execution"]
    fn model_present_factory_generate_chat_and_reuse() {
        use crate::{artifact::converter::{ConversionSourceManifest, prepare_nanbeige42_input},
            native_engine::{strict_int4::{Int4ModelWeights, Int4ModelLoadBudget, Int4ModelLoadRequirement,
                Int4MemoryRequirement, Int4MemoryBudget, Int4RunBudget}, kv::KV_BYTES_PER_TOKEN}};
        struct Continue;
        impl DecodeStepControl for Continue {
            fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
        }
        let directory = std::env::var("FNLP_INT4_TEST_SOURCE_DIR").expect("explicit source directory required");
        let manifest = std::env::var("FNLP_INT4_TEST_SOURCE_MANIFEST").expect("explicit pinned manifest required");
        let source = prepare_nanbeige42_input(directory, ConversionSourceManifest::load_pinned(manifest).unwrap(), false, 65536).unwrap();
        let load = Int4ModelLoadRequirement::for_group_width(128).unwrap();
        let weights = Int4ModelWeights::from_prepared(&source, 128, Int4ModelLoadBudget {
            max_payload_bytes: load.peak_payload_bytes().unwrap(), max_source_values: load.source_values,
            max_groups: load.groups, max_shard_revalidation_bytes: load.shard_revalidation_bytes,
        }, &mut Continue).unwrap();
        let context = 256;
        let budget = TaskBudget { max_input_tokens: 128, max_output_tokens: 4, max_output_bytes: 65536,
            max_grammar_states: 1, max_kv_bytes: (context * KV_BYTES_PER_TOKEN) as u64 };
        let planner = Int4ChatPlanner::for_model(weights.identity(), budget, ChatLimits::default()).unwrap();
        let bound = bind_candidate_model(planner.execution_identity().clone(), weights.identity()).unwrap();
        assert_eq!(&bound, planner.execution_identity());
        assert_eq!(planner.group_width(), weights.group_width());
        let options = GenerationOptions::greedy(4, 256, planner.eos_token_id());
        let generate = planner.plan_generate(&GenerateRequest { item_id: "real-text".to_owned(), sample_index: 0,
            prompt: "Reply with a short greeting.".to_owned(), generation: options.clone(), budget }).unwrap();
        let chat = planner.plan_chat(&ChatRequest { item_id: "real-chat".to_owned(), sample_index: 0,
            messages: vec![ChatMessage { role: ChatRole::User, content: "Reply with a short greeting.".to_owned() }],
            generation: options, budget }).unwrap();
        let need = Int4MemoryRequirement::for_context(context, 128).unwrap();
        let mut engine = StrictInt4Engine::new(&weights, context, Int4MemoryBudget { max_kv_bytes: need.kv_bytes,
            max_rope_bytes: need.rope_bytes, max_scratch_payload_bytes: need.scratch_payload_bound }).unwrap();
        let run = |plan: &PreparedInt4Chat, engine: &mut StrictInt4Engine<'_>| {
            plan.execute(plan.execution_identity(), engine, 1, Int4GenerationBudget {
                native: Int4RunBudget::exact(plan.planned_work()), max_kv_bytes: need.kv_bytes,
                max_sampler_bytes: plan.native_plan().sampler_bytes(),
            }, &mut Continue).unwrap()
        };
        let a = run(&generate, &mut engine);
        assert!(engine.kv_cache().all_slots_have_len(0)); assert!(!engine.is_poisoned());
        let b = run(&chat, &mut engine);
        assert_eq!(a.result.token_ids, b.result.token_ids); assert_eq!(a.result.content, b.result.content);
        assert_eq!(a.model_work, b.model_work);
        let repeated = run(&generate, &mut engine);
        assert!(a == repeated); assert!(engine.kv_cache().all_slots_have_len(0));
    }
}
