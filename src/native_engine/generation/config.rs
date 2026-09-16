//! Typed bias-key boundary: JSON string spellings must not alias token IDs.
use super::*;

pub(super) fn deserialize_bias<'de, D: serde::Deserializer<'de>>(deserializer: D)
    -> Result<BTreeMap<u32, i32>, D::Error> {
    struct BiasVisitor;
    impl<'de> serde::de::Visitor<'de> for BiasVisitor {
        type Value = BTreeMap<u32, i32>;
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("bounded canonical decimal token IDs and fixed-point biases")
        }
        fn visit_map<M: serde::de::MapAccess<'de>>(self, mut map: M) -> Result<Self::Value, M::Error> {
            let mut result = BTreeMap::new();
            while let Some(key) = map.next_key::<String>()? {
                if result.len() >= 4096 { return Err(<M::Error as serde::de::Error>::custom("bias entry limit")); }
                let token = key.parse::<u32>().ok().filter(|id| id.to_string() == key)
                    .ok_or_else(|| <M::Error as serde::de::Error>::custom("noncanonical bias token ID"))?;
                if token as usize >= NANBEIGE_VOCAB_SIZE { return Err(<M::Error as serde::de::Error>::custom("bias token outside vocabulary")); }
                let bias = map.next_value::<i32>()?;
                if bias.unsigned_abs() > 100_000 { return Err(<M::Error as serde::de::Error>::custom("bias value limit")); }
                if result.insert(token, bias).is_some() { return Err(<M::Error as serde::de::Error>::custom("duplicate bias token ID")); }
            }
            Ok(result)
        }
    }
    deserializer.deserialize_map(BiasVisitor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::Cell, rc::Rc};
    #[derive(Deserialize)]
    struct Bias { #[serde(deserialize_with = "deserialize_bias")] values: BTreeMap<u32, i32> }
    fn bias(input: &str) -> Result<Bias, ()> {
        let value = canonjson::parse_str(input).map_err(|_| ())?;
        serde_json::from_value(value).map_err(|_| ())
    }
    #[test]
    fn numeric_bias_keys_cannot_alias_or_duplicate() {
        for key in ["01", "+1", " 1", "1e0", "4294967296", "-1"] {
            assert!(bias(&format!(r#"{{"values":{{"1":100,"{key}":200}}}}"#)).is_err());
        }
        assert!(bias(r#"{"values":{"1":100,"1":200}}"#).is_err());
        assert_eq!(bias(r#"{"values":{"0":-100,"1":200}}"#).unwrap().values.get(&1), Some(&200));
    }
    #[test]
    fn canonical_bias_roundtrips_and_domain_bounds_are_enforced() {
        let mut options = GenerationOptions::greedy(1, 100, 0); options.logit_bias_milli.insert(7, 500);
        let bytes = canonjson::canonical_string(&options).unwrap();
        let parsed: GenerationOptions = serde_json::from_value(canonjson::parse_str(&bytes).unwrap()).unwrap();
        assert_eq!(parsed.logit_bias_milli, options.logit_bias_milli);
        assert!(bias(r#"{"values":{"166144":0}}"#).is_err());
        assert!(bias(r#"{"values":{"1":100001}}"#).is_err());
        let entries: BTreeMap<_, _> = (0..4097).map(|id| (id.to_string(), 0)).collect();
        assert!(serde_json::from_value::<Bias>(serde_json::json!({"values":entries})).is_err());
    }
    fn identity() -> ExecutionIdentity {
        let d = Sha256Digest::of_bytes(b"fixture");
        ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
            artifact_format: "fixture".to_owned(), quant_recipe: "bf16-verbatim".to_owned(), packing_set_digest: d,
            tokenizer_digest: d, template_digest: d, task_spec: "generate-v1".to_owned(), taskir_digest: d,
            prompt_digest: d, grammar_compiler_version: "none".to_owned(), schema_digest: d,
            numerics_profile: NumericsProfile::HfBf16Eager, kv_dtype: "bf16".to_owned(), sampler_version: "fixture".to_owned(),
            thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None, calibration_digest: d, decision_policy_digest: d,
            backend_semantic_version: "fixture".to_owned(), host_class: None, compiler_identity: None }
    }
    fn plan(item: &str, sample: u64) -> GenerationPlan {
        GenerationPlan::compile(vec![5, 6], GenerationOptions::greedy(1, 100, 0), identity(), item, sample, GenerationLimits::default()).unwrap()
    }
    #[test]
    fn admitted_identity_binds_stable_item_and_sample_index() {
        let first = plan("one", 0); let other_item = plan("two", 0); let other_sample = plan("one", 1);
        assert_ne!(first.execution_identity().decision_policy_digest, other_item.execution_identity().decision_policy_digest);
        assert_ne!(first.execution_identity().decision_policy_digest, other_sample.execution_identity().decision_policy_digest);
        assert!(first.verify_identity(other_item.execution_identity()).is_err());
        assert!(first.verify_identity(other_sample.execution_identity()).is_err());
        first.verify_identity(plan("one", 0).execution_identity()).unwrap();
    }
    #[test]
    fn option_preflight_rejects_large_stops_and_invalid_sampling_without_a_prompt() {
        let mut options = GenerationOptions::greedy(1, 100, 0);
        options.stop_suffixes.push(vec![b'x'; 4097]); assert!(options.validate(GenerationLimits::default()).is_err());
        options.stop_suffixes.clear();
        options.sampling = GenerationSampling::Seeded { effective_seed: [0; 32], temperature_milli: 0, top_k: None, top_p_ppm: 1000000 };
        assert!(options.validate(GenerationLimits::default()).is_err());
    }
    #[test]
    fn cancellation_during_sink_reservation_drops_permit_without_delivering() {
        struct Control(Rc<Cell<Option<DecodeCancellationKind>>>);
        impl DecodeStepControl for Control {
            fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { self.0.get() }
        }
        struct Permit(Rc<Cell<usize>>);
        impl Drop for Permit { fn drop(&mut self) { self.0.set(self.0.get() + 1); } }
        struct Sink { cause: Rc<Cell<Option<DecodeCancellationKind>>>, drops: Rc<Cell<usize>>, delivered: usize }
        impl DecodeEventSink for Sink {
            type Permit = Permit; type Error = &'static str;
            fn reserve(&mut self, _: &DecodeTokenEvent) -> Result<Permit, Self::Error> {
                self.cause.set(Some(DecodeCancellationKind::Deadline)); Ok(Permit(self.drops.clone()))
            }
            fn permit(&mut self, _: Permit, _: DecodeTokenEvent) -> Result<(), Self::Error> { self.delivered += 1; Ok(()) }
        }
        struct Decoder;
        impl DecodeByteDecoder for Decoder {
            type Error = &'static str;
            fn decode_token_ids(&self, ids: &[u32]) -> Result<Vec<u8>, Self::Error> { Ok(vec![b'a'; ids.len()]) }
        }
        let cause = Rc::new(Cell::new(None)); let drops = Rc::new(Cell::new(0));
        let mut sink = Sink { cause: cause.clone(), drops: drops.clone(), delivered: 0 }; let mut calls = 0;
        let result = plan("one", 0).run(&Decoder, 7, &mut sink, &mut Control(cause), |_| {
            calls += 1; let mut logits = vec![-100.0; NANBEIGE_VOCAB_SIZE]; logits[1] = 10.0; Ok(logits)
        });
        assert!(matches!(result, Err(GenerationError::Cancelled(DecodeCancellationKind::Deadline))));
        assert_eq!(calls, 2); assert_eq!(sink.delivered, 0); assert_eq!(drops.get(), 1);
    }
}
