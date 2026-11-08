//! Discovery is model-free and does not imply native execution evidence.
use serde_json::{Value, json};
use super::FAITHFULNESS_PROTOCOL;
use crate::assay::native_faithfulness::{EXECUTION_COLLECTION_FORMAT, OBSERVATION_COLLECTION_FORMAT};

pub(super) fn document(operation: &str) -> Value {
    let name = json!({"type":"string","minLength":1,"maxLength":256});
    let ids = json!({"type":"array","minItems":1,"uniqueItems":true,"items":name});
    let bound = json!({"type":"number","minimum":-1,"maximum":1});
    json!({
        "schema_version":1,"protocol":FAITHFULNESS_PROTOCOL,"operation":operation,"backend":"native-faithfulness-int4",
        "evidence":"native_model_replay","native_model_replay":true,"activation_authority":false,
        "supported_tasks":["faithfulness"],"score_space":"full_vocab_sequence_logprob",
        "framing":"One complete canonical UTF-8 JSON report followed by LF; --cases is strict NDJSON.",
        "model_input":{
            "candidate":"Explicit --snapshot, --expected-model-sha256 and --group-width.",
            "baseline":"Qualification additionally requires --baseline-snapshot and --baseline-model-sha256; its --baseline-group-width is independent.",
            "pins":"Independent model-content pins come from trusted conversion evidence; snapshot headers are not trust anchors.",
            "resolution":"Explicit snapshots only; no active-model lookup, conversion, download, fallback or activation."
        },
        "key":{"file_bytes":32,"argv_secret":false,"file_profile":"Existing protected regular key file and owner-controlled output parent."},
        "execution":{
            "source_encoding":"Pinned byte-preserving untrusted source/claim encoder; fixed trusted judge scaffold.",
            "candidate_language":"Exactly C, E or U plus scored EOS; all vocabulary rows contribute to every denominator.",
            "heads":"Complete source plus every deterministic UTF-8 evidence window. One-window sources reuse the full-source head.",
            "max_heads":{"option":"--max-heads","default":32,"minimum":1,"maximum":32},
            "resource_contract":"Quote both model residents, one sequential native engine, complete prepared heads, private rows/metrics and encoded report before IO. One original command deadline and five-axis work ledger cover both arms. Caps never truncate sources or windows."
        },
        "invariants":[
            "Reject duplicate JSON keys, unknown fields, malformed UTF-8, blank rows and non-finite numbers.",
            "Pin exact complete case bytes and validate every split before opening either snapshot.",
            "Every declared ID appears exactly once in its named split. All three splits are nonempty.",
            "Exact source bytes are disjoint across splits. Distinct claims about the same source are allowed within one split; duplicate source/claim pairs are refused.",
            "Only locked_test claims execute. Gold, IDs and split names never enter model prompts.",
            "Each result binds the sealed model, exact request, complete five-axis work and every score head. Independently revalidate full-source partitions, source quotes, scored EOS and final relation or abstention.",
            "Unsupported is a normal relation. Ambiguous distribution, insufficient evidence and conflicting evidence remain typed abstentions.",
            "Paired qualification reuses one parsed population and frozen policy; separate baseline cases and policy overrides are unsupported.",
            "No complete success report follows failed loading, execution, validation, encoding or expired publication. Output storage remains charged through physical write/flush.",
            "File output is protected, staged and create-new. Stdout is not rollbackable."
        ],
        "private_bindings":{
            "execution_collection":{"format":EXECUTION_COLLECTION_FORMAT,"binds":"Ordered complete actual locked-test execution identities with source/claim/gold/budget associations."},
            "observation_collection":{"format":OBSERVATION_COLLECTION_FORMAT,"binds":"Every complete native faithfulness result, all evidence windows, exact quotes, head scores, abstentions and physical work."},
            "privacy":"Private source/claim, row IDs, gold and raw content hashes leave only domain-separated HMAC commitments. Public metadata is the caller-declared dataset/population/host scope and fixed model recipe."
        },
        "metrics":{
            "accuracy":"Correct final decided relations divided by all held-out claims. Abstentions earn no credit.",
            "labels":"Per relation confusion counts, precision, recall and F1. Abstained positive labels count as missed recall; absent support and predictions produce null F1. Macro F1 requires all three declared relations to have defined F1.",
            "brier":"Mean half-sum squared one-hot error of uncalibrated full-source candidate weights, range zero to one; never a factual correctness certificate.",
            "selective_risk":"Incorrect decided claims divided by all decided claims; null when every claim abstains.",
            "coverage":"Fraction of claims with a final decided relation. All native abstention reasons are counted separately.",
            "uncertainty":"Claim-weighted estimates; bootstrap draws whole distinct exact-source clusters, carrying all their claims together. Paired arms use the same source draws. Undefined replicate ratios make that interval null.",
            "minimum_samples":{"default":20,"minimum":5,"unit":"distinct locked-test sources"},
            "insufficient_data":"Typed successful result when too few sources are present; adding correlated claims cannot meet the minimum.",
            "scope":"Gold is supplied human labeling. Byte-verified evidence establishes membership only. Correlated model judgments and candidate weights are not truth, calibration or entailment certificates."
        },
        "qualification":{
            "host_scope_evidence":"caller_declared","gate_basis":"point_estimates",
            "successful_outcomes":["passed","rejected","insufficient_data"],
            "bounds":"Accuracy, macro F1 and coverage decreases are baseline minus candidate; Brier increase is candidate minus baseline. Negative limits require improvement. Undefined required metrics fail.",
            "paired_changes":"Every reported change and its paired interval is candidate minus baseline.",
            "activation_authority":false
        },
        "json_schema":{"$schema":"https://json-schema.org/draft/2020-12/schema","$defs":{
            "policy":{"type":"object","additionalProperties":false,
                "required":["minimum_candidate_weight_ppm","minimum_margin_milli","evidence_window_bytes","max_evidence_windows","max_evidence_spans"],
                "properties":{"minimum_candidate_weight_ppm":{"type":"integer","minimum":0,"maximum":1000000},
                    "minimum_margin_milli":{"type":"integer","minimum":0,"maximum":4294967295_u64},
                    "evidence_window_bytes":{"type":"integer","minimum":1,"maximum":4294967295_u64},
                    "max_evidence_windows":{"type":"integer","minimum":1,"maximum":31},
                    "max_evidence_spans":{"type":"integer","minimum":1,"maximum":31}},
                "description":"max_evidence_spans must not exceed max_evidence_windows; complete evidence capacity is required."},
            "splits":{"type":"object","additionalProperties":false,"required":["development","calibration","locked_test"],
                "properties":{"development":ids,"calibration":ids,"locked_test":ids}},
            "suite":{"type":"object","additionalProperties":false,
                "required":["schema_version","task","dataset_id","dataset_version","named_population","policy","expected_input_sha256","splits"],
                "properties":{"schema_version":{"const":1},"task":{"const":"faithfulness"},"dataset_id":name,"dataset_version":name,
                    "named_population":{"type":"string","minLength":1,"maxLength":1024},"policy":{"$ref":"#/$defs/policy"},
                    "expected_input_sha256":{"type":"string","pattern":"^[0-9a-f]{64}$"},"splits":{"$ref":"#/$defs/splits"}}},
            "case":{"type":"object","additionalProperties":false,
                "required":["schema_version","id","split","source","claim","relation"],
                "properties":{"schema_version":{"const":1},"id":name,"split":{"enum":["development","calibration","locked_test"]},
                    "source":{"type":"string","minLength":1},"claim":{"type":"string","minLength":1},
                    "relation":{"enum":["contradicted","entailed","unsupported"]}}},
            "qualification_gates":{"type":"object","additionalProperties":false,
                "required":["schema_version","max_accuracy_decrease","max_macro_f1_decrease","max_brier_increase","max_coverage_decrease"],
                "properties":{"schema_version":{"const":1},"max_accuracy_decrease":bound,"max_macro_f1_decrease":bound,
                    "max_brier_increase":bound,"max_coverage_decrease":bound}},
            "report_envelope":{"type":"object","additionalProperties":false,
                "required":["schema_version","protocol","operation","evidence","activation_authority","report"],
                "properties":{"schema_version":{"const":1},"protocol":{"const":FAITHFULNESS_PROTOCOL},"operation":{"const":operation},
                    "evidence":{"const":"native_model_replay"},"activation_authority":{"const":false},"report":{"type":"object"}}}
        }}
    })
}
