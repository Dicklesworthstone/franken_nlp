//! Model-free discovery; exact schema/gold JSON stays inside string values.
use serde_json::{Value, json};
use super::EXTRACT_PROTOCOL;
use crate::assay::native_extract::{NATIVE_TASK, EXECUTION_COLLECTION_FORMAT, OBSERVATION_COLLECTION_FORMAT,
    MAX_SCHEMAS, MAX_FIELDS_PER_ROW, MAX_JSON_NODES_PER_ROW, MAX_JSON_DEPTH, MAX_LANGUAGES, MAX_OCCURRENCES_PER_ROW};

pub(super) fn document(operation: &str) -> Value {
    let name = json!({"type":"string","minLength":1,"maxLength":256});
    let ids = json!({"type":"array","minItems":1,"uniqueItems":true,"items":name});
    let decrease = json!({"type":"number","minimum":-1,"maximum":1});
    json!({
        "schema_version":1,"protocol":EXTRACT_PROTOCOL,"operation":operation,"backend":"native-extract-int4",
        "evidence":"native_model_replay","native_model_replay":true,"activation_authority":false,
        "supported_tasks":[NATIVE_TASK],"score_space":"not_computed",
        "framing":"One complete canonical UTF-8 JSON report followed by LF; --cases input is strict NDJSON.",
        "model_input":{
            "candidate":"Explicit --snapshot, --expected-model-sha256 and --group-width.",
            "baseline":"Qualification additionally requires --baseline-snapshot and --baseline-model-sha256; --baseline-group-width is independent.",
            "pins":"Model-content pins must be retained independently from trusted conversion evidence; snapshot headers are not trust anchors.",
            "resolution":"No active model, conversion, network download or fallback is selected."
        },
        "key":{"file_bytes":32,"argv_secret":false,
            "file_profile":"Protected regular key and owner-controlled output parent; no key, schema, gold or document fragments in diagnostics."},
        "admission":{
            "max_schemas":{"option":"--max-schemas","default":16,"minimum":1,"maximum":MAX_SCHEMAS},
            "max_schema_bytes":{"option":"--max-schema-bytes","default":262144,"minimum":1,"maximum":1048576},
            "max_fields_per_row":{"option":"--max-fields-per-row","default":256,"minimum":1,"maximum":MAX_FIELDS_PER_ROW},
            "max_json_nodes_per_row":{"option":"--max-json-nodes-per-row","default":16384,"minimum":1,"maximum":MAX_JSON_NODES_PER_ROW},
            "max_json_depth":{"option":"--max-json-depth","default":64,"minimum":1,"maximum":MAX_JSON_DEPTH},
            "max_languages":{"option":"--max-languages","default":64,"minimum":1,"maximum":MAX_LANGUAGES},
            "max_occurrences_per_row":{"option":"--max-occurrences-per-row","default":16384,"minimum":1,"maximum":MAX_OCCURRENCES_PER_ROW},
            "max_verification_steps":{"option":"--max-verification-steps","default":67108864,"minimum":1,"maximum":4294967296_u64},
            "contract":"Quote raw input, exact gold trees, temporary schema roots, compiler/source work, all resident models and complete output before IO. Enforce catalogue/node/depth/field/language bounds during parsing; never truncate."
        },
        "invariants":[
            "Outer JSON rejects duplicate keys, unknown fields, malformed UTF-8 and non-finite numbers.",
            "schema_json and expected_json are exact JSON strings, never numeric serde Value trees; inner parsers independently enforce the bounded 38-digit decimal domain.",
            "Every schema declaration and gold value passes native validation under the declared limits before either model is opened; source-specific executable feasibility is checked again during native planning.",
            "A source_membership schema has verbatim fields; structural schemas cannot silently weaken source annotations.",
            "The suite pins exact complete case bytes; each declared ID occurs once in its named split, and exact source units are unique across the corpus.",
            "Only locked_test documents execute. All development/calibration rows remain validated and privately bound.",
            "Only exact source text, the selected schema and fixed task policy enter model planning; expected JSON, IDs, language and split metadata do not enter prompts.",
            "Complete native results are independently checked for schema validity, source membership, terminal EOS, execution identity and actual generated-length work.",
            "Both arms use the same parsed cases, exact schema catalogue and fixed compiler/source policy under shared model, mask, verification and bootstrap allowances.",
            "Gold source checking across all splits consumes the shared verification allowance once; both native arms spend the remainder.",
            "No complete successful report is published after failed loading, replay, finalization or encoding.",
            "Protected file publication is create-new and staged; stdout writes are not rollbackable."
        ],
        "private_bindings":{
            "execution_collection":{"commitment_field":"bindings.execution_collection","format":EXECUTION_COLLECTION_FORMAT,
                "row_count_field":"bindings.execution_collection_rows","binds":"Ordered full native execution identities for actual locked-test documents."},
            "observation_collection":{"commitment_field":"bindings.observation_collection","format":OBSERVATION_COLLECTION_FORMAT,
                "row_count_field":"bindings.observation_collection_rows","binds":"Exact schema/source/gold association and complete native JSON, source occurrences, tokens and work."},
            "privacy":"Private values and schemas leave only as domain-separated HMAC commitments. Public metadata includes caller-declared dataset/population, schema IDs, languages and host scope."
        },
        "metrics":{
            "fields":"Terminal scalars and empty containers are fields. Matching requires the same typed structural location and exact value; object-key order is irrelevant and array order is significant.",
            "typed_locations":"Object key '0' differs from array index 0. Parent container type changes cannot produce accidental path matches.",
            "exact_numbers":"Equivalent decimal spellings compare equal without f64; strings remain exact Unicode scalar sequences with no case, whitespace or Unicode normalization.",
            "errors":"Missing fields are false negatives, extra fields are false positives, and a wrong value at an existing field contributes one of each.",
            "document_accuracy":"Fraction of exact whole-JSON document matches; field support includes null and empty object/array values.",
            "uncertainty":"Deterministic whole-document bootstrap recomputes micro ratios from document counts. Paired arms use the same draws; by-schema/language breakdowns are point estimates.",
            "undefined":"Undefined ratios are null; an interval is null when any required bootstrap ratio is undefined.",
            "minimum_samples":{"default":20,"minimum":5,"unit":"locked-test documents"},
            "insufficient_data":"Successful typed result below minimum_samples; no measured-quality or qualification-pass claim.",
            "scope":"Exact agreement measures supplied human gold. Schema validity and source membership do not prove semantic correctness; ambiguous source occurrences are not silently selected. No probability, ECE/Brier or calibration is computed."
        },
        "qualification":{"host_scope_evidence":"caller_declared","gate_basis":"point_estimates",
            "successful_outcomes":["passed","rejected","insufficient_data"],
            "decrease_bounds":"Baseline minus candidate; negative bounds require improvement. Undefined required metrics fail their gates.",
            "paired_changes":"Candidate minus baseline. Per-document field FP+FN determines candidate_better, candidate_worse and equal counts.",
            "activation_authority":false},
        "json_schema":{"$schema":"https://json-schema.org/draft/2020-12/schema","$defs":{
            "schema_definition":{"type":"object","additionalProperties":false,
                "required":["id","schema_json","grounding"],"properties":{"id":name,
                    "schema_json":{"type":"string","minLength":1,"description":"Exact raw JSON in the native finite schema subset; bounded independently by --max-schema-bytes."},
                    "grounding":{"enum":["structural","source_membership"]}}},
            "splits":{"type":"object","additionalProperties":false,"required":["development","calibration","locked_test"],
                "properties":{"development":ids,"calibration":ids,"locked_test":ids}},
            "suite":{"type":"object","additionalProperties":false,
                "required":["schema_version","task","dataset_id","dataset_version","named_population","schemas","expected_input_sha256","splits"],
                "properties":{"schema_version":{"const":1},"task":{"const":NATIVE_TASK},"dataset_id":name,"dataset_version":name,
                    "named_population":{"type":"string","minLength":1,"maxLength":1024},
                    "schemas":{"type":"array","minItems":1,"maxItems":MAX_SCHEMAS,"items":{"$ref":"#/$defs/schema_definition"}},
                    "expected_input_sha256":{"type":"string","pattern":"^[0-9a-f]{64}$"},"splits":{"$ref":"#/$defs/splits"}}},
            "case":{"type":"object","additionalProperties":false,
                "required":["schema_version","id","split","text","schema_id","language","expected_json"],
                "properties":{"schema_version":{"const":1},"id":name,"split":{"enum":["development","calibration","locked_test"]},
                    "text":{"type":"string","minLength":1},"schema_id":name,
                    "language":{"type":"string","minLength":1,"maxLength":63,"pattern":"^[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*$"},
                    "expected_json":{"type":"string","minLength":1,"description":"Exact JSON gold; parsed independently with node/depth/field bounds and validated against the row's declared schema."}}},
            "qualification_gates":{"type":"object","additionalProperties":false,
                "required":["schema_version","max_field_f1_decrease","max_field_recall_decrease","max_document_accuracy_decrease"],
                "properties":{"schema_version":{"const":1},"max_field_f1_decrease":decrease,
                    "max_field_recall_decrease":decrease,"max_document_accuracy_decrease":decrease}},
            "report_envelope":{"type":"object","additionalProperties":false,
                "required":["schema_version","protocol","operation","evidence","activation_authority","report"],
                "properties":{"schema_version":{"const":1},"protocol":{"const":EXTRACT_PROTOCOL},"operation":{"const":operation},
                    "evidence":{"const":"native_model_replay"},"activation_authority":{"const":false},"report":{"type":"object"}}}
        }}
    })
}
