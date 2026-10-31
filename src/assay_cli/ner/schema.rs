//! Model-free discovery for strict native span evaluation and comparison.
use serde_json::{Value, json};
use super::NER_PROTOCOL;
use crate::{assay::native_ner::{EXECUTION_COLLECTION_FORMAT, OBSERVATION_COLLECTION_FORMAT,
    MAX_SPANS_PER_ROW, MAX_OCCURRENCES_PER_ROW, MAX_LANGUAGES}, tasks::ner::EntityType};

pub(super) fn document(operation: &str) -> Value {
    let name = json!({"type":"string","minLength":1,"maxLength":256});
    let ids = json!({"type":"array","minItems":1,"uniqueItems":true,"items":name});
    let kinds: Vec<_> = EntityType::ALL.iter().map(|kind| kind.label()).collect();
    let decrease = json!({"type":"number","minimum":-1,"maximum":1});
    json!({
        "schema_version":1,"protocol":NER_PROTOCOL,"operation":operation,"backend":"native-ner-int4",
        "evidence":"native_model_replay","native_model_replay":true,"activation_authority":false,
        "supported_tasks":["named_entity_recognition"],"score_space":"not_computed",
        "framing":"One complete canonical UTF-8 JSON report followed by LF; --cases input is NDJSON.",
        "model_input":{
            "candidate":"Explicit --snapshot, --expected-model-sha256 and --group-width.",
            "baseline":"Qualification additionally requires --baseline-snapshot and --baseline-model-sha256; --baseline-group-width is independent.",
            "pins":"Supply independent model-content pins from trusted conversion evidence; snapshot headers are not trust anchors.",
            "resolution":"These commands select no active catalog, conversion, download or fallback."
        },
        "key":{"file_bytes":32,"argv_secret":false,
            "file_profile":"Protected regular key and owner-controlled output parent; no key or source bytes in diagnostics."},
        "span_admission":{
            "max_spans_per_row":{"option":"--max-spans-per-row","default":64,"minimum":1,"maximum":MAX_SPANS_PER_ROW},
            "max_occurrences_per_row":{"option":"--max-occurrences-per-row","default":16384,"minimum":1,"maximum":MAX_OCCURRENCES_PER_ROW},
            "max_languages":{"option":"--max-languages","default":64,"minimum":1,"maximum":MAX_LANGUAGES},
            "max_verification_steps":{"option":"--max-verification-steps","default":67108864,"minimum":1,"maximum":4294967296_u64},
            "contract":"Quote declared capacities before input/model IO. Check suite max_entities before cases, then each span-array length and language geometry before typed gold rows. Never truncate a gold or occurrence list."
        },
        "invariants":[
            "Reject duplicate JSON keys, unknown fields, malformed UTF-8 and non-finite numbers.",
            "Pin the exact complete case file and validate every declared split before opening a model.",
            "Every declared ID occurs exactly once in its named split; all splits and exact-text source units are disjoint.",
            "Only locked_test documents execute. Development and calibration rows remain validated private binding inputs.",
            "Gold uses half-open UTF-8 byte intervals at scalar boundaries; the evaluator independently computes scalar offsets.",
            "Each gold interval has exactly one declared entity type; duplicate intervals are refused even when their types differ. Distinct nested and overlapping intervals are allowed.",
            "Only text and frozen NER options enter source requests. Gold, row IDs, language metadata and split names never enter prompts.",
            "Each complete native result is independently checked against the exact source, model identity and declared type set.",
            "Repeated-surface ambiguity retains all exact occurrences, selects no hidden location and earns no span match.",
            "Both comparison arms use the same parsed cases and fixed source policy with shared nonrenewable model, mask, verification and bootstrap allowances.",
            "No complete successful report is published after failed loading, replay, finalization or output encoding.",
            "File output is create-new, staged and protected; stdout writes are not rollbackable."
        ],
        "private_bindings":{
            "execution_collection":{"commitment_field":"bindings.execution_collection","format":EXECUTION_COLLECTION_FORMAT,
                "row_count_field":"bindings.execution_collection_rows","binds":"Ordered complete per-document execution identities for actual locked-test replay."},
            "observation_collection":{"commitment_field":"bindings.observation_collection","format":OBSERVATION_COLLECTION_FORMAT,
                "row_count_field":"bindings.observation_collection_rows","binds":"Ordered source association, byte/scalar gold, complete native entities and all ambiguous occurrences, tokens and actual work."},
            "privacy":"Private content is exported only through domain-separated HMAC commitments. Public metadata includes caller-declared dataset/population, entity types, languages and host scope."
        },
        "metrics":{
            "exact":"One-to-one matches require identical byte boundaries and entity type. An extra duplicate prediction is a false positive.",
            "type_relaxed":"One-to-one matches require identical byte boundaries and ignore only entity type; this is not overlap matching.",
            "document_accuracy":"Fraction of documents with no exact false positive or false negative, including correct empty sets.",
            "anchoring":"Reports anchored/ambiguous prediction counts and fractions, total ambiguous candidate spans and duplicate anchored predictions.",
            "language_scope":"By-language counts use caller-declared language tags; they are not language detection or independently sampled language evidence.",
            "uncertainty":"Deterministic bootstrap resamples whole documents and recomputes corpus ratios from their counts. Paired arms use identical document draws.",
            "undefined":"A zero denominator yields null, never a fabricated perfect score. An interval is null if any bootstrap replicate has an undefined ratio.",
            "minimum_samples":{"default":20,"minimum":5,"unit":"locked-test documents"},
            "insufficient_data":"Successful typed result below minimum_samples; no measured-quality or qualification-pass claim.",
            "scope":"Source membership proves occurrence, not human label correctness. No mention probability, ECE/Brier, acceptance threshold or calibration is computed."
        },
        "qualification":{
            "host_scope_evidence":"caller_declared","gate_basis":"point_estimates",
            "successful_outcomes":["passed","rejected","insufficient_data"],
            "decrease_bounds":"Every gate uses baseline minus candidate. Negative bounds require improvement; an undefined required score fails its gate.",
            "paired_changes":"Candidate minus baseline; document-level exact FP+FN also reports candidate_better, candidate_worse and equal counts.",
            "activation_authority":false
        },
        "json_schema":{"$schema":"https://json-schema.org/draft/2020-12/schema","$defs":{
            "entity_type":{"enum":kinds},
            "options":{"type":"object","additionalProperties":false,
                "required":["types","max_entities","max_mention_scalars"],"properties":{
                    "types":{"type":"array","minItems":1,"maxItems":8,"uniqueItems":true,"items":{"$ref":"#/$defs/entity_type"}},
                    "max_entities":{"type":"integer","minimum":1,"maximum":MAX_SPANS_PER_ROW},
                    "max_mention_scalars":{"type":"integer","minimum":1,"maximum":65536}}},
            "splits":{"type":"object","additionalProperties":false,"required":["development","calibration","locked_test"],
                "properties":{"development":ids,"calibration":ids,"locked_test":ids}},
            "suite":{"type":"object","additionalProperties":false,
                "required":["schema_version","task","dataset_id","dataset_version","named_population","options","expected_input_sha256","splits"],
                "properties":{"schema_version":{"const":1},"task":{"const":"named_entity_recognition"},
                    "dataset_id":name,"dataset_version":name,"named_population":{"type":"string","minLength":1,"maxLength":1024},
                    "options":{"$ref":"#/$defs/options"},"expected_input_sha256":{"type":"string","pattern":"^[0-9a-f]{64}$"},
                    "splits":{"$ref":"#/$defs/splits"}}},
            "gold_span":{"type":"object","additionalProperties":false,
                "required":["entity_type","byte_start","byte_end"],"properties":{
                    "entity_type":{"$ref":"#/$defs/entity_type"},"byte_start":{"type":"integer","minimum":0},
                    "byte_end":{"type":"integer","minimum":1}},
                "description":"Nonempty half-open source byte interval; both offsets must be UTF-8 scalar boundaries. Scalar coordinates are derived internally."},
            "case":{"type":"object","additionalProperties":false,
                "required":["schema_version","id","split","text","language","spans"],"properties":{
                    "schema_version":{"const":1},"id":name,"split":{"enum":["development","calibration","locked_test"]},
                    "text":{"type":"string","minLength":1},"language":{"type":"string","minLength":1,"maxLength":63,
                        "pattern":"^[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*$"},
                    "spans":{"type":"array","maxItems":MAX_SPANS_PER_ROW,"items":{"$ref":"#/$defs/gold_span"}}}},
            "qualification_gates":{"type":"object","additionalProperties":false,
                "required":["schema_version","max_exact_f1_decrease","max_type_relaxed_f1_decrease",
                    "max_exact_recall_decrease","max_document_accuracy_decrease"],
                "properties":{"schema_version":{"const":1},"max_exact_f1_decrease":decrease,
                    "max_type_relaxed_f1_decrease":decrease,"max_exact_recall_decrease":decrease,
                    "max_document_accuracy_decrease":decrease}},
            "report_envelope":{"type":"object","additionalProperties":false,
                "required":["schema_version","protocol","operation","evidence","activation_authority","report"],
                "properties":{"schema_version":{"const":1},"protocol":{"const":NER_PROTOCOL},"operation":{"const":operation},
                    "evidence":{"const":"native_model_replay"},"activation_authority":{"const":false},"report":{"type":"object"}}}
        }}
    })
}
