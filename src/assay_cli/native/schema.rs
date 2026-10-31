//! Discovery is model-free and performs no file or key access.
use serde_json::{Value, json};
use super::NATIVE_PROTOCOL;
use crate::assay::native_classify::{EXECUTION_COLLECTION_FORMAT, OBSERVATION_COLLECTION_FORMAT,
    VECTOR_OBSERVATION_COLLECTION_FORMAT, MAX_LABELS};

pub(super) fn document(operation: &str) -> Value {
    let name = json!({"type":"string","minLength":1});
    let ids = json!({"type":"array","minItems":1,"uniqueItems":true,"items":name});
    let number = json!({"type":"number","minimum":0,"maximum":1});
    let decrease = json!({"type":"number","minimum":-1,"maximum":1});
    let tasks = if operation == "calibrate" { vec!["binary_classification"] }
        else { vec!["binary_classification", "multiclass_classification", "multilabel_classification"] };
    let suites = if operation == "calibrate" { json!({"$ref":"#/$defs/binary_suite"}) }
        else { json!({"oneOf":[{"$ref":"#/$defs/binary_suite"},{"$ref":"#/$defs/multiclass_suite"},
            {"$ref":"#/$defs/multilabel_suite"}]}) };
    let cases = if operation == "calibrate" { json!({"$ref":"#/$defs/binary_case"}) }
        else { json!({"oneOf":[{"$ref":"#/$defs/binary_case"},{"$ref":"#/$defs/multiclass_case"},
            {"$ref":"#/$defs/multilabel_case"}]}) };
    json!({
        "schema_version":1,
        "protocol":NATIVE_PROTOCOL,
        "operation":operation,
        "backend":"native-int4",
        "evidence":"native_model_replay",
        "native_model_replay":true,
        "activation_authority":false,
        "supported_tasks":tasks,
        "label_admission":{"option":"--max-labels","default":2,"minimum":1,"maximum":MAX_LABELS,
            "contract":"Quote the complete declared label cap before input/model IO; refuse a parsed suite above it before snapshot loading."},
        "framing":"One complete canonical UTF-8 JSON report followed by LF; --cases input is NDJSON.",
        "model_input":{
            "candidate":"Explicit --snapshot, --expected-model-sha256 and --group-width.",
            "baseline":"Qualification additionally requires --baseline-snapshot and --baseline-model-sha256; --baseline-group-width is independent.",
            "pins":"Model-content pins come from independently retained conversion evidence; snapshot headers are not trust anchors.",
            "resolution":"No active catalog, conversion, network download or fallback is selected by these commands."
        },
        "key":{"file_bytes":32,"argv_secret":false,
            "file_profile":"Protected regular key and owner-controlled output parent; no key or document bytes in diagnostics."},
        "invariants":[
            "Reject duplicate JSON keys, unknown fields, malformed UTF-8 and non-finite numbers.",
            "The suite pins the exact complete case file before any model is loaded.",
            "Every declared ID occurs once in its named split; all splits and exact-text source units are disjoint.",
            "Rows contain labeled text only; probability, unit digest and execution identity are derived internally.",
            "Binary suites retain the declared positive-event score. Multiclass suites use one exclusive head; multilabel suites use independent present/absent heads, never a label-competitive softmax.",
            "Gold uses exactly positive (binary), label (multiclass), or labels (multilabel); it never enters a model prompt. Empty gold sets are valid only for multilabel.",
            "Actual native decisions and policy abstentions drive classification quality. Label-set reliability resamples whole documents and never treats correlated label decisions as independent samples.",
            "Every complete row identity, private input and score collection is bound with the supplied HMAC key.",
            "Qualification executes both models over the same parsed population with one nonrenewable native work and checkpoint budget.",
            "All five native work axes apply to the complete run, including both qualification arms.",
            "Quality and reliability gates are caller declarations; passing never grants artifact activation authority.",
            "No successful output prefix is published after failed loading, replay, finalization or report encoding.",
            "File output is create-new, staged and protected; stdout writes are not rollbackable."
        ],
        "private_bindings":{
            "scope":"Each native scorecard; qualification has independent candidate and baseline scorecards.",
            "execution_collection":{"commitment_field":"bindings.execution_collection",
                "format":EXECUTION_COLLECTION_FORMAT,"row_count_field":"bindings.execution_collection_rows",
                "binds":"Canonical ordered full per-row native execution identities."},
            "observation_collection":{"commitment_field":"bindings.observation_collection",
                "format":OBSERVATION_COLLECTION_FORMAT,"row_count_field":"bindings.observation_collection_rows",
                "label_vector_format":VECTOR_OBSERVATION_COLLECTION_FORMAT,
                "binds":"Canonical row association, complete task-appropriate gold, native score vector, selected labels and abstentions; the report declares its task-specific framing version."},
            "privacy":"Only domain-separated HMAC commitments, framing versions and row counts are exported."
        },
        "metrics":{
            "classification":"Binary retains classification; multiclass/multilabel expose label_set with per-label counts, native abstentions and aggregate accuracy/F1.",
            "label_set_fields":["mode","rows","label_count","decided_rows","abstained_rows","partially_decided_rows",
                "exact_match_accuracy","decided_exact_match_accuracy","hamming_loss","label_decision_coverage",
                "micro_precision","micro_recall","micro_f1","macro_f1","labels"],
            "label_set_abstention":"abstained_rows includes partially decided documents; hamming_loss counts abstained label decisions as errors; per-label recall/F1 counts abstained positives as missed positives.",
            "binary_reliability":"Uncalibrated normalized positive candidate weight; existing positive-event Brier/ECE/acceptance/risk semantics remain unchanged.",
            "multiclass_reliability":"Brier is half the sum of one-vs-rest squared errors per document (range 0..1); ECE uses the top-label weight and correctness. Acceptance requires a native decision and top weight at least acceptance_threshold.",
            "multilabel_reliability":"Brier averages independent per-label squared errors per document; ECE pools label events while bootstrap resamples entire documents. Acceptance requires every head decided and every chosen-side score at least acceptance_threshold: p for present, 1-p for absent. Risk is exact-set mismatch.",
            "labels":"Human gold labels are supplied by the caller and are not independently authenticated by model execution.",
            "insufficient_data":"A successful typed result when the locked test population is below minimum_samples."
        },
        "qualification":{
            "host_scope_evidence":"caller_declared",
            "gate_basis":"point_estimates",
            "successful_outcomes":["passed","rejected","insufficient_data"],
            "decrease_bounds":"Negative accuracy/F1 decrease bounds require improvement; an undefined required F1 fails its gate.",
            "accuracy_gate":"max_accuracy_decrease uses binary accuracy or label_set.exact_match_accuracy, according to the frozen task.",
            "activation_authority":false
        },
        "calibration":{
            "supported_task":"binary_classification",
            "unsupported_tasks":"Multiclass/multilabel fitting or artifact application refuses before either snapshot is opened; raw evaluation and qualification are supported.",
            "scope":crate::assay::native_classify::calibrate::SCOPE,
            "fit_command":"calibrate native-int4",
            "default_method":"isotonic",
            "methods":["isotonic","temperature"],
            "fit_population":"Only calibration rows fit parameters; only locked-test rows determine reported reliability metrics.",
            "temperature":"Exact 0/1 calibration weights refuse; no implicit epsilon clipping.",
            "fit_context":["--valid-from","--valid-through","--shift-policy"],
            "shift_policies":["raw-scores-uncalibrated","conservative-abstain"],
            "application_context":["--calibration","--expected-calibration-sha256","--as-of","--shift-assessment"],
            "baseline_context":["--baseline-calibration","--expected-baseline-calibration-sha256","--baseline-as-of","--baseline-shift-assessment"],
            "shift_indicator":"Detected shift requires its arm's bounded nonsecret indicator; in-distribution rejects one.",
            "authentication":"Pin the exact complete native calibrate envelope before parsing; authenticate report, artifact and parameter seals with the same key and namespace before model loads.",
            "binding":"Exact frozen dataset plus complete actual native execution and observation collections; a different model, source population, task budget or label/policy binding refuses.",
            "score_semantics":"Transforms reliability scores only; observed native selected labels and policy abstentions remain unchanged.",
            "independent_arms":"Either, both or neither qualification arm may supply a calibration artifact. Paired reliability gates compare each arm's effective transformed or raw score.",
            "conservative_fallback":"A conservative calibration abstention produces a typed successful abstained outcome and cannot pass qualification gates.",
            "fit_status_path":"report.body.outcome.status",
            "fit_statuses":["fitted","insufficient_data"],
            "application_statuses":["evaluated","abstained"],
            "qualification_statuses":["compared","abstained"],
            "disclosure_statuses":["raw","calibrated","uncalibrated","abstained"],
            "file_bound":"--max-output-bytes caps each complete imported calibration report as well as the final output.",
            "private_parameters":"Fitted artifacts intentionally contain sensitive numeric parameters, including isotonic score knots; retain protected file output."
        },
        "json_schema":{
            "$schema":"https://json-schema.org/draft/2020-12/schema",
            "$defs":{
                "label":{"type":"object","additionalProperties":false,"required":["id","description"],
                    "properties":{"id":name,"description":{"type":"string","maxLength":4096}}},
                "policy":{"type":"object","additionalProperties":false,
                    "required":["minimum_candidate_weight_ppm","minimum_margin_ppm"],
                    "properties":{"minimum_candidate_weight_ppm":{"type":"integer","minimum":0,"maximum":1000000},
                        "minimum_margin_ppm":{"type":"integer","minimum":0,"maximum":1000000}}},
                "splits":{"type":"object","additionalProperties":false,
                    "required":["development","calibration","locked_test"],
                    "properties":{"development":ids,"calibration":ids,"locked_test":ids}},
                "suite":suites,
                "binary_suite":suite("binary_classification"),
                "multiclass_suite":suite("multiclass_classification"),
                "multilabel_suite":suite("multilabel_classification"),
                "case":cases,
                "binary_case":case("positive"),
                "multiclass_case":case("label"),
                "multilabel_case":case("labels"),
                "reliability_gates":{"type":"object","additionalProperties":false,
                    "required":["schema_version","max_brier_increase","max_ece_increase","max_selective_risk",
                        "min_acceptance_rate","minimum_accepted"],
                    "properties":{"schema_version":{"const":1},"max_brier_increase":decrease,
                        "max_ece_increase":decrease,"max_selective_risk":number,"min_acceptance_rate":number,
                        "minimum_accepted":{"type":"integer","minimum":1}}},
                "qualification_gates":{"type":"object","additionalProperties":false,
                    "required":["schema_version","reliability","max_accuracy_decrease","max_macro_f1_decrease"],
                    "properties":{"schema_version":{"const":1},"reliability":{"$ref":"#/$defs/reliability_gates"},
                        "max_accuracy_decrease":decrease,"max_macro_f1_decrease":decrease}},
                "report_envelope":{"type":"object","additionalProperties":false,
                    "required":["schema_version","protocol","operation","evidence","activation_authority","report"],
                    "properties":{"schema_version":{"const":1},"protocol":{"const":NATIVE_PROTOCOL},
                        "operation":{"const":operation},"evidence":{"const":"native_model_replay"},
                        "activation_authority":{"const":false},"report":{"type":"object"}}},
                "calibration_envelope":{"type":"object","additionalProperties":false,
                    "required":["schema_version","protocol","operation","evidence","activation_authority","report"],
                    "properties":{"schema_version":{"const":1},"protocol":{"const":NATIVE_PROTOCOL},
                        "operation":{"const":"calibrate"},"evidence":{"const":"native_model_replay"},
                        "activation_authority":{"const":false},"report":{"type":"object","additionalProperties":false,
                            "required":["body","seal"],"properties":{"body":{"type":"object"},"seal":{"type":"object"}}}}}
            }
        }
    })
}

fn suite(task: &str) -> Value {
    let name = json!({"type":"string","minLength":1});
    let mut required = vec!["schema_version", "task", "dataset_id", "dataset_version",
        "named_population", "expected_input_sha256", "splits"];
    let mut properties = json!({"schema_version":{"const":1},"task":{"const":task},
        "dataset_id":name,"dataset_version":name,"named_population":name,
        "policy":{"$ref":"#/$defs/policy"},
        "expected_input_sha256":{"type":"string","pattern":"^[0-9a-f]{64}$"},
        "splits":{"$ref":"#/$defs/splits"}});
    if task == "binary_classification" {
        required.extend(["positive_label", "negative_label"]);
        properties["positive_label"] = json!({"$ref":"#/$defs/label"});
        properties["negative_label"] = json!({"$ref":"#/$defs/label"});
    } else {
        required.push("labels");
        properties["labels"] = json!({"type":"array","minItems":if task == "multiclass_classification" { 2 } else { 1 },
            "maxItems":MAX_LABELS,"items":{"$ref":"#/$defs/label"},
            "description":"Exact unique label IDs; labels sort by ID while descriptions retain their exact bytes."});
    }
    json!({"type":"object","additionalProperties":false,"required":required,"properties":properties})
}

fn case(gold: &str) -> Value {
    let name = json!({"type":"string","minLength":1,"maxLength":256});
    let mut properties = json!({"schema_version":{"const":1},"id":name,
        "split":{"enum":["development","calibration","locked_test"]},"text":{"type":"string","minLength":1}});
    properties[gold] = match gold {
        "positive" => json!({"type":"boolean"}),
        "label" => name,
        _ => json!({"type":"array","uniqueItems":true,"maxItems":MAX_LABELS,"items":name}),
    };
    json!({"type":"object","additionalProperties":false,
        "required":["schema_version","id","split","text",gold],"properties":properties})
}
