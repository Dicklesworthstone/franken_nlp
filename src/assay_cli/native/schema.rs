//! Discovery is model-free and performs no file or key access.
use serde_json::{Value, json};
use super::NATIVE_PROTOCOL;
use crate::assay::native_classify::{EXECUTION_COLLECTION_FORMAT, OBSERVATION_COLLECTION_FORMAT};

pub(super) fn document(operation: &str) -> Value {
    let name = json!({"type":"string","minLength":1});
    let ids = json!({"type":"array","minItems":1,"uniqueItems":true,"items":name});
    let number = json!({"type":"number","minimum":0,"maximum":1});
    let decrease = json!({"type":"number","minimum":-1,"maximum":1});
    json!({
        "schema_version":1,
        "protocol":NATIVE_PROTOCOL,
        "operation":operation,
        "backend":"native-int4",
        "evidence":"native_model_replay",
        "native_model_replay":true,
        "activation_authority":false,
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
            "The two suite labels form one exclusive classification task; the declared positive label determines the reported positive candidate weight.",
            "Actual selected labels and policy abstentions drive classification quality; acceptance_threshold independently controls positive-event reliability statistics.",
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
                "binds":"Canonical row association, gold label, actual positive candidate weight, selected label and abstention decision."},
            "privacy":"Only domain-separated HMAC commitments, framing versions and row counts are exported."
        },
        "metrics":{
            "classification":"Actual held-out confusion counts, abstentions, accuracy and per-class/macro F1.",
            "reliability":"Uncalibrated normalized binary candidate weight; Brier/ECE/positive-event acceptance and risk use existing Assay options.",
            "labels":"Human gold labels are supplied by the caller and are not independently authenticated by model execution.",
            "insufficient_data":"A successful typed result when the locked test population is below minimum_samples."
        },
        "qualification":{
            "host_scope_evidence":"caller_declared",
            "gate_basis":"point_estimates",
            "successful_outcomes":["passed","rejected","insufficient_data"],
            "decrease_bounds":"Negative accuracy/F1 decrease bounds require improvement; an undefined required F1 fails its gate.",
            "activation_authority":false
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
                "suite":{"type":"object","additionalProperties":false,
                    "required":["schema_version","task","dataset_id","dataset_version","named_population",
                        "positive_label","negative_label","expected_input_sha256","splits"],
                    "properties":{"schema_version":{"const":1},"task":{"const":"binary_classification"},
                        "dataset_id":name,"dataset_version":name,"named_population":name,
                        "positive_label":{"$ref":"#/$defs/label"},"negative_label":{"$ref":"#/$defs/label"},
                        "policy":{"$ref":"#/$defs/policy"},
                        "expected_input_sha256":{"type":"string","pattern":"^[0-9a-f]{64}$"},
                        "splits":{"$ref":"#/$defs/splits"}}},
                "case":{"type":"object","additionalProperties":false,
                    "required":["schema_version","id","split","text","positive"],
                    "properties":{"schema_version":{"const":1},"id":name,
                        "split":{"enum":["development","calibration","locked_test"]},
                        "text":{"type":"string","minLength":1},"positive":{"type":"boolean"}}},
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
                        "activation_authority":{"const":false},"report":{"type":"object"}}}
            }
        }
    })
}
