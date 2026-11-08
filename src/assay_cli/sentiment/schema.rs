//! Model-free discovery for the versioned native sentiment Assay protocol.
use serde_json::{Value, json};
use super::SENTIMENT_PROTOCOL;
use crate::assay::native_sentiment::{EXECUTION_COLLECTION_FORMAT, OBSERVATION_COLLECTION_FORMAT};

pub(super) fn document(operation: &str) -> Value {
    let name = json!({"type":"string","minLength":1,"maxLength":256});
    let ids = json!({"type":"array","minItems":1,"uniqueItems":true,"items":name});
    let axis = json!({"enum":["valence","arousal","dominance","approach"]});
    let error_change = json!({"type":"number","minimum":-2,"maximum":2});
    let optional_change = json!({"type":["number","null"],"minimum":-2,"maximum":2});
    json!({
        "schema_version":1,"protocol":SENTIMENT_PROTOCOL,"operation":operation,"backend":"native-sentiment-int4",
        "evidence":"native_model_replay","native_model_replay":true,"activation_authority":false,
        "supported_tasks":["sentiment"],"score_space":"full_vocab_sequence_logprob",
        "framing":"One complete canonical UTF-8 JSON report followed by LF; --cases is strict NDJSON.",
        "model_input":{
            "candidate":"Explicit --snapshot, --expected-model-sha256 and --group-width.",
            "baseline":"Qualification also requires --baseline-snapshot and --baseline-model-sha256; --baseline-group-width is independent.",
            "pins":"Independently trusted model-content pins are required; a snapshot's own header is not its trust anchor.",
            "resolution":"No active-model lookup, conversion, network download, fallback or activation."
        },
        "key":{"file_bytes":32,"argv_secret":false,"file_profile":"Existing protected regular key file and owner-controlled output parent."},
        "execution":{
            "source_encoding":"Pinned byte-preserving untrusted document input and fixed trusted task scaffold.",
            "candidate_language":"Exact byte-token strings -1, -0.5, 0, 0.5, 1, each followed by scored EOS; every denominator uses the full vocabulary.",
            "axes":"One independent head per fixed suite axis; no cross-axis probability simplex, truncation or omitted axis.",
            "max_axes":{"option":"--max-axes","default":4,"minimum":1,"maximum":4},
            "resource_contract":"Quote both resident models, one reusable native engine, all prepared heads, private observations/rank scratch and report staging before IO. One original deadline and five-axis native work ledger cover all documents and both arms."
        },
        "gold_units":{
            "wire":"Each gold value_milli is an exact integer from -1000 through 1000.",
            "coordinate":"value_milli / 1000 gives sentiment-v1's normalized -1..1 coordinate, NOT -100..100.",
            "anchors":{"valence":["negative","positive"],"arousal":["calm","activated"],
                "dominance":["powerless","in_control"],"approach":["withdrawal","approach"]},
            "between_bins":"Human ratings need not coincide with model bins and are never rounded.",
            "authority":"Gold is caller-supplied human rating, not model agreement or evidence about hidden mental states."
        },
        "invariants":[
            "Reject duplicate JSON keys, unknown fields, malformed UTF-8, blank lines and invalid scalar values.",
            "Pin complete case bytes and validate every row and split before opening either model snapshot.",
            "All three splits are nonempty and ID-disjoint; every declared ID appears exactly once.",
            "Exact source text must be globally unique, including across development, calibration and locked_test.",
            "Every case has exactly one human rating for each fixed suite axis, with no omitted or foreign axis.",
            "Only locked_test rows execute; gold, IDs and split names never enter prompts.",
            "Recheck complete native scores, exact candidate bins, EOS, full-vocabulary semantics, model work, moments and policy abstentions.",
            "Qualification uses one parsed population, fixed axes and unchanged task policy for both models.",
            "Failed loading, execution, validation or expired publication emits no complete success report.",
            "Complete results remain memory-owned through final write and flush; file output is protected and create-new."
        ],
        "private_bindings":{
            "execution_collection":{"format":EXECUTION_COLLECTION_FORMAT,"binds":"Ordered complete native identities associated with document, gold, split, ID and exact task budget."},
            "observation_collection":{"format":OBSERVATION_COLLECTION_FORMAT,"binds":"Every complete native result including all scores, means, dispersion, entropy, abstentions and work."},
            "privacy":"Private sources, row IDs, individual gold and raw content digests leave only domain-separated HMAC commitments. Public metadata is declared dataset/population/host scope and fixed recipe information."
        },
        "metrics":{
            "unit":"One complete unique document; axes do not increase the independent sample count.",
            "mae":"Mean absolute error between every native distribution mean and human value_milli/1000, including abstained documents; range 0..2.",
            "rmse":"Root mean squared error on the same all-document coordinate; range 0..2.",
            "spearman":"Pearson correlation of average ranks with exact tie handling; range -1..1. Constant prediction/gold vectors produce null with a typed reason. Point estimate only.",
            "estimated_coverage":"Fraction of documents whose native axis policy returns Estimated.",
            "estimated_mae":"MAE only among Estimated documents; null if none. It is reported alongside all-document error and coverage.",
            "uncertainty":"Whole-document percentile bootstrap for MAE, RMSE, coverage and estimated-only MAE. Identical seeded draws are used across every axis and paired arm. A replica with no estimates makes that conditional interval null.",
            "work":"--max-bootstrap-draws charges document draws times actual axes times arms, not a separately renewable allowance per axis.",
            "minimum_samples":{"default":20,"minimum":5,"unit":"distinct locked-test documents"},
            "insufficient_data":"Typed successful result when too few independent held-out documents exist.",
            "scope":"Uncalibrated descriptive model estimates compared with human ratings; no captured-mass, mental-state or correctness-confidence claim."
        },
        "qualification":{
            "host_scope_evidence":"caller_declared","gate_basis":"per_axis_point_estimates",
            "successful_outcomes":["passed","rejected","insufficient_data"],
            "bounds":"Every requested gate must pass separately on every axis. Error increases are candidate minus baseline; rank and coverage decreases are baseline minus candidate. Negative limits require improvement.",
            "optional_gates":"Omit or null optional rank/estimated-MAE gates to leave those metrics out. If requested, an undefined metric fails that gate.",
            "paired_changes":"Every reported change and paired interval is candidate minus baseline.",
            "activation_authority":false
        },
        "json_schema":{"$schema":"https://json-schema.org/draft/2020-12/schema","$defs":{
            "policy":{"type":"object","additionalProperties":false,
                "required":["minimum_peak_weight_ppm","maximum_normalized_entropy_ppm"],
                "properties":{"minimum_peak_weight_ppm":{"type":"integer","minimum":0,"maximum":1000000},
                    "maximum_normalized_entropy_ppm":{"type":"integer","minimum":0,"maximum":1000000}}},
            "splits":{"type":"object","additionalProperties":false,"required":["development","calibration","locked_test"],
                "properties":{"development":ids,"calibration":ids,"locked_test":ids}},
            "suite":{"type":"object","additionalProperties":false,
                "required":["schema_version","task","dataset_id","dataset_version","named_population","axes","policy","expected_input_sha256","splits"],
                "properties":{"schema_version":{"const":1},"task":{"const":"sentiment"},"dataset_id":name,"dataset_version":name,
                    "named_population":{"type":"string","minLength":1,"maxLength":1024},
                    "axes":{"type":"array","minItems":1,"maxItems":4,"uniqueItems":true,"items":axis},
                    "policy":{"$ref":"#/$defs/policy"},"expected_input_sha256":{"type":"string","pattern":"^[0-9a-f]{64}$"},
                    "splits":{"$ref":"#/$defs/splits"}}},
            "gold":{"type":"object","additionalProperties":false,"required":["axis","value_milli"],
                "properties":{"axis":axis,"value_milli":{"type":"integer","minimum":-1000,"maximum":1000}}},
            "case":{"type":"object","additionalProperties":false,"required":["schema_version","id","split","text","gold"],
                "properties":{"schema_version":{"const":1},"id":name,"split":{"enum":["development","calibration","locked_test"]},
                    "text":{"type":"string","minLength":1},"gold":{"type":"array","minItems":1,"maxItems":4,"items":{"$ref":"#/$defs/gold"}}}},
            "qualification_gates":{"type":"object","additionalProperties":false,
                "required":["schema_version","max_mae_increase","max_rmse_increase","max_coverage_decrease"],
                "properties":{"schema_version":{"const":1},"max_mae_increase":error_change,"max_rmse_increase":error_change,
                    "max_coverage_decrease":{"type":"number","minimum":-1,"maximum":1},
                    "max_spearman_decrease":optional_change,"max_estimated_mae_increase":optional_change}},
            "report_envelope":{"type":"object","additionalProperties":false,
                "required":["schema_version","protocol","operation","evidence","activation_authority","report"],
                "properties":{"schema_version":{"const":1},"protocol":{"const":SENTIMENT_PROTOCOL},"operation":{"const":operation},
                    "evidence":{"const":"native_model_replay"},"activation_authority":{"const":false},"report":{"type":"object"}}}
        }}
    })
}
