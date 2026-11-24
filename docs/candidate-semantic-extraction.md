# INT8 extraction with semantic field review

`fnlp candidate extract-semantic` runs constrained extraction and then reviews explicitly configured scalar claims against the complete original source, using one resident INT8 model and one owned invocation. The result retains the original extraction unchanged and adds per-field diagnostics.

This current-candidate route requires `asupersync-runtime` and an explicitly selected local INT8 artifact. The second reader is the same model: its output is correlated, experimental, and uncalibrated. It does not repair fields, decide acceptance, or certify correctness.

## Complete example

The checked-in [schema](examples/int8-semantic/schema.json), [options](examples/int8-semantic/options.json), and [source](examples/int8-semantic/source.txt) demonstrate a revenue field and an exact source quote. The claim rule says what the integer means; the system does not infer a field's meaning from its name.

```sh
fnlp candidate extract-semantic docs/examples/int8-semantic/source.txt \
  --model /models/candidate-int8.fnlpq \
  --memory-mib 16384 \
  --context-tokens 2048 \
  --max-new-tokens 128 \
  --max-input-bytes 512 \
  --max-result-bytes 16384 \
  --schema docs/examples/int8-semantic/schema.json \
  --source-membership \
  --semantic-options docs/examples/int8-semantic/options.json
```

These are input and configuration examples, with no asserted model output or quality result. The options explicitly opt into the experimental reader and provide a revision, typed claim paths, judge thresholds, evidence coverage, both-stage budgets, semantic limits, and all five complete-request model-work ceilings. Options and schema must be bounded local files; the source may instead come from stdin.

The example's claim rule renders `prefix + exact JSON scalar + suffix`. For a revenue value of `1250000`, that means `Acme reported revenue of 1250000 US dollars for 2025.` Strings remain quoted JSON scalars, and decimal values are not converted through binary floating point. A `property` path step identifies a literal schema property; `each_item` applies a rule to the scalar leaves of an array. Every configured path and scalar kind must match the schema before model loading. Duplicate rules, unknown fields, duplicate JSON keys, and rules on verbatim fields are refused.

## Reading the result

The candidate response's `output.extraction.result.output.json` is the original exact JSON string. `output.semantic.fields` contains JSON pointers and one of:

- `entailed`: the configured claim receives an entailment result under the explicit judge policy.
- `contradicted`: the judge finds contrary source evidence under that policy.
- `unsupported`: the judge does not find support for the claim under that policy.
- `not_checked`: the field was not given an accepted semantic assessment; the associated reason explains why.

The example's `quote` field uses `x-fnlp-source: "verbatim"`. Its independent source-membership validation remains part of extraction, and semantic review reports `verbatim_membership_only`. Null values, fields without claim rules, empty containers, and judge abstentions also retain explicit `not_checked` reasons. Missing optional fields are not invented. A failed or cancelled stage returns no successful combined result; it is never translated into `unsupported`.

`same_model_correlated` and `experimental_uncalibrated_diagnostic_only` remain visible in the receipt. Evidence quotes and original-source coordinates allow inspection, but source membership alone does not prove entailment.

## Bounds and execution

The options file is read first under caller preparation admission. The CLI then prices the declared maximum source, retained field plans and source copies, native context, intermediate results, and combined output before reading schema, source, or model metadata. It rechecks the quote against the consumed plan before weights are loaded. The source and every dynamic judge must fit the explicit context and evidence policy; no tail is silently dropped.

The example uses a 2,048-position context. Its judge `max_kv_bytes` is `369098752`, the complete resident bf16 KV allocation for that context (`2048 * 180224`). Change this field with `--context-tokens`; both stages must admit the same reused native context. `--max-result-bytes` bounds the original extraction, while `limits.max_output_bytes` in the options bounds the combined semantic result. The extraction bound must also fit `limits.max_input_json_bytes`.

The five native ceilings cover forward positions, projected logits, attention pairs, projection dot products, and projection multiply-accumulates. The maximum extraction reservation remains charged even when extraction stops early. Actual values determine the review prompts, so every dynamic judge is compiled and preflighted after extraction and before the first review forward. All judges run within the remaining complete-request allowance.

This initial CLI route uses serial execution, byte-fallback input, and full-vocabulary projections. Its command does not accept compact-input, selected-row, sampling, or worker-schedule flags. The deadline spans preparation, loading, extraction, review, checked serialization, write, and flush; output and preparation charges remain held through delivery. Memory values describe the process resource ledger, not a measured operating-system RSS limit.

## Rust API

Prepare a real `PreparedInt8SemanticExtraction` from a pinned extraction plan, judge planner, and explicit `Int8SemanticOptions`. Use `Int8SemanticHostLimits::requirements` for the complete quote, then call `NlpEngine::execute_int8_semantic_extract`. The host consumes the plan, reuses one strict INT8 engine for both stages, preserves typed cancellation causes, and returns a guarded `HostedOutput<Int8SemanticExtractionRun>`.
