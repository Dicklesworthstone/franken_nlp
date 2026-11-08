# Native faithfulness Assay

`fnlp eval native-faithfulness-int4` runs a pinned INT4 snapshot against a
frozen source/claim corpus with human-supplied relations. `fnlp qualify
native-faithfulness-int4` compares explicit candidate and baseline snapshots
on the same held-out claims and policy. Both execute the real native judge;
the case format does not accept supplied probabilities or predictions.

Discover the exact contracts without opening a model, key or case file:

```sh
fnlp eval native-faithfulness-int4 --schema
fnlp qualify native-faithfulness-int4 --schema
```

Execution requires the `asupersync-runtime` feature and the supported protected
file profile, currently Linux x86-64 or AArch64. Use an existing protected key
file containing exactly 32 secret bytes, a private owner-controlled directory,
and a new output pathname. The key and output parent require the existing
0600-file/0700-directory profile. Keep model-content pins independently from
trusted conversion evidence; a snapshot header is not an independent pin.

## Freeze the corpus and policy

Each NDJSON line has exactly six fields. Relations are `entailed`,
`contradicted`, or `unsupported`; unsupported is an ordinary source-relative
label, not an execution error or abstention.

The following complete illustrative `faithfulness-cases.ndjson` contains
eight rows, including six test claims about five distinct test sources. Save
the lines exactly as shown, each ending in LF, including the last line:

```jsonl
{"schema_version":1,"id":"dev-01","split":"development","source":"The archive opens at 08:00.","claim":"The archive opens at 08:00.","relation":"entailed"}
{"schema_version":1,"id":"cal-01","split":"calibration","source":"Invoice 42 is unpaid.","claim":"Invoice 42 has been paid.","relation":"contradicted"}
{"schema_version":1,"id":"test-01a","split":"locked_test","source":"Laurel depot closes at 17:00.","claim":"Laurel depot closes at 17:00.","relation":"entailed"}
{"schema_version":1,"id":"test-01b","split":"locked_test","source":"Laurel depot closes at 17:00.","claim":"Laurel depot closes at 18:00.","relation":"contradicted"}
{"schema_version":1,"id":"test-02","split":"locked_test","source":"The receipt lists exactly two tickets.","claim":"The receipt lists three tickets.","relation":"contradicted"}
{"schema_version":1,"id":"test-03","split":"locked_test","source":"Bo stores red cups.","claim":"Bo stores the cups in a drawer.","relation":"unsupported"}
{"schema_version":1,"id":"test-04","split":"locked_test","source":"Uma returned the book on Monday.","claim":"Uma returned the book on Monday.","relation":"entailed"}
{"schema_version":1,"id":"test-05","split":"locked_test","source":"The train departs at 09:00.","claim":"The train departs at 09:15.","relation":"contradicted"}
```

This `faithfulness-suite.json` pins those exact case bytes:

```json
{
  "schema_version": 1,
  "task": "faithfulness",
  "dataset_id": "source-claims-example",
  "dataset_version": "1",
  "named_population": "Illustrative source-relative claim labels",
  "policy": {
    "minimum_candidate_weight_ppm": 500000,
    "minimum_margin_milli": 100,
    "evidence_window_bytes": 256,
    "max_evidence_windows": 31,
    "max_evidence_spans": 31
  },
  "expected_input_sha256": "65a131967ec7019b15baebb0c04d5796888705266f878293c327b90bbc6ca74e",
  "splits": {
    "development": ["dev-01"],
    "calibration": ["cal-01"],
    "locked_test": ["test-01a", "test-01b", "test-02", "test-03", "test-04", "test-05"]
  }
}
```

For your own corpus, freeze every ID and recompute the ordinary SHA-256 of the
complete NDJSON bytes, including line endings. Every ID must occur exactly
once in its declared split, and all three splits must be nonempty. The same
exact source cannot appear in different splits. Distinct claims about one
source are allowed within a split; duplicate source/claim pairs are refused.
Malformed UTF-8, duplicate JSON keys, unknown fields and blank rows fail.

All splits are validated before either model opens; only `locked_test` claims
execute. Gold labels, IDs and split names do not enter model prompts. The
frozen policy controls uncalibrated candidate-weight and score-margin
thresholds, complete UTF-8 evidence windows, and retained evidence capacity.
Qualification applies that same policy to both models.

## Evaluate one model

Set `ASSAY_MEMORY_MIB` to the modeled allowance for the complete quoted
operation on your host. These commands use bounded small-corpus limits; enlarge
them explicitly for a larger frozen suite.

```sh
fnlp eval native-faithfulness-int4 \
  --snapshot candidate.int4 \
  --expected-model-sha256 "$CANDIDATE_MODEL_SHA256" --group-width 128 \
  --suite faithfulness-suite.json --cases faithfulness-cases.ndjson \
  --key-file assay.key --key-id local-faithfulness-key --namespace faithfulness-v1 \
  --memory-mib "$ASSAY_MEMORY_MIB" \
  --max-suite-bytes 65536 --max-input-bytes 1048576 --max-line-bytes 65536 --max-rows 100 \
  --context-tokens 2048 --max-heads 32 \
  --minimum-samples 5 --bootstrap-resamples 200 --bootstrap-seed 0 \
  --timeout-seconds 3600 --output faithfulness-evaluation.json
```

`--cases -` reads bounded stdin. Output omission or `--output -` writes one
complete canonical JSON report followed by LF. The protocol is
`fnlp.assay.native-faithfulness-int4`, version 1. File output is protected,
staged and create-new. Writer failure or a late timeout can leave visible
bytes; require successful completion and a complete report envelope.

Each judgment scores exactly C, E or U plus EOS using full-vocabulary
denominators. It reads the complete source and every evidence window; a
one-window source reuses its complete-source head. Source, claim and trusted
scaffold must fit the native context. Caps refuse oversized work without
truncating a source, dropping contrary windows or publishing partial metrics.

## Read the scorecard

| Quantity | Interpretation |
| --- | --- |
| Accuracy | Correct final decided relations divided by every held-out claim; abstentions earn no credit. |
| Per-relation precision, recall, F1 | Confusion counts for all three labels; abstained positives count as missed recall. |
| Macro F1 | Mean across all three relations; undefined if any relation has neither support nor predictions. |
| Brier | Mean half-sum of squared one-hot errors from the full-source candidate weights; range 0–1, uncalibrated. |
| Coverage | Fraction of claims with a final decided relation. |
| Selective risk | Incorrect decided claims divided by decided claims; undefined when all claims abstain. |
| Abstentions | Separate counts for ambiguous distribution, insufficient evidence and conflicting evidence. |

Point estimates count claims. Bootstrap draws count **distinct exact-source
clusters**, retaining every claim about a sampled source together. Paired
models use the same source draws. `--minimum-samples` counts sources, defaults
to 20, and cannot be below 5. The example has five sources, not six; adding
more claims about one source cannot satisfy a larger minimum. Too few sources
produce a typed `insufficient_data` report. An undefined ratio in any bootstrap
replicate makes that metric's interval absent rather than inventing a value.

## Compare candidate and baseline

Freeze explicit `faithfulness-gates.json` before inspecting held-out results:

```json
{
  "schema_version": 1,
  "max_accuracy_decrease": 0.02,
  "max_macro_f1_decrease": 0.02,
  "max_brier_increase": 0.01,
  "max_coverage_decrease": 0.02
}
```

```sh
fnlp qualify native-faithfulness-int4 \
  --snapshot candidate.int4 --expected-model-sha256 "$CANDIDATE_MODEL_SHA256" \
  --group-width 128 \
  --baseline-snapshot baseline.int4 --baseline-model-sha256 "$BASELINE_MODEL_SHA256" \
  --baseline-group-width 128 \
  --suite faithfulness-suite.json --cases faithfulness-cases.ndjson \
  --gates faithfulness-gates.json --host-scope declared-local-cpu-v1 \
  --key-file assay.key --key-id local-faithfulness-key --namespace faithfulness-v1 \
  --memory-mib "$ASSAY_PAIRED_MEMORY_MIB" \
  --max-suite-bytes 65536 --max-input-bytes 1048576 --max-line-bytes 65536 --max-rows 100 \
  --context-tokens 2048 --max-heads 32 \
  --minimum-samples 5 --bootstrap-resamples 200 --bootstrap-seed 0 \
  --timeout-seconds 3600 --output faithfulness-qualification.json
```

The paired allowance must cover both resident models. There is one parsed
population and no separate baseline cases or policy overrides. Gates use
point estimates: accuracy, macro-F1 and coverage decreases are baseline minus
candidate; Brier increase is candidate minus baseline. Negative limits require
improvement. Undefined required metrics fail. Reported paired changes and
their intervals consistently use candidate minus baseline.

Inspect the report's `decision.status`: `passed`, `rejected` and
`insufficient_data` are all successfully completed comparisons. Exit zero
alone does not mean the candidate passed the gates. Host scope is explicitly
caller-declared, and a passed report grants no activation or release authority.

## Memory, work and privacy

Before IO, the CLI prices both model residents where applicable, the larger
sequential loading/execution envelope, complete prepared heads, private
records and metrics, output staging and cleanup. `--memory-mib` is a process
ledger ceiling, not an RSS measurement. The preparation reserve defaults to
512 MiB and cannot be lowered. Input/row/output caps affect the quote even
when the actual corpus is smaller.

The following ceilings are shared across all claims, heads and both models:

| Option | Default |
| --- | ---: |
| `--max-forward-positions` | 131072 |
| `--max-projected-logits` | 1000000000 |
| `--max-attention-pairs` | 100000000000 |
| `--max-dot-products` | 10000000000000 |
| `--max-multiply-accumulates` | 1000000000000000 |

`--max-result-bytes` bounds each complete native judgment (default 1 MiB);
`--max-output-bytes` bounds the entire final report including its envelope
and LF (default 4 MiB). `--max-heads` is 1–32. The suite permits at most 31
evidence windows plus the complete-source head. `--max-bootstrap-draws`
defaults to 10000000 and charges both paired arms. One original
`--timeout-seconds` allowance covers loading, replay, metrics and publication;
neither another row nor the baseline renews it.

Reports expose aggregate metrics and nonsecret dataset/population/host
metadata. Private sources, claims, IDs, exact gold rows, raw scores and raw
content hashes appear only through domain-separated keyed commitments. The
execution collection binds each actual prepared identity to its exact
source/claim/policy/budget; the observation collection binds every native
head, quote, abstention and physical work record. Evidence quotes are checked
against source bytes internally and are not emitted in the scorecard.

These reports measure model agreement with caller-supplied human labels.
Candidate weights are uncalibrated; source-membership checks and correlated
model judgments do not establish factual truth or entailment certificates.
Every report retains `activation_authority=false` and audit-only receipt scope.

## Evidence status

**UNRUN:** Rust builds/tests, model-backed replay, task-quality claims and
target performance for this change. Source regression fixtures cover split
isolation, correlated claims, request/result tampering, abstention accounting,
paired uncertainty, resource limits, cancellation and privacy. Those fixtures
do not qualify a model. Execution evidence belongs to the repository's
controller-selected DSR checkpoint; this guide makes no release claim.
