# Native INT4 classification Assay

`fnlp eval native-int4` executes the native binary, exclusive classifier over a
frozen labeled text population. `fnlp qualify native-int4` executes an explicitly
pinned candidate and baseline over that same population, then applies declared
classification-quality and reliability gates. Both commands require an existing
INT4 snapshot and an independently retained model-content SHA-256. They do not
select an active model, convert weights, download a model, or activate artifacts.

These commands need the `asupersync-runtime` feature and the protected local file
profile on Linux x86_64/aarch64. `eval native-int4 --schema` and
`qualify native-int4 --schema` describe the data contracts without a model, key,
file read, or runtime. The original `eval/calibrate/qualify scored` commands keep
their supplied-observation contracts; native calibration is not exposed here.

## Freeze a labeled population

The UTF-8 cases file contains one strict JSON object per line. Its only fields
are `schema_version`, `id`, `split`, nonempty `text`, and Boolean `positive`:

```jsonl
{"schema_version":1,"id":"dev-1","split":"development","text":"This device is easy to use.","positive":true}
{"schema_version":1,"id":"cal-1","split":"calibration","text":"The replacement arrived broken.","positive":false}
{"schema_version":1,"id":"test-1","split":"locked_test","text":"I am pleased with this purchase.","positive":true}
{"schema_version":1,"id":"test-2","split":"locked_test","text":"The battery stopped charging.","positive":false}
{"schema_version":1,"id":"test-3","split":"locked_test","text":"Setup was quick and the controls work.","positive":true}
{"schema_version":1,"id":"test-4","split":"locked_test","text":"The screen failed on the first day.","positive":false}
{"schema_version":1,"id":"test-5","split":"locked_test","text":"The product matches the description.","positive":true}
```

Pin the SHA-256 of the **exact complete file bytes**, including line endings, in
the suite manifest. Replace the placeholder below with that lowercase digest:

```json
{
  "schema_version": 1,
  "task": "binary_classification",
  "dataset_id": "owned-reviews",
  "dataset_version": "1",
  "named_population": "Illustrative user-owned product reviews",
  "positive_label": {"id": "positive", "description": "An approving product review."},
  "negative_label": {"id": "negative", "description": "A disapproving product review."},
  "expected_input_sha256": "REPLACE_WITH_64_LOWERCASE_HEX_CHARACTERS",
  "splits": {
    "development": ["dev-1"],
    "calibration": ["cal-1"],
    "locked_test": ["test-1", "test-2", "test-3", "test-4", "test-5"]
  }
}
```

Every split must be nonempty. IDs have unique, exact membership, and identical
source text may not appear twice anywhere in the population. The parser derives
source-unit hashes itself, rejects duplicate JSON keys and unknown fields, and
checks the complete byte pin before loading a model. Rows never accept supplied
probabilities or execution identities. File order can differ from the split
lists; replay uses canonical ID order, and the byte pin still binds the original
file exactly. Keep dataset IDs, population descriptions, key IDs and namespaces
nonsecret: reports intentionally disclose that metadata.

The suite can optionally declare `policy` with
`minimum_candidate_weight_ppm` and `minimum_margin_ppm`, each in `0..=1000000`.
Omission uses zero for both. This is the native classifier's abstention policy.
The separate `--acceptance-threshold` controls positive-event reliability
statistics; it does not replace the classifier's selected label or task policy.

## Evaluate a pinned model

Prepare an existing owner-protected file containing exactly 32 secret key bytes,
and an owner-controlled directory for new reports. The key is read from the file;
there is no argv secret option. Set `CANDIDATE_PIN` from independently retained
conversion evidence, not by trusting an unverified snapshot header.

```sh
fnlp eval native-int4 \
  --snapshot /models/candidate.int4 --expected-model-sha256 "$CANDIDATE_PIN" \
  --group-width 128 --memory-mib 16384 --kernel scalar \
  --suite /private/suite.json --cases /private/cases.ndjson \
  --key-file /private/assay.key --key-id owned-key --namespace owned-reviews-v1 \
  --max-rows 1000 --max-input-bytes 1048576 --max-suite-bytes 262144 \
  --max-output-bytes 1048576 --minimum-samples 5 --bootstrap-resamples 128 \
  --output /private/evaluation.json
```

The seven-row population and minimum of five illustrate the interface, not a
quality qualification. The normal minimum is 20 locked-test rows. Choose the
population, sample requirement, kernel, memory ceiling and work limits for the
intended evaluation. A parsed population below `minimum_samples` returns
`measurement.status = "insufficient_data"` and `classification = null`, with no
invented metric estimates. The command still records the actual completed replay.

The report contains native held-out confusion counts, abstentions, accuracy over
all held-out rows, decided accuracy, and per-class/macro F1. An abstention earns
no accuracy credit. A missing class with no predictions has undefined F1 rather
than an invented perfect score. Reliability metrics use the **uncalibrated
positive candidate weight** normalized over the two complete label sequences;
this is not a calibrated real-world probability. Brier score, ECE, selective risk
and bootstrap intervals use the existing Assay measurement rules.

The native task scores full-vocabulary continuations including scored EOS. The
actual native selected label, including lexical label-ID ties and policy
abstentions, drives classification quality. Every split is replayed; only the
locked-test split contributes to reported evaluation metrics. The report binds
the complete ordered collection of actual per-row execution identities with
HMAC in `bindings.execution_collection`. A separate
`bindings.observation_collection` commits to each canonical row association,
gold label, actual positive candidate weight, selected label and abstention
decision. Its framing version is
`ordered_native_binary_observations_sha256_framed_v1`; both collections expose
their format and row count alongside the HMAC. The suite, input, split membership
and result seal are also bound. Documents, private row IDs, individual scores,
and bare private input digests are omitted.

## Compare candidate and baseline

Native qualification requires a separate strict gate document. These values
illustrate the format; the population owner chooses its acceptance policy:

```json
{
  "schema_version": 1,
  "reliability": {
    "schema_version": 1,
    "max_brier_increase": 0.005,
    "max_ece_increase": 0.01,
    "max_selective_risk": 0.2,
    "min_acceptance_rate": 0.25,
    "minimum_accepted": 5
  },
  "max_accuracy_decrease": 0.01,
  "max_macro_f1_decrease": 0.01
}
```

```sh
fnlp qualify native-int4 \
  --snapshot /models/candidate.int4 --expected-model-sha256 "$CANDIDATE_PIN" \
  --baseline-snapshot /models/baseline.int4 --baseline-model-sha256 "$BASELINE_PIN" \
  --group-width 128 --baseline-group-width 128 --memory-mib 16384 --kernel scalar \
  --suite /private/suite.json --cases /private/cases.ndjson \
  --key-file /private/assay.key --key-id owned-key --namespace owned-reviews-v1 \
  --gates /private/native-gates.json --host-scope declared-workstation \
  --max-rows 1000 --max-input-bytes 1048576 --max-suite-bytes 262144 \
  --max-output-bytes 1048576 --minimum-samples 5 --bootstrap-resamples 128 \
  --output /private/qualification.json
```

The candidate and baseline have independent content pins and group widths. Both
execute the same parsed documents, labels, policy, task budgets and split
membership. Gates use point estimates. Negative decrease bounds require an
improvement, and undefined required metrics fail their gates. The report includes
both complete scorecards, every gate comparison, paired per-row Brier changes,
and combined native work. `passed`, `rejected`, and `insufficient_data` are all
successful report outcomes with exit status zero. The supplied `host_scope` is
explicitly `caller_declared`; execution does not authenticate that declaration.
Every report has `activation_authority = false` and audit-only receipt grade.

## Admission, work and output

Before key, suite, case or model IO, the CLI computes a checked complete memory
requirement. Qualification admits both complete resident models plus the peak of
one sequential native engine. Raw parsing storage, owned cases and retained
scores, compiler preparation, metric/bootstrap scratch, report serialization,
output publication and cleanup claims are additional. The explicit
`--memory-mib` is a process memory-ledger ceiling, not a measured RSS result; an
insufficient declaration refuses. Tight input/report bounds reduce the required
reservation. `--preparation-mib` defaults to 128 and prices compiler/request
headroom separately. `--max-result-bytes` bounds each task result, while
`--max-output-bytes` bounds the complete final report including its envelope/LF.
After parsing, the CLI checks the complete held-out bootstrap workload across
both arms and the actual receipt namespace contract before either snapshot load.

The five work ceilings (`--max-forward-positions`, `--max-projected-logits`,
`--max-attention-pairs`, `--max-dot-products`, `--max-multiply-accumulates`) cover
the **entire replay, including both qualification arms**. A new row or baseline
does not renew them. One physical hosted dispatch shares replay checkpoints and
remaining elapsed time, and finalizes the report under the same owned
reservations. Snapshot loads have their own `--snapshot-checkpoints` ceiling;
all stages consume one cooperative `--timeout-seconds` deadline. Checks between
file operations do not make arbitrary operating-system blocking IO preemptible.

Output uses protocol `fnlp.assay.native-int4` version 1. The outer evidence value
is `native_model_replay`; each native scorecard uses
`native_int4_classification_replay`. A new protected file is staged and published
only after loading, replay, metric finalization and complete report encoding
succeed. Existing outputs are not overwritten. Omit `--output`, or use `-`, for
one complete JSON/LF stdout write; a failing output stream cannot be rolled back.
Typed failures use a sanitized `assay_error` on stderr and a nonzero exit status.

The ignored model fixture
`model_present_native_assay_replays_and_paired_work_are_exact_and_private`
requires `FNLP_INT4_PREFILL_SNAPSHOT`, `FNLP_INT4_PREFILL_MODEL_SHA256`,
`FNLP_INT4_TEST_MEMORY_MIB`, and `FNLP_INT4_TEST_KERNEL`; optional
`FNLP_INT4_TEST_GROUP_WIDTH` defaults to 128. It isolates the process resource
host, evaluates owned text, replays the same pinned model in both arms, checks
exact zero paired changes and doubled work, refuses a paired run priced for only
one arm, and exercises private output, insufficient samples and mismatched pins.
This is regression source, not a claimed model-quality measurement or recorded
execution result.
