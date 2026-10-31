# Native INT4 classification Assay

`fnlp eval native-int4` executes the native binary, exclusive classifier over a
frozen labeled text population. `fnlp calibrate native-int4` fits a persistable,
authenticated reliability transform using only its calibration split.
`fnlp qualify native-int4` executes an explicitly pinned candidate and baseline
over the same population, optionally applies independent calibration artifacts,
then applies declared classification-quality and reliability gates. All commands require an existing
INT4 snapshot and an independently retained model-content SHA-256. They do not
select an active model, convert weights, download a model, or activate artifacts.

These commands need the `asupersync-runtime` feature and the protected local file
profile on Linux x86_64/aarch64. Each operation supports `native-int4 --schema`
to describe its data contracts without a model, key, file read, or runtime. The
original `eval/calibrate/qualify scored` commands keep their supplied-observation
contracts and cannot supply artifacts to the native calibration route.

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
and bare private input digests are omitted from ordinary evaluation reports.
Calibration artifacts intentionally contain fitted numeric parameters; isotonic
knots include observed calibration score boundaries. Treat those artifacts as
sensitive and retain protected file output.

## Fit, save and explicitly apply native calibration

Fitting requires at least `--minimum-samples` rows in **both** the calibration
and locked-test splits. The seven-row illustrative suite above has only one
calibration row and therefore produces a typed `insufficient_data` fit with no
usable artifact. The following commands assume a larger, independently labeled
`calibration-suite.json` and `calibration-cases.ndjson`, with their exact complete
byte pin and disjoint memberships already frozen.

```sh
fnlp calibrate native-int4 \
  --snapshot /models/candidate.int4 --expected-model-sha256 "$CANDIDATE_PIN" \
  --group-width 128 --memory-mib 16384 --kernel scalar \
  --suite /private/calibration-suite.json --cases /private/calibration-cases.ndjson \
  --key-file /private/assay.key --key-id owned-key --namespace owned-reviews-v1 \
  --method isotonic --valid-from 2026-01-01 --valid-through 2026-12-31 \
  --shift-policy conservative-abstain \
  --max-rows 1000 --max-input-bytes 1048576 --max-suite-bytes 262144 \
  --max-output-bytes 1048576 --output /private/native-calibration.json
```

Isotonic is the default and accepts native weights at zero or one. Explicit
`--method temperature` performs deterministic temperature fitting and refuses
endpoint calibration weights; it never introduces an implicit epsilon clamp.
Both methods fit only calibration rows. The held-out reliability measurement is
computed after fitting without feeding locked-test scores or labels into the
fit. The original native label decisions, abstentions and classification metrics
remain unchanged.

Retain the SHA-256 of the **complete output file bytes**, including the outer
native envelope and trailing LF, as `CALIBRATION_PIN`. Application authenticates
the full report, artifact and parameter seals using the same key and namespace:

```sh
fnlp eval native-int4 \
  --snapshot /models/candidate.int4 --expected-model-sha256 "$CANDIDATE_PIN" \
  --group-width 128 --memory-mib 16384 --kernel scalar \
  --suite /private/calibration-suite.json --cases /private/calibration-cases.ndjson \
  --key-file /private/assay.key --key-id owned-key --namespace owned-reviews-v1 \
  --calibration /private/native-calibration.json --expected-calibration-sha256 "$CALIBRATION_PIN" \
  --as-of 2026-10-31 --shift-assessment in-distribution \
  --max-rows 1000 --max-input-bytes 1048576 --max-suite-bytes 262144 \
  --max-output-bytes 1048576 --output /private/calibrated-evaluation.json
```

The artifact scope is `exact_frozen_native_classification_replay`. Reapplication
reruns the same frozen source population and verifies both complete actual
execution and observation collections. Changing the input, split membership,
labels, policy, task budget, model or native execution binding refuses. This
artifact does not authorize calibrated claims on a new production population.
It also never converts a supplied-observation artifact into native evidence.

Date and shift are explicit caller declarations. Detected shift requires
`--shift-assessment detected --shift-indicator population-change`; an
in-distribution declaration rejects a shift indicator. Indicators are bounded,
reject control characters/line separators, and appear in the sealed disclosure,
so use nonsecret values. An expired, not-yet-valid or shifted artifact follows
its declared `raw-scores-uncalibrated` or `conservative-abstain` policy. Tampering,
a wrong key/namespace, or a changed replay binding is always a hard error and
cannot use that fallback.

Native fit/apply reports have a separately sealed `report.body` and
`report.seal`. A fit exposes `report.body.outcome.status = fitted` or
`insufficient_data`. An application exposes `evaluated` with an `evaluation`
scorecard, or `abstained` with no invented measurement. Its
`report.body.application.status` records `calibrated`, `uncalibrated`, or
`abstained` and the reason. A calibrated scorecard transforms reliability
probabilities only; it does not claim that its transformed score selected the
observed native label.

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

To apply calibration during qualification, add the candidate's `--calibration`,
`--expected-calibration-sha256`, `--as-of` and `--shift-assessment` arguments.
The baseline has independent `--baseline-calibration`,
`--expected-baseline-calibration-sha256`, `--baseline-as-of` and
`--baseline-shift-assessment` arguments, plus `--baseline-shift-indicator` for
detected shift. Each artifact must come from fitting its own model on this exact
frozen suite. Either, both or neither arm may be calibrated; omission selects
raw reliability for that arm without fitting anything implicitly.

With at least one artifact, the outer qualified report has
`report.body.candidate_application` and `baseline_application` disclosures.
`report.body.outcome.status = compared` contains the ordinary `qualification`
report, whose paired Brier differences and reliability gates use each arm's
effective calibrated or declared raw-fallback scores. Actual native label/F1
gates remain based on observed native decisions. If either artifact requires
conservative abstention, the outcome is `abstained` and contains no numeric
qualification decision that could pass. These explicit refusal reports still
exit zero; authentication, binding, IO and resource failures exit nonzero.

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
Fit admission additionally includes calibration scans/sort/PAV storage,
parameters and signed report staging. Apply admission includes each independent
artifact file, strict envelope parsing, authenticated prepared owner and restored
parameters before model loads and through hosted finalization. Each imported
complete artifact file is also bounded by `--max-output-bytes`. Fitting and
application run within the hosted cooperative cancellation/deadline envelope.

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

The separate ignored fixture
`model_present_native_calibration_round_trip_and_independent_arms_preserve_authority`
uses the same explicit model environment and child-process isolation. It fits
isotonic on five calibration rows, saves and reloads the complete protected
report, evaluates and compares both calibrated arms, verifies unchanged native
decisions and exact paired work, refuses conservative expired qualification,
and checks key/namespace/tamper/byte-pin/source-binding failures before opening
a model. Its source adds regression coverage without asserting model quality or
claiming that the fixture has been executed.
