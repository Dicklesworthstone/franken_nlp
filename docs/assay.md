# Local Assay for supplied binary-event observations

`fnlp eval scored` evaluates probabilities and human labels supplied by the
caller. `fnlp calibrate scored` fits a temperature or isotonic transform on a
separate calibration split, measures it on locked-test rows, and writes a
reloadable, keyed report. These commands do not load a model. Their evidence
is explicitly `provided_observations`, and every report has
`activation_authority: false`.

`fnlp qualify scored` compares supplied candidate and baseline observations over
the same frozen units and human labels, under caller-declared gates and host
scope. Its local comparison result also has no native-execution or activation
authority. It does not load either model or claim to have observed the named host.

This is the supplied-observation backend of user-owned Assay. Native
candidate-versus-active model replay and qualification-gated model activation
remain separate functionality. A local report makes no project-wide quality,
release, population-coverage, or model-execution claim.

## Inputs and score meaning

A suite has one named **positive binary event** and one negative label. Each
observation supplies `probability = p(positive)` and a human `positive` label.
ECE and Brier describe that positive-event score. An observation is accepted
when `p(positive)` meets the declared threshold; selective risk measures the
fraction of accepted positive-event predictions whose human label is false.
This threshold is not `max(p, 1-p)` classification confidence. Arbitrary logits,
multiclass maxima, and legal-token mass are not interchangeable with this
declared probability.

Use two files:

- A strict JSON **suite manifest**, with schema version, full canonical
  `ExecutionIdentity`, dataset ID/version, population, labels, exact observation
  byte digest, and development/calibration/locked-test ID lists.
- **Observation NDJSON**, with one schema-versioned row per ID. Each row contains
  its split, probability, human label, and `unit_sha256` identifying the original
  evaluation unit. Use the digest of the same underlying input unit across
  different runs, so duplicate inputs cannot evade leakage checks by changing IDs.

The parser rejects duplicate JSON keys, unknown fields, malformed probabilities,
duplicate IDs, duplicate unit digests, missing or extra split members, split
overlap, and any mismatch between the exact observation bytes and the manifest
pin. Development, calibration, and locked-test units must be disjoint. Rows are
sorted canonically before floating-point aggregation; changing their input order
requires updating the exact byte pin but does not change their measurements.

The observation format is:

```json
{"schema_version":1,"id":"ticket-001","split":"locked_test","probability":0.83,"positive":true,"unit_sha256":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"}
```

The digest above is illustrative. Actual digests must identify the declared bytes.
Inspect the exact manifest, identity and observation fields without reading any
files or secrets:

```sh
fnlp eval scored --schema
fnlp calibrate scored --schema
fnlp qualify scored --schema
```

## Privacy and local storage

Supply an existing **32-byte secret key file**, a nonsecret `--key-id`, and a
nonsecret `--namespace`. There is no secret-value command-line option. Keep the
same key and namespace when applying a fitted report or comparing observations
from the same suite. The key and its bytes never appear in output.

Input and split identities are exported as domain-separated HMAC commitments.
Private row IDs, unit SHA-256 values, raw input hashes, and private execution-key
projections are not exported. Named dataset/population/label metadata and aggregate
statistics remain visible; choose those names intentionally.
Each dataset binding's `metadata` also names the recipe, task specification,
trusted prompt-template hash, numerics profile and thinking mode. The template
hash is distinct from the private document-specific prompt digest, which remains
inside the keyed execution binding.

The current protected file profile is Linux x86_64/aarch64 with procfs. A key must
be an owned regular file under a private `0700` parent with permissions excluding
group/other access, no symlink path components, and no hard-link aliases. A new
report supplied with `-o` is created as `0600` under a private `0700` parent.
Existing destinations are refused. Unsupported profiles refuse before input IO;
schema discovery remains available everywhere. Same-UID/root attackers and
malicious mounts are outside this file profile.

Without `-o`, the complete report goes to stdout. Computation and bounded
serialization finish before publication. A failed input or computation emits no
success prefix. A writer error can still interrupt delivery to a pipe; it is
reported as a failure, not a complete report.

## Complete synthetic workflow

The following fixture is intentionally synthetic. It exercises transport,
fitting, persistence and application without claiming model evaluation. It
creates a new private directory, so rerunning it refuses an existing directory.
Python is used only to prepare example data, not by the `fnlp` commands.

```python
import hashlib
import json
import os
from pathlib import Path

root = Path("assay-demo")
root.mkdir(mode=0o700)

def save(name, content):
    fd = os.open(root / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as output:
        output.write(content)

def digest(content):
    return hashlib.sha256(content).hexdigest()

def json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()

save("key", os.urandom(32))
rows, splits = [], {}
for split, prefix, count in [
    ("development", "dev", 2),
    ("calibration", "cal", 20),
    ("locked_test", "test", 20),
]:
    splits[split] = []
    for index in range(count):
        item_id = f"{prefix}-{index}"
        splits[split].append(item_id)
        rows.append({
            "schema_version": 1, "id": item_id, "split": split,
            "probability": 0.7 if index % 2 == 0 else 0.3,
            "positive": index % 2 == 0,
            "unit_sha256": digest(f"synthetic-unit:{item_id}".encode()),
        })
observations = b"".join(json_bytes(row) for row in rows)
save("observations.ndjson", observations)

identity = {
    "schema_version": 1,
    "source_revision": "f56ec5a9650268aa098496734743c25ea778bd2d",
    "artifact_format": "synthetic-fixture-v1",
    "quant_recipe": "synthetic-fixture-int8-v1",
    "task_spec": "synthetic-binary-event-v1",
    "grammar_compiler_version": "fixture-grammar-v1",
    "numerics_profile": "strict-quantized-v1",
    "kv_dtype": "bf16", "sampler_version": "greedy-v1",
    "thinking_mode": "disabled", "tool_mode": "none",
    "backend_semantic_version": "fixture-cpu-v1",
    "host_class": None, "compiler_identity": None,
}
for field in [
    "logical_model_digest", "packing_set_digest", "tokenizer_digest",
    "template_digest", "taskir_digest", "prompt_digest", "schema_digest",
    "calibration_digest", "decision_policy_digest",
]:
    identity[field] = digest(f"synthetic:{field}".encode())
manifest = {
    "schema_version": 1, "task": "binary_event",
    "dataset_id": "synthetic-demo", "dataset_version": "1",
    "named_population": "synthetic-fixture-only",
    "positive_label": "yes", "negative_label": "no",
    "execution_identity": identity,
    "expected_input_sha256": digest(observations), "splits": splits,
}
save("suite.json", json_bytes(manifest))

# Paired baseline: exactly the same units, labels and split membership.
baseline_rows = [dict(row, probability=0.6 if row["positive"] else 0.4) for row in rows]
baseline_observations = b"".join(json_bytes(row) for row in baseline_rows)
save("baseline.ndjson", baseline_observations)
baseline_manifest = dict(
    manifest,
    expected_input_sha256=digest(baseline_observations),
    execution_identity=dict(identity, logical_model_digest=digest(b"synthetic:baseline-model")),
)
save("baseline-suite.json", json_bytes(baseline_manifest))
save("gates.json", json_bytes({
    "schema_version": 1,
    "max_brier_increase": 0.0,
    "max_ece_increase": 0.0,
    "max_selective_risk": 0.1,
    "min_acceptance_rate": 0.4,
    "minimum_accepted": 5,
}))
```

Real evaluation uses the complete execution identity from the actual scoring
workflow, real unit digests and independently supplied human labels. The fixture
hashes above are not substitute model provenance.

Evaluate the supplied raw probabilities:

```sh
fnlp eval scored \
  --suite assay-demo/suite.json --observations assay-demo/observations.ndjson \
  --key-file assay-demo/key --key-id demo-key-v1 --namespace demo-suite-v1 \
  --minimum-samples 20 --acceptance-threshold 0.8 \
  --bootstrap-resamples 200 --bootstrap-seed 17 \
  -o assay-demo/raw-evaluation.json
```

Fit exclusively on the calibration split and measure on the locked test:

```sh
fnlp calibrate scored \
  --suite assay-demo/suite.json --observations assay-demo/observations.ndjson \
  --key-file assay-demo/key --key-id demo-key-v1 --namespace demo-suite-v1 \
  --method temperature --valid-from 2026-01-01 --valid-through 2026-12-31 \
  --shift-policy conservative-abstain --bootstrap-seed 17 \
  -o assay-demo/calibration.json
```

Use `--method isotonic` for a deterministic monotonic PAV fit. Temperature fitting
requires open input probabilities (`0 < p < 1`); isotonic fitting supports the
closed interval. Fitted parameters and the original measurement are authenticated
in the saved report. Too-small calibration or locked-test populations return
`insufficient_data` with no fitted artifact. The minimum supported declared sample
size is five; the default is twenty. These thresholds are refusal rules, not a
claim that twenty observations establish population-level reliability.

Record the complete calibration file's SHA-256 at trusted creation time. Supply
that independently retained value when applying the report:

```sh
fnlp eval scored \
  --suite assay-demo/suite.json --observations assay-demo/observations.ndjson \
  --key-file assay-demo/key --key-id demo-key-v1 --namespace demo-suite-v1 \
  --calibration assay-demo/calibration.json \
  --expected-calibration-sha256 "$SAVED_CALIBRATION_SHA256" \
  --as-of 2026-06-01 --shift-assessment in-distribution \
  --bootstrap-seed 17 -o assay-demo/calibrated-evaluation.json
```

The pin covers the entire calibration report, including its terminating newline.
Application requires matching suite, split, execution, key and namespace
commitments. It also requires the observed date and an explicit caller assessment
of shift. Use `--shift-assessment detected --shift-indicator changed-population`
when a named shift condition applies. Expiry or shift follows the saved policy:
`conservative-abstain` emits a successful abstention without calibrated metrics;
`raw-scores-uncalibrated` reports raw measurements labeled uncalibrated. No command
infers exchangeability or distribution stability from supplied scores.

## Paired local qualification

Compare candidate and baseline observation files under the example's explicit
gates:

```sh
fnlp qualify scored \
  --suite assay-demo/suite.json --observations assay-demo/observations.ndjson \
  --baseline-suite assay-demo/baseline-suite.json \
  --baseline-observations assay-demo/baseline.ndjson --gates assay-demo/gates.json \
  --host-scope caller-declared-demo-host \
  --key-file assay-demo/key --key-id demo-key-v1 --namespace demo-suite-v1 \
  --minimum-samples 20 --acceptance-threshold 0.5 --bootstrap-seed 17 \
  -o assay-demo/comparison.json
```

Pairing requires the same dataset/version/population, event labels, split IDs,
unit digests and human labels. A different gold label or underlying unit is a
scope error even if each file separately passes its own digest and split checks.
The execution identities must retain the same task/scoring semantics; explicit
model, quantization, packing, numerics and backend/host/compiler differences are
the subjects being compared. Neither a raw input hash nor a caller-provided host
name proves that a model ran. The receipt records `host_scope_evidence:
"caller_declared"`, `receipt_grade: "audit_only"`, and
`activation_authority: false`.

All gate fields are required. `max_brier_increase` and `max_ece_increase` are
candidate-minus-baseline changes in `[-1, 1]`; a negative ceiling requires an
improvement of at least that magnitude. `max_selective_risk` and
`min_acceptance_rate` are candidate point-estimate bounds in `[0, 1]`.
`minimum_accepted` is an explicit count from one through the admitted row limit.
Together these gates prevent a low reported risk from concealing negligible
acceptance. No acceptance policy is invented when a field is absent.

These are **point-estimate gates** on the supplied locked-test population,
recorded explicitly as `gate_basis: "point_estimates"`.
Candidate and baseline bootstrap intervals remain visible in their evaluation
reports, but are not converted into gate-confidence or population-coverage
claims. The paired summary reports mean Brier change and counts of units where
candidate Brier error is better, worse or equal. Named gate results contain the
observed value, caller limit and verdict. A missing selective-risk estimate does
not pass a risk gate.

`passed`, `rejected` and `insufficient_data` are successful comparison outcomes
(exit 0), distinguished in `report.decision.status`. Insufficient samples produce
no paired metrics. Structural, scope or digest errors fail with no success
output. The receipt seals the declared policy, host scope, both bound evaluations
and the decision under the supplied key; it grants no native replay, release or
model-activation authority.

## Limits, reports and errors

Defaults bound the suite to 1 MiB, observation input to 64 MiB, each line to
64 KiB, row count to 100,000, and the complete report to 4 MiB. These are explicit
payload/work ceilings, not a process RSS guarantee. Bootstrap work is bounded
by both `--max-bootstrap-resamples` (default 1,000) and
`--max-bootstrap-draws` (default 10,000,000). The chosen resample count defaults to
200. Paired qualification counts both arms against the same draw budget; input
byte and row limits apply to each supplied dataset. Increase a ceiling only
within the library's hard maximum; all invalid
row-independent options refuse before files are opened.

Every successful output is a single canonical JSON record followed by LF:

```json
{"schema_version":1,"protocol":"fnlp.assay.scored","operation":"eval","evidence":"provided_observations","activation_authority":false,"report":{"...":"operation-specific report"}}
```

The field order shown is illustrative; the canonical writer sorts keys. This
protocol is separate from frozen robot v3 events. The evaluation report names
its options, keyed dataset/split/execution bindings, locked-test metrics and
seeded percentile-bootstrap intervals. All report sections share the complete
output byte ceiling.

Typed insufficient-data or calibration-policy abstention is a successful result
(exit 0), not a fabricated metric. Invalid arguments use exit 2; malformed input
uses 4; digest/artifact/scope mismatches use 7; resource/output limits use 9.
Publication/write failure remains a failure. Error records on stderr contain
fixed codes and reasons, never private IDs, score rows, secret bytes or paths.

Implementation and regression source are provided; Rust execution, native model
parity, target-host measurements and release qualification still require the
repository's controller-selected DSR validation.
