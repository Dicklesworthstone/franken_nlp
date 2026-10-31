# Native NER Assay: frozen spans, actual replay, paired qualification

`fnlp eval native-ner-int4` evaluates an independently pinned native INT4
snapshot against a frozen, user-owned named-entity recognition corpus.
`fnlp qualify native-ner-int4` runs candidate and baseline snapshots over the
same locked-test documents and applies explicit span-quality gates. Both
commands emit one canonical JSON report under `fnlp.assay.native-ner-int4` v1.

The evaluator executes the source-constrained NER task and independently checks
every returned mention and occurrence before scoring. It reports exact spans,
type-relaxed spans, document accuracy, source anchoring and paired document
uncertainty. NER's score space is `not_computed`: there are no mention
probabilities, Brier/ECE estimates, acceptance thresholds or NER calibration
artifacts. Passing qualification grants no model activation authority.

Schema discovery requires no model, key, files or runtime host:

```sh
fnlp eval native-ner-int4 --schema
fnlp qualify native-ner-int4 --schema
```

## Freeze a source-span corpus

The suite manifest has exactly these fields:

```json
{
  "schema_version": 1,
  "task": "named_entity_recognition",
  "dataset_id": "owned-customer-documents",
  "dataset_version": "2026-10-31",
  "named_population": "Owner-labeled customer documents in the declared languages",
  "options": {
    "types": ["person", "organization", "location"],
    "max_entities": 64,
    "max_mention_scalars": 256
  },
  "expected_input_sha256": "REPLACE_WITH_SHA256_OF_THE_COMPLETE_CASE_FILE",
  "splits": {
    "development": ["dev-001"],
    "calibration": ["cal-001"],
    "locked_test": ["test-001", "test-002"]
  }
}
```

Replace the illustrative IDs with every ID in your complete case file and pin
its exact bytes with 64 lowercase hexadecimal SHA-256 digits. All three splits
must be nonempty, disjoint and completely represented. IDs and exact source
texts must be unique across the entire corpus. Newline changes alter the input
pin. Duplicate keys, unknown fields, malformed UTF-8, blank case lines, missing
rows and foreign split membership refuse before either snapshot is opened.

Cases are one JSON object per line. For example:

```json
{"schema_version":1,"id":"test-001","split":"locked_test","text":"Zoë met Omar.","language":"en","spans":[{"entity_type":"person","byte_start":0,"byte_end":4},{"entity_type":"person","byte_start":9,"byte_end":13}]}
```

Gold spans use nonempty, half-open UTF-8 **byte** intervals. `Zoë` occupies bytes
`[0,4)` even though it contains three Unicode scalars. The parser checks both
boundaries against the original source and computes scalar offsets itself.
Caller-supplied scalar offsets, mention text, confidence, model predictions,
source hashes and execution identities are not accepted in a gold span.

Each interval has one declared entity type. Duplicate intervals are rejected,
including two different types attached to the same boundaries. Distinct nested
or overlapping spans are supported, and `spans: []` represents an empty gold
set. The type vocabulary is `person`, `organization`, `location`, `date`,
`time`, `money`, `product`, and `event`; the suite selects a unique nonempty
subset. Gold cannot use a type outside that subset.

`language` is owner-declared metadata: one to 63 ASCII letters/digits with
optional nonempty hyphen-separated components, such as `en` or `zh-Hant`.
Tags are preserved and grouped exactly as declared. No language detection or
automatic normalization is performed. Keep dataset IDs, population descriptions,
language tags and host-scope labels nonsecret because they appear in reports.

Only `locked_test` documents execute. Development and calibration rows are
validated and included in private dataset/split bindings, so the held-out
population cannot silently change. This NER route performs no calibration fit.
Only the source text and frozen NER options enter a model request; gold spans,
IDs, language tags and split names do not enter prompts.

The suite's `max_entities` bounds the number of predicted entities and must fit
the declared `--max-spans-per-row`. That CLI cap also bounds each gold list.
`max_mention_scalars` bounds generated mentions, not the human gold annotation
length. Caps are checked before typed gold-row allocation; capacity is never
enforced by truncating gold or occurrence lists.

## Evaluate one explicit model

Use an existing snapshot and retain its model-content digest independently
from trusted conversion evidence. Reading a digest from the same snapshot
header does not establish an independent pin. The key file must already be a
protected regular file containing exactly 32 secret bytes; do not place the key
in command arguments. Select a new report filename in an owner-controlled
directory.

```sh
fnlp eval native-ner-int4 \
  --snapshot candidate.int4 \
  --expected-model-sha256 "$CANDIDATE_MODEL_SHA256" \
  --group-width 128 \
  --suite ner-suite.json --cases ner-cases.ndjson \
  --key-file assay.key --key-id local-ner-key --namespace ner-evaluation-v1 \
  --memory-mib "$ASSAY_MEMORY_MIB" \
  --minimum-samples 20 --bootstrap-resamples 200 --bootstrap-seed 0 \
  --context-tokens 2048 --max-new-tokens 512 \
  --max-spans-per-row 64 --max-occurrences-per-row 16384 \
  --output candidate-ner-scorecard.json
```

`ASSAY_MEMORY_MIB` is an explicit process-memory ledger ceiling for this run.
The command quotes a complete modeled peak before reading the suite, key or
snapshot: raw strict input and typed gold, model loading, resident weights,
native K/V and workspace, tokenizer/vocabulary preparation, source indices,
grammar/mask storage, retained document counts, bootstrap state, encoded output
and cleanup. Qualification includes both resident models and one sequential
native execution workspace. This accounting is not an RSS measurement.

`--cases -` reads bounded NDJSON from stdin. Omitted output or `--output -`
writes the complete report to stdout; protected file output stages a complete
create-new report and preserves its memory claim through physical flush. A
stdout sink can fail after receiving a prefix, so use protected file output
when atomic publication matters.

## Understand the scorecard

The outer `report` is the sealed native scorecard. `validated_rows` counts all
source rows; `executed_rows` counts locked-test documents. A population below
`--minimum-samples` returns `measurement.status: "insufficient_data"` rather
than a measured scorecard. The default minimum is 20 documents and the smallest
permitted minimum is five.

For `measurement.status: "measured"`, `measurement.metrics` contains:

| Field | Meaning |
| --- | --- |
| `exact` | Corpus micro precision/recall/F1 and TP/FP/FN counts; a match requires identical byte boundaries and entity type. |
| `type_relaxed` | The same one-to-one matching with type ignored; boundaries must still match exactly. |
| `document_accuracy` | Fraction of documents with no exact FP or FN, including correctly predicted empty sets. |
| `anchoring` | Anchored and ambiguous prediction counts/fractions, total ambiguous candidate spans, and extra duplicate anchored proposals. |
| `by_type` | Exact span counts and ratios for every suite-declared entity type, including types absent from locked test. |
| `by_language` | Counts and point estimates grouped by owner-declared language metadata. |
| `exact_interval`, `type_relaxed_interval`, `document_accuracy_interval` | Uncertainty from resampling whole documents and recomputing corpus ratios. |

Matching is one-to-one. Extra duplicate predictions cannot increase recall and
count as false positives. An ambiguous surface form retains all exact source
occurrences and earns no match: the system does not quietly choose whichever
occurrence agrees with gold. That proposal contributes one false positive;
unmatched gold remains false negative. Type-relaxed scoring is a diagnostic for
type mistakes, not a partial-overlap metric.

A ratio with a zero denominator is `null`; in particular, an entirely empty
gold/predicted corpus does not acquire a fabricated F1 of one. An interval is
`null` if any resampled population has an undefined ratio. Document accuracy
still measures empty-set correctness. Bootstrap samples whole documents, not
individual mentions or correlated candidate occurrences. `--confidence-level`
controls statistical intervals, not model confidence. `--bootstrap-seed` makes
statistical resampling reproducible and does not enable model sampling.

The report contains actual model work on all five native axes, mask visits,
independent verification steps and total source verification steps. The latter
includes the two native source checks plus the evaluator's independent third
check. Source membership establishes that a mention occurs in the input; human
annotation correctness remains the corpus owner's responsibility.

Private source text, row IDs, gold intervals, generated tokens, full native
observations and per-row execution identities remain inside the run. The report
exports domain-separated HMAC commitments, collection versions/counts and
aggregate quality. Dataset metadata and language/type aggregates remain public
to the report's recipient.

## Qualify a candidate against a baseline

Supply a complete strict gate file. Every field is required, bounds are finite
numbers in `[-1,1]`, and no policy is inferred from model scores:

```json
{
  "schema_version": 1,
  "max_exact_f1_decrease": 0.01,
  "max_type_relaxed_f1_decrease": 0.01,
  "max_exact_recall_decrease": 0.01,
  "max_document_accuracy_decrease": 0.01
}
```

```sh
fnlp qualify native-ner-int4 \
  --snapshot candidate.int4 --expected-model-sha256 "$CANDIDATE_MODEL_SHA256" \
  --baseline-snapshot baseline.int4 --baseline-model-sha256 "$BASELINE_MODEL_SHA256" \
  --group-width 128 --baseline-group-width 128 \
  --suite ner-suite.json --cases ner-cases.ndjson \
  --gates ner-gates.json --host-scope declared-local-host \
  --key-file assay.key --key-id local-ner-key --namespace ner-evaluation-v1 \
  --memory-mib "$ASSAY_PAIRED_MEMORY_MIB" \
  --minimum-samples 20 --bootstrap-resamples 200 --bootstrap-seed 0 \
  --output ner-qualification.json
```

Each gate measures **baseline minus candidate**. A positive tolerance permits
that amount of decline; zero permits no decline; a negative bound requires
improvement. Gates use point estimates and an undefined required score fails
its gate. `decision.paired` reports candidate-minus-baseline changes and
intervals from identical whole-document draws, plus counts of documents with
better, worse or equal exact FP+FN totals. Both sealed arm scorecards remain in
the report, with complete execution and observation bindings.

`decision.status` is `passed`, `rejected` or `insufficient_data`. All three are
successful complete evaluations and therefore exit successfully; a caller
choosing an operational action must inspect the status. `host_scope_evidence`
is `caller_declared` and `activation_authority` is always false.

## Resource and failure boundaries

| Control | Default and scope |
| --- | --- |
| `--max-spans-per-row` | 64; range 1–4096, both suite predicted-entity capacity and per-row gold capacity. |
| `--max-occurrences-per-row` | 16384; range 1–65536, complete source occurrence evidence for one document. |
| `--max-languages` | 64; range 1–128 across the validated corpus. |
| `--max-verification-steps` | 67108864; range 1–4294967296, shared by all source-verification passes, held-out rows and both arms. |
| `--mask-step-node-visits` | 2000000 per constrained mask traversal. |
| `--max-mask-node-visits` | 1000000000000 across the complete run, including both arms. |
| `--max-new-tokens` | 512 per complete constrained output, including terminal EOS. |
| `--max-result-bytes` | 1048576 per complete native document result. |
| `--max-grammar-states` | 4096; grammar preparation is separately priced in the complete peak. |
| `--max-bootstrap-draws` | 10000000 total document draws; qualification charges both arms. |
| `--max-output-bytes` | 4194304 for the complete final protocol envelope and newline. |

Native forward positions, projected logits, attention pairs, dot products and
multiply-accumulates each have explicit whole-run limits. These limits,
aggregate mask work, verification work and the run deadline never restart for
a second document or baseline model. Complete per-document worst-case work is
admitted before execution, then actual work is charged. A tight remaining
ceiling can therefore refuse the next document even if an unusually short
generation might have fit. No input, occurrence list, native result or report
is silently truncated to fit.

Long documents must fit the declared prompt/context envelope. This evaluator
does not automatically chunk or summarize them, and it selects no active model,
downloads, conversions or fallback backend. Input/pin failures, resource
refusals, typed decode/work/cancellation causes and invalid structured results
produce sanitized error envelopes without successful scorecard publication.

Regression sources cover strict discovery, argument refusal before input,
complete paired memory quotations, held-out bootstrap admission, nested native
error categories and complete output caps. The ignored model fixture requires
explicit independently pinned model, memory and kernel environment settings;
it covers same-model paired replay, exact work accounting, Unicode gold,
source/pin refusal and private report output. Added test sources require the
repository's controller-admitted validation; they are not evidence of measured
model quality or a completed benchmark.
