# Long-document native redaction

`fnlp candidate redact --chunked` applies the existing redaction actions to a
bounded document that need not fit one native NER context. It combines whole-text
rule detection with source-aligned INT8 NER chunks, independently verifies every
proposed occurrence, and performs one edit over original-document coordinates.
Verification is enabled by default and reruns BOTH detector families on the actual
transformed text with a newly computed partition.

```sh
fnlp candidate redact report.txt --chunked \
  --model ./model.fnlpq --memory-mib 12288 \
  --max-new-tokens 128 --max-total-mask-node-visits 128000000000
```

The memory and work allowances in this example are explicit illustrative bounds,
not measured requirements or a promise that every input will fit. The command
requires `asupersync-runtime` and a compatible explicitly selected local candidate
INT8 artifact. It does not download, discover or activate a model. Input is exact
plain UTF-8, not NDJSON. Omit the path or use `-` for stdin except when stdin is
explicitly reserved for a pseudonym key. Empty long-mode documents are refused.

Without `--chunked`, the existing single-context native redaction path and output
remain unchanged. `fnlp redact --rules-only` is the separate model-free command and
is unchanged. Long-mode options require `--chunked`; they are never silently
ignored by the short path. Unavailable runtime builds refuse before key, source,
NER-options or model I/O.

## Complete detection before one edit

The rule scanner sees the ENTIRE source. An email, URL or other selected rule
match crossing a NER chunk boundary is therefore still visible to that scanner.
Rules are not independently applied to fragments and concatenated afterward.
The existing versioned rule detectors retain their narrow coverage and budgets.

NER uses the actual pinned scaffold, schema and source encoder to tighten the
requested chunk token capacity. The source partition covers every original byte
exactly once, preserving Unicode scalar and CRLF boundaries. All exact native
plans for a stage are compiled and admitted before its first forward. There is
one resident model and one native engine, with reset contexts between chunks.

Each unchanged NER result is independently rescanned against its own source chunk.
Its task/profile, selected entity types, complete repeated-occurrence list and
byte/scalar coordinates must agree. All verified occurrences are lifted to the
original document. Duplicated proposals are verified and charged before overlap
union. A missing or corrupt last occurrence, wrong scalar coordinate, omitted
chunk or changed profile fails the whole operation, not just that chunk.

Whole-text rule matches and lifted NER spans enter the existing connected-overlap
union. It covers the entire overlapping region and retains the participating
kinds and detectors. The existing edit engine then applies mask, placeholder or
pseudonym actions once against unchanged original coordinates. Untouched slices
remain byte-exact. `--include-map` retains the existing opt-in original/output
coordinate mapping; no native NER transcripts or raw matched strings are added.

**Independent NER chunks can split entities or lose context.** This is not an
exhaustive entity census, an overlapping-window model or proof of PII recall.
Names spanning a boundary may be missed by NER even though whole-text rule
matches remain available. A valid source span proves location, not correct entity
typing. Pseudonyms and a clean declared-detector scan are not anonymization.

## Fresh verification and bounded failures

After the edit, verification scans the whole transformed text for rules and
re-tokenizes/re-partitions that transformed text for fresh native NER. Original
names, proposals, chunk boundaries and offsets are not reused as a shortcut.
Placeholders and pseudonyms may expand the text, so the second stage may need a
different number of chunks and more context or output allowance. If it exceeds
any configured bound, no successful redaction result is returned.

Residual detections reject publication of the edited text. The lower-level
borrowed API retains a bounded report of transformed-source coordinates, kinds
and detector scope, without matched text. The hosted path discards those vectors
under its live reservation and returns the established typed residual-count
error. It does not return a large uncharged error payload. `--no-verify` remains
an explicit opt-out and produces `verification: not_requested`.

A completed long result has `execution: portable-int8-whole-rules-chunk-ner-redetect-v1`,
`detector_scope: whole-source-rules-independent-ner-chunks-v1`, final `result`,
per-stage original/verification geometry and work, aggregate reserved/actual
native counters and mask visits, and independent grounding scan work. Its warning
fields state the boundary, recall and anonymization limitations. The candidate
wrapper keeps `evidence: non_authoritative`.

The inner `clean_declared_union` status means only that this declared whole-rule
and independent-chunk NER rerun found no residual detections. It is not a
single-context-equivalence or privacy certificate. Only a completed envelope is
published. Native failure, cancellation, budget refusal or residual detection
publishes no successful partial text. A transport write/flush failure can still
leave partial external bytes and is an error, without retry.

## Limits

Existing `--max-new-tokens` and `--max-result-bytes` bound EACH native NER chunk.
The latter also bounds the complete final long-redaction envelope. Existing
`--max-input-bytes` bounds the original read (65536 bytes by default, at most
1 MiB). Transformed-source planning and rule scans allow the larger of the
original input ceiling and final result ceiling; this does not enlarge the read.

| Long-mode option | Default | Scope |
| --- | ---: | --- |
| `--max-ner-chunks` | 64 | Maximum chunks per original/transformed stage; maximum 256 |
| `--max-ner-chunk-bytes` | Context-token count as a byte ceiling | Additional byte ceiling, not an estimate of tokens |
| `--max-ner-tokenizer-calls` | 8192 | Partition tokenizer calls per stage |
| `--max-ner-map-bytes` | 16777216 | Complete intermediate map and per-value byte cap per stage |
| `--max-total-mask-node-visits` | 64000000000 | BOTH stages together |
| `--max-forward-positions` | 262144 | BOTH stages together |
| `--max-projected-logits` | 10000000000 | BOTH stages together |
| `--max-attention-pairs` | 1000000000000 | BOTH stages together |
| `--max-dot-products` | 1000000000000 | BOTH stages together |
| `--max-multiply-accumulates` | 10000000000000000 | BOTH stages together |

The map cap is at most 16 MiB; live and cumulative serialized-value limits are
respectively twice and four times this cap. One map runs at a time. The existing
`--max-mask-node-visits` is per NER chunk and its per-mask ceiling remains separate.
All conservative original-stage native work and mask reservations are charged
before its first forward. Verification can use only the remaining whole-run
allowance. Early EOS does not refund the original reservation. Raising a chunk
count cannot multiply compute authority; default work/mask ceilings may refuse
an operation below the per-stage chunk count cap.

`--max-detections` bounds the complete rule-plus-NER occurrence count within each
stage, before overlap deduplication, and bounds final edit regions. Rule work is
bounded per whole-source scan. Independent NER rescanning additionally shares
one `GroundingBudget` across all chunks and both stages (CLI defaults: 4096 fields,
16384 matches and 67108864 scan units). These allowances do not reset at a chunk
or verification boundary. Embeddings can set explicit GroundingBudget values in
RedactionRequest. No automatic narrowing of detector scope is used to fit a limit.

## Process ownership and embedding

Use `NlpEngine::redact_int8_document` with the existing resident model, owned source,
pinned planner/vocabulary Arcs, `RedactConfig<LongRedactionConfig>`, optional owned
RedactionPseudonyms and one cancellation token. The default `RedactConfig` type
still selects the existing short detector. These are concrete static entry points,
not a runtime plugin or caller-supplied native implementation.

The host checks model/domain identity and FULL resident KV capacity. It retains
actual source capacity and key/namespace storage with the explicit preparation
reservation. The temporary reservation prices all admitted native chunk outputs,
coordinate lifts, reduction values and serializer staging, plus existing edit
headroom. It is not sized only to the final redacted text. Prepared source plans
and intermediate NER transcripts die before their reservations; final output
ownership survives serialization, writing and flushing. These are modeled ledger
commitments, not measured or OS-enforced RSS limits. Larger chunk/grammar limits
can require higher `--preparation-mib` and aggregate `--memory-mib` allowances.

The CLI checks rules and exact original-stage planning before weight loading.
The transformed stage cannot be preplanned before editing; its entire plan is
admitted after transformation but before its first verification forward. One
cooperative controller spans all native chunks, occurrence checks, edits and
final checks. Blocking I/O and individual bounded tokenizer/scanner operations
are not preempted mid-operation. A failed engine should be discarded rather than
retried through a partially executed invocation.

The lower-level `Int8DocumentRedactor` reuses the actual source-map engine and
requires the embedding application to retain its resource/error-output guards.
It is not a model activation receipt. Private prompt/source digests and generated
NER token transcripts are not exported by the long-redaction envelope.

## Grouped INT4 documents and explicit native chunk cohorts

`fnlp candidate source-int4 redact-document` provides the same complete-document
rule/NER/edit/verification composition using a selected grouped INT4 source
conversion or independently pinned local snapshot. It defaults to serial NER
chunks. Add `--cohort-chunks N` to execute up to `N` independent NER chunks
together through the existing native cohort decoder, for both the original
document and the freshly partitioned transformed document.

```sh
fnlp candidate source-int4 redact-document report.txt \
  --source-dir ./pinned-source --source-manifest ./source-manifest.json \
  --memory-mib 24576 --cohort-chunks 3
```

The memory ceiling is illustrative, not a measured requirement. The selected
group width, context, conversion route and declared chunk/result limits determine
admission. A snapshot uses `--snapshot ./model.int4` together with an independently
trusted `--expected-model-sha256`; its cohort route is identical after loading.
This command requires the `asupersync-runtime` feature and does not download a
model or switch backends.

The INT4 command defaults to a 1 MiB original input cap, 1 GiB preparation
reserve, 64 NER chunks per stage, and 256,000,000,000 total mask visits across both
stages. Its whole-operation dot-product ceiling is 10,000,000,000,000,000. The
other complete-map, source-evidence, rule/action and nonrenewable-budget contracts
above apply. `--max-source-fields`, `--max-source-evidence-spans` and
`--max-source-scan-steps` explicitly bound each independent evidence pool.

`--cohort-chunks` accepts 1 through 64 and cannot exceed `--max-ner-chunks`. An
explicit width of 1 selects the cohort executor; omitting the option selects the
existing serial executor. The last window may contain fewer rows because the
document has ended. The implementation never silently lowers the selected
window or retries serial execution after a refusal. `redact-corpus` and
`redact-job start|resume` reject this option until those routes support it.

The complete declared cohort window is priced before key, document or model I/O.
Its native KV, RoPE, scratch and decoder workspace remain reserved through both
stages. The second stage can grow after placeholders or pseudonyms expand the
text, so admission does not use only the original document's observed peak.
The stages reuse their cohort memory; they do not allocate two simultaneous
native windows. Complete source plans, all retained chunk outputs, independent
coordinate checks, edit storage and final result ownership remain separately
charged. A task's per-chunk KV ceiling stays unchanged when a cohort is selected.

The transient native cohort result has a 64 MiB ceiling, including framing. At
width 64, the default 1 MiB per-chunk `--max-result-bytes` allowance exceeds that
ceiling once framing is added and is refused. For example, explicitly choosing
`--max-result-bytes 524288` admits that result-envelope geometry; it also tightens
each chunk's output and the final semantic result. Other memory and work limits
must still fit. No limit is silently resized to make a cohort run.

All original chunk plans and actual cohort geometry are checked before a
requested `--save-snapshot` is written. After editing, the entire new partition
is checked before the first verification forward. Each chunk retains its own
grammar, causal KV and execution identity. The complete native/evidence result
passes through the same independent rescans, coordinate lifting, collision and
overlap policy, full256 pseudonym scope, and final edit as serial execution.
Original and verification stages consume the same whole-operation work, mask and
evidence ledgers; neither a new cohort nor a stage boundary renews them.

Successful serial INT4 output keeps protocol
`fnlp-candidate-int4-redact-document-v1`. Explicit cohorts use
`fnlp-candidate-int4-redact-document-cohorts-v1`. The cohort wrapper separates
physical scheduling from the semantic result:

| Field inside the outer `output` | Meaning |
| --- | --- |
| `execution` | `grouped-int4-document-redaction-chunk-cohorts-v1` |
| `cohort_chunks` | Explicit maximum simultaneous chunk count |
| `original_cohorts` | Original-stage chunk/window counts, grouped steps, projection groups and work |
| `verification_cohorts` | Corresponding fresh verification counters, or `null` with `--no-verify` |
| `output` | Existing complete-document semantic redaction, including both stage receipts and final edited result |

The wrapper adds a bounded 4096-byte allowance around the existing semantic
result cap. Both routes stage the entire JSON response, recheck the original
deadline before publishing, and retain output ownership through writing and
flush. Serialization failure, cancellation, an incomplete stage or a deadline
reached before publication emits no successful result. Transport failure or a
late flush can still leave external bytes and is reported as an error.

Embedding applications use `NlpEngine::redact_int4_document_cohorts` with the same
`RedactConfig<Int4LongRedactionConfig>` as the serial host, selecting the window
through `detector.mapping.reduction.map_batch_chunks`. The pure
`Int4LongRedactionConfig::cohort_requirements` quote covers either stage's maximum
window. Lower-level callers can preflight actual original geometry with
`Int4DocumentRedactor::preflight_cohorts` and run with `redact_cohorts`, while
retaining the declared memory and error-output owners. The returned
`Int4LongRedactionCohortRun` owns the unchanged semantic result and separate
physical counters. Its `validate_schedule` checks internal consistency; it does
not authenticate an externally supplied receipt.

The source includes focused admission, all-axis budget, identity preservation,
fresh-verification, schedule-corruption, mode-isolation and deadline-publication
fixtures. An ignored model gate,
`source_scalar_and_pinned_snapshot_cohorts_match_after_complete_repartitioned_verification`,
compares complete semantic results from scalar source conversion against pinned
snapshot serial and cohort widths 1 and 3 under scalar and an explicit available
SIMD kernel. It checks original and transformed stage work independently and
requires multiple cohort windows. It requires `FNLP_INT4_TEST_SOURCE_DIR`,
`FNLP_INT4_TEST_SOURCE_MANIFEST`, `FNLP_INT4_TEST_MEMORY_MIB`,
`FNLP_INT4_TEST_REDACT_DOCUMENT_SNAPSHOT` naming a new file, and
`FNLP_INT4_TEST_KERNEL`; it never deletes an existing snapshot. This gate has not
run in this session. Shared physical-step counts are structural data, not a
latency or throughput measurement. DSR compilation/tests, actual-model parity,
NER quality and target-host performance remain pending.

## Validation handoff

The implementation adds 26 model-free regression definitions: 12 core, five host,
five CLI (including a feature-disabled test), and four pinned-preflight/delivery
checks. They cover whole-text rules across neural boundaries, repeated Unicode
mentions, corrupt final evidence, full source coverage, overlap provenance,
shared grounding/native budgets, fresh residual coordinates, typed cancellation,
full KV/temporary memory admission, mode isolation and output-accounting changes.
Private synthetic evidence is used only for validation/corruption fixtures, not
as evidence that a neural model executed successfully.

Source/API, ownership and targeted GitHub diff review were performed. Rust
compilation, Cargo tests, scripts, Actions, DSR and full-model execution were NOT
run; executable validation remains with the designated controller. Real-model
redaction recall, residual-detection quality, boundary effects, short/long
comparison and performance remain unmeasured. No passing-build, recall or safety
claim is made by this composition.
