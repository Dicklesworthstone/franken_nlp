# Candidate user-schema extraction

`fnlp candidate extract` connects exact local JSON schemas and source text to
native constrained INT8 extraction. It requires `asupersync-runtime`, an
explicit local current-candidate `.fnlpq` and a process memory ceiling. It does
not download or activate a model, certify an artifact, or award task quality.

```sh
fnlp candidate extract document.txt --schema schema.json \
  --model ./model.fnlpq --memory-mib 8192
```

Omit the document path or use `-` for stdin. The schema must be a separate local
file, capped at 64 KiB. It is not a remote schema URL or a second stdin reader.
Example schema:

```json
{"type":"object","properties":{"company":{"type":"string","maxLength":128}},"required":["company"],"additionalProperties":false}
```

The supported language is the existing bounded JSON grammar subset, not all of
JSON Schema. Duplicate keys, unsupported keywords and remote `$ref` constructs
are refused. The schema remains exact text throughout compilation, identity
binding and prompting; it is never converted through floating-point JSON values.
The decoder retains its 38-significant-digit decimal domain. The whole typed
result includes the extracted JSON as a string, preserving its numeric lexemes.

## Explicit source membership

Structural mode guarantees only schema validation. To require exact source
quotations, annotate the appropriate string fields with
`"x-fnlp-source":"verbatim"` and pass `--source-membership`:

```json
{"type":"object","properties":{"company":{"type":"string","maxLength":128,"x-fnlp-source":"verbatim"}},"required":["company"],"additionalProperties":false}
```

Source mode requires at least one annotated field. Structural mode refuses
unbound source annotations; neither silently falls back to the other. The
existing source grammar and independent occurrence validator use the exact
original document, not schema names or invented evidence. Unannotated fields
remain ordinary semantic extraction. Membership proves that a quote occurs,
not that a field is semantically correct or that the model found every fact.

## Explicit semantic field review

Use `fnlp candidate extract-semantic` with a separate `--semantic-options FILE`
to review configured scalar claims against the complete original source after
extraction. This route preserves the original JSON and adds explicit correlated,
uncalibrated field diagnostics; it does not repair or automatically accept the
result. See [the complete INT8 semantic extraction example](candidate-semantic-extraction.md)
for the schema, source, options, bounds, and result fields.

## Planning, execution and limits

The actual grammar is checked against the actual source before model metadata
access. The shared extraction planner then compiles and seals exact prompt,
schema, source, policy and model identity before weights are materialized.
Schema property names and source text are separately encoded without privileged
role/thinking/tool controls. There is no flatten-and-retokenize path, sampled
label shortcut, parse-failure retry or second model runtime.

Defaults match the structured source commands: 2048 context tokens, 512 output
tokens including EOS, 65536 document bytes, 1 MiB whole typed result and 512 MiB
modeled preparation. The complete prompt includes schema, document and trusted
scaffold; a large schema can exhaust context even for a short document. Actual
prompt tokens determine admission and native work. Nothing is silently
truncated. Grammar-state, per-mask, aggregate mask, native-work, KV, deadline
and output limits remain enforced by the existing layers.

Preparation accepts the caller's explicit shared cancellation control.
Byte-default preparation checks around its bounded compiler calls; the compact
encoder also polls inside tokenization. Grammar compilation and blocking IO
remain cooperative around calls. The single-request host receives the same owned
sealed executable through `into_extraction_plan`, without copying prompt tokens
or recompiling the grammar. It independently compares admitted model identity
and retains output memory authority through serialization, writing and flushing.
All output retains `scope=real-artifact-current-candidate` and
`evidence=non_authoritative`. A disabled runtime feature refuses before opening
schema/source/model files. Failures use closed diagnostics, never parser
excerpts, schema property names, source snippets or local model paths.

## Compact schema and source input

The single-request command supports an explicit compact input policy:

```sh
fnlp candidate extract document.txt --schema schema.json \
  --model ./model.fnlpq --memory-mib 8192 \
  --input-encoding safe-bpe-v1 \
  --max-tokenizer-pair-checks 4194304 \
  --max-tokenizer-pair-bytes 67108864
```

The schema and document share **one complete tokenizer allowance**. Pair
comparisons, examined pair bytes and produced untrusted tokens accumulate
across both segments. If either segment exhausts the remaining allowance,
preparation refuses without a fallback or renewed budget. The schema remains
separate task data and cannot authorize a source quotation. Both encoded
segments preserve their exact original bytes and exclude template-control
tokens, including when property names or source text contain literal markers.

`--max-input-bytes` still caps the original document; the schema retains its
separate 64-KiB cap. The compact policy's aggregate byte ceiling is the sum of
those two caps, limited to 1 MiB. At the default document setting this is
131072 bytes. A 1-MiB document setting therefore leaves room for the schema
only when the actual document is smaller than its maximum. These original-byte
limits are independent of the actual compiled prompt-token limit, which also
includes trusted scaffolding and reserves the requested output space.

Memory admission prices the complete schema-plus-document encoder workspace
beside the retained original schema, source, grammar and result buffers before
opening schema, document or model files. Larger input limits can require
`--preparation-mib 1024` or a higher explicit reserve; compact token counts do
not discount source bytes or grammar storage. The preparation owner survives
output serialization, writing and flushing.

The selected policy and every tokenizer ceiling bind execution identity even
when two policies produce identical tokens. Compact responses add an
`input_encoding` field with that policy. Extracted JSON remains an exact string,
including 38-digit decimal lexemes; source evidence retains original offsets.
`--selected-rows` composes with compact input and both grounding modes.
Byte-fallback remains the default, with its existing plan identity and response
shape. Compact input changes the prompt token sequence; identical model output,
task quality, tokenizer parity and measured speed are not established by this
wiring. INT8 extraction corpora and retained extraction jobs keep their existing
byte-input contracts.

## Persistent compute workers and prompt prefill

Single-request extraction can use one persistent INT8 projection team and
bounded layer-major prompt processing together:

```sh
fnlp candidate extract document.txt --schema schema.json \
  --model ./model.fnlpq --memory-mib 8192 \
  --compute-threads 4 --prefill-rows 4 \
  --input-encoding safe-bpe-v1
```

`--compute-threads` accepts 1 through 64 and counts the coordinator plus its
row workers; runtime threads are additional. The default of 1 keeps serial
execution. A wider team is formed once for the whole extraction request and
serves prompt processing and constrained decoding across both decoder passes.
`--prefill-rows` accepts 1 through 64 tokens from this one prompt per morsel.
It works with either serial execution or a worker team. Omitting it keeps the
existing sequential prompt schedule. Neither option batches source documents.

Both schedules consume the same identity-sealed extraction plan and vocabulary.
They preserve structural or source-membership mode, the exact original schema
and decimal domain, grammar limits, original source evidence and the requested
output budget. They also compose with `--selected-rows`; the complete legal
grammar row set still has to fit its explicit ceiling. Scheduling does not
change the compact-input policy or introduce free-text sampling controls.

Before schema, source or model IO, admission includes the caller's complete
schema/source preparation, the separately charged transferred plan and
vocabulary, native KV and scratch, coordinator prefill scratch, fixed worker
payload and bounded result staging. The two preparation owners coexist; compact
tokens and selected heads do not discount their original-byte storage. Invalid
geometry or an insufficient memory ceiling refuses before any private file is
opened. The installed runtime capacity must match the complete admitted worker
and prompt-row geometry. Shared request cancellation, checkpoint and deadline
authority remains in force through native cleanup and physical worker join.

Successful worker execution adds a separate `compute` receipt describing
completed physical projections. The exact extraction result remains unchanged
in shape, including its JSON string and source occurrences. Serial results
omit this field, retaining the previous default response shape. Output memory
and the caller's preparation remain held through serialization, writing and
flushing. These options also have the complete-document counterpart below;
extraction corpora and retained jobs retain their existing schedules. Numerical equivalence,
extraction quality and measured speed require model-present evidence.

The library offers `execute_int8_extract_layer_major`,
`execute_int8_extract_with_threads` and
`execute_int8_extract_layer_major_with_threads`. Each takes a sealed plan,
resident model, vocabulary and `SourceLimits` covering the transferred
preparation and native/mask bounds. Threaded variants also take the kernel
and compute width; the layer-major variant places `Int8PrefillLimits`
before the cancellation token. The original sequential
`execute_int8_extract` API retains its existing contract.

## Complete-document extraction with one persistent team

`fnlp candidate map --task extract --schema schema.json` accepts the same
`--compute-threads 1..=64` and `--prefill-rows 1..=64` options. One admitted team
and native context span the complete original document, including prompt
prefill and constrained decoding for every source-aligned chunk. The two
options can be selected independently; their defaults retain serial execution.
Other `candidate map` tasks reject these scheduling options before private IO.

Every chunk still uses the exact original schema and the existing independently
validated extraction plan. The output retains independent JSON strings with
their original decimal lexemes. Required fields must be satisfiable within
each chunk; objects are not merged or repaired. Source-membership evidence
uses original document coordinates. The complete source/result binding is
checked again before output. Native work, grammar-mask work and independent
source-verification budgets never renew at a chunk boundary.

Memory admission includes all retained prompts and grammars, both preparation
owners, worker and coordinator storage, the complete chunk-result frontier,
source-evidence buffers and final serialization. The CLI checks that complete
peak before schema, source or model reads. Increase the explicit preparation
and process ceilings when a larger declared chunk/result frontier requires it;
fewer prompt tokens do not discount retained schema or original source bytes.
The host independently reserves the actual prepared operation. Cancellation,
errors and panic retain those claims until physical worker drain, and any
failed chunk prevents successful document publication.

Threaded success adds a separate `compute` receipt after join. Existing compact
input metadata remains separate, and serial responses omit the compute field.
Result ownership and caller preparation survive serialization, writing and
flushing. The hosted APIs are `extract_int8_document_with_threads` and
`extract_int8_document_layer_major_with_threads`; serial prefill uses
`extract_int8_document_layer_major`. They reuse the same complete-document
finalizer and exact schema/source identity as `extract_int8_document`.

Regression definitions cover combined schedules, all admitted widths and
prompt geometry boundaries, complete-frontier memory, one-byte refusal,
unchanged whole-document limits, exact JSON-string transport and original
native cancellation categories. These source-reviewed definitions remain
unexecuted pending the required DSR checkpoint and do not establish model
parity, extraction quality or measured throughput.

## Evidence scope

Regression sources cover command routing, duplicate/unsupported schemas,
exact decimals, explicit grounding, source/control containment, consuming plan
transfer, context limits and shared preparation cancellation. Compact-input
regressions cover complete workspace admission before IO, original and combined
byte limits, cumulative schema/source tokenizer exhaustion, policy-bound
identity, selected-row composition, every cancellation cause and unchanged
byte-default output. Worker/prefill regressions cover independent and combined
schedules, admitted geometry, exact peak refusal before IO, unchanged sealed
identity under both grounding and encoding modes, selected-head composition
and serial receipt omission. They use real pinned planning and grammar code,
not a fake inference-success producer.
Compilation, Rust tests, model-present execution, fidelity, performance and
controller DSR qualification were not run in this implementation session.
