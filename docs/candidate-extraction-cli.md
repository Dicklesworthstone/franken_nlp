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

## Planning, execution and limits

The actual grammar is checked against the actual source before model metadata
access. The shared extraction planner then compiles and seals exact prompt,
schema, source, policy and model identity before weights are materialized.
Schema property names and source text are separately encoded without privileged
role/thinking/tool controls. There is no flatten-and-retokenize path, sampled
label shortcut, parse-failure retry or second model runtime.

Defaults match the structured source commands: 2048 context tokens, 512 output
tokens including EOS, 65536 input bytes, 1 MiB whole typed result and 512 MiB
modeled preparation. Complete prompt bytes include the schema and trusted
scaffold; a large schema can exhaust context even for a short document. Nothing
is silently truncated. Grammar-state, per-mask, aggregate mask, native-work,
KV, deadline and output limits remain enforced by the existing layers.

Preparation now accepts the caller's explicit shared cancellation control
before and after its bounded tokenizer/compiler calls. It does not promise
mid-operation preemption. The single-request host receives the same owned
sealed executable through `into_extraction_plan`, without copying prompt tokens
or recompiling the grammar. It independently compares admitted model identity
and retains output memory authority through serialization, writing and flushing.
All output retains `scope=real-artifact-current-candidate` and
`evidence=non_authoritative`. A disabled runtime feature refuses before opening
schema/source/model files. Failures use closed diagnostics, never parser
excerpts, schema property names, source snippets or local model paths.

## Evidence scope

Regression sources cover command routing, duplicate/unsupported schemas,
exact decimals, explicit grounding, source/control containment, consuming plan
transfer, context limits and shared preparation cancellation. They use real
pinned planning and grammar code, not a fake inference-success producer.
Compilation, Rust tests, model-present execution, fidelity, performance and
controller DSR qualification were not run in this implementation session.
