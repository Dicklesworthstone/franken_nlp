# Explicit candidate source-task commands

The `candidate` command family now connects source-backed NER, keyphrases,
cited summaries and passage QA to the existing process-hosted strict-INT8
implementation. These are source-level integrations; they are not a build,
model-quality, numerical-fidelity or release certification.

A binary with `asupersync-runtime` and an explicitly selected compatible local
current-candidate INT8 `.fnlpq` are required. The default feature graph keeps
help/argument parsing but refuses before opening input, options or model files.
There is no download, implicit model selection, catalog activation or network.

## Usage

```sh
fnlp candidate ner article.txt --model ./model.fnlpq --memory-mib 8192
fnlp candidate keyphrases article.txt --model ./model.fnlpq --memory-mib 8192
fnlp candidate summarize article.txt --model ./model.fnlpq --memory-mib 8192
fnlp candidate answer question.json --model ./model.fnlpq --memory-mib 8192
```

The first three commands read exact UTF-8 source text. Omit the input path or
use `-` for stdin. No trimming, normalization or silent truncation is applied.
`answer` instead reads one JSON object:

```json
{"question":"Where did Alice move?","passages":[{"id":"p1","text":"Alice moved to Paris."},{"id":"p2","text":"Bob stayed in Lyon."}]}
```

Question and passage data remain separate in the pinned planner. Evidence
quotes must fit wholly within an original passage. Duplicate passage IDs,
duplicate JSON keys (including escaped aliases), empty questions, unknown
fields and more than 32 passages are refused. The native task also enforces
its existing passage-ID, source-membership and final-result contracts.

Each task uses its existing typed options by default. `--options FILE` accepts
a bounded local JSON file containing the complete options for that task. For
example, NER options can select the type vocabulary and mention bounds:

```json
{"types":["person","organization","location","event"],"max_entities":32,"max_mention_scalars":128}
```

Option files are capped at 16 KiB. Unknown, partial, invalid and duplicate-key
options are errors, not a fallback to defaults. Options cannot change the task,
execution identity, prompt instructions or resource budget. `--options -` is
refused so it cannot consume the source's stdin. Sampling/logprob/thinking/tool
switches are absent: these tasks use the existing constrained greedy decoder.

## Execution and output contracts

The command reserves preparation memory before reading private input or
constructing tokenizers. It parses and validates options, inspects bounded
candidate metadata, builds the pinned control-excluding source planner and
vocabulary, and compiles the exact schema/source task before materializing
weights. All candidate identity fields are compared after independent loading
so a changed path cannot silently select a different admitted model.

`NlpEngine::execute_int8_source` owns native memory, blocking/scoped runtime
entry, finite model/mask work, source finalization and physical completion.
The CLI does not invent a second model or relabel an eager plan as INT8.
The complete prompt plus reserved output must fit `--context-tokens`.

Output is one completed JSON object with the same explicit
`scope=real-artifact-current-candidate` and `evidence=non_authoritative`
provenance wrapper as candidate generation. Task output retains its typed
result and complete native work. No success prefix is written before task
validation. Source spans prove source membership and coordinates, not complete
entity recall or correct entity types. Exact citations do not by themselves
prove that summary bullets or answers are semantically supported.

`--max-result-bytes` limits the entire typed task result, not just its generated
text. Up to 4096 additional bytes are reserved for candidate provenance and the
terminating newline. Serialization is staged and bounded; the result and
preparation memory guards remain alive through stdout write and flush. A
broken pipe can truncate transport bytes but always produces failure.

## Resource and failure bounds

Defaults are 2048 context tokens, 512 constrained tokens, 64 KiB input, 1 MiB
typed result, 4096 grammar states, two million nodes per grammar-mask traversal
and one billion aggregate mask-node visits. Every limit is finite and checked;
invalid limits fail before input/model IO. `--timeout-seconds` covers the whole
invocation, including preparation and loading. Native execution receives only
the remaining duration. Tokenization and blocking filesystem calls are checked
around their bounded operations; this is not OS-thread preemption.

The default 512 MiB preparation reservation is a modeled allocation allowance,
not measured RSS. Hosted source execution independently reserves the same
amount for transferred plans/vocabulary, conservatively overlapping the CLI
reservation. Weight, KV, scratch and output charges also count against the
explicit `--memory-mib` ceiling. Insufficient aggregate memory is a refusal,
not a request to allocate outside the process ledger.

Diagnostics never print nested parser/native errors, private input or paths.
No tests, real-model runs or DSR qualification were executed while implementing
this change. Added regression sources cover all four dispatch routes, option
and QA boundaries, complete prompt/output budgets, pinned plan identity,
preparation deadlines and feature-disabled no-IO behavior.
