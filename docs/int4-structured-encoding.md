# Compact input for resident INT4 structured streams

The five resident candidate commands `extract-batch`, `ner-batch`,
`keyphrases-batch`, `summarize-batch` and `answer-batch` accept the explicit
`--input-encoding safe-bpe-v1` policy. They use the existing bounded,
byte-preserving, control-excluding encoder. `byte-fallback` remains the default;
the omitted option adds no policy field to the existing output envelope.

```bash
fnlp candidate source-int4 extract-batch documents.ndjson \
  --schema extraction.schema.json --source-membership \
  --snapshot nanbeige.int4.snapshot --expected-model-sha256 "$MODEL_SHA256" \
  --memory-mib 8192 --input-encoding safe-bpe-v1 --compute-threads 4

fnlp candidate source-int4 keyphrases-batch documents.ndjson \
  --snapshot nanbeige.int4.snapshot --expected-model-sha256 "$MODEL_SHA256" \
  --memory-mib 32768 --input-encoding safe-bpe-v1 \
  --cohort-documents 4 --prefill-rows 2
```

The policy composes with exact `--max-selected-rows` decoding. CPU teams remain
incompatible with cohort/packed schedules; choosing an input encoding does not
relax that restriction. Source-converted models and independently pinned
snapshots use the same input contract.

## One allowance per complete record

| Route | Untrusted segments sharing the allowance | Evidence authority |
| --- | --- | --- |
| Extraction | Exact schema bytes and original document | Original document only for annotated verbatim fields |
| NER, keyphrases, summary | Original document | Original document bytes and scalar offsets |
| Passage answer | Question, original passage manifest, joined evidence | Original passage text and its complete passage coordinates |

The allowance bounds cumulative input bytes, emitted input tokens, pair checks
and examined pair bytes. It does not renew between segments. Pair caps default
to 4,194,304 checks and 67,108,864 examined bytes, configurable with
`--max-tokenizer-pair-checks` and `--max-tokenizer-pair-bytes`. Those flags require
explicit Safe-BPE selection. Zero, overflowing or unsupported limits refuse
before private schema/default/input/model IO.

Extraction admits the declared source cap plus the 64-KiB schema allowance,
bounded by the tokenizer's one-MiB combined-input ceiling. A source near that
ceiling leaves less space for the schema. QA's complete request shares the
declared input ceiling, including its rendered passage metadata; limiting each
passage independently would not bound the whole request.

Actual encoded tokens, the complete trusted scaffold and reserved generated
output must fit context. UTF-8 bytes, source spans and exact schema numbers are
preserved independently of token boundaries. Input is never truncated to fit;
pair-work or context exhaustion returns a bounded refusal. The selected policy
is fixed for the stream and binds each prepared task's execution identity.
Per-record `task_args` cannot replace it.

Each new record receives its own tokenizer allowance. Whole-stream transport,
record count, native work, masks, checkpoints and the original deadline retain
their existing cumulative limits across records and flushes. There is no new
claim of a shared whole-corpus pair-comparison counter.

## Admission and library integration

Sequential preparation needs one transient encoder workspace even when several
cohort plans and one lookahead remain live. Both CLI and hosted admission add
that workspace once alongside existing preparation, retained results, native
scratch, KV and IO reservations. Teams retain the charge through worker join
and terminal publication. Selected rows and packed prefill keep their own work
and workspace accounting.

Extraction library callers freeze an `UntrustedEncoding` with
`Int4ExtractionBatchPlanner::with_document_encoding`. Source callers use the
existing `SourceTaskPlanner::with_document_encoding`. The serial, CPU-team and
cohort hosted batch methods inspect the actual policy and use policy-aware
native constructors. Public cohort callers can quote with
`requirements_for_execution_with_document_encoding`. Legacy native constructors,
uncontrolled extraction preparation, and byte-capacity map/job routes reject a
compact policy rather than accepting a quote for a different encoding.

Every compact candidate event carries the complete `input_encoding`, including
typed record errors, flush and `run_complete`, even for an empty stream. The
inner canonical record is wrapped without reserializing numbers or source
coordinates. Successful whole-run consumption still requires both a terminal
completion and exit code zero.

## Qualification status

Regression sources cover per-record aggregate limits, immutable policy binding,
original evidence coordinates, cancellation, exact memory quotes, scheduling
composition, refusal before IO and event provenance. Syntax/source inspection
does not establish executable behavior. Compilation, executable regression
tests, native numerical parity, model quality and performance remain unrun
pending the repository's controller-owned DSR production checkpoint and
model-present gates. Safe-BPE is a distinct input profile; no byte/BPE output
equivalence, tokenizer-oracle parity or throughput claim is made.
