# Long-document entity discovery and resolution

`fnlp candidate resolve --discover-entities --chunked` discovers named entities
in source-aligned chunks, lifts independently checked occurrences into the exact
original documents, then resolves ONE complete document snapshot. It does not
turn chunks into artificial documents or cluster each chunk separately.

```sh
fnlp candidate resolve snapshot.json \
  --discover-entities --chunked \
  --model ./model.fnlpq --memory-mib 12288 \
  --graph-reserve-mib 128 --max-input-bytes 1048576 \
  --max-ner-tokens 128 --max-ner-chunks 64 \
  --max-snapshot-ner-chunks 256 \
  --max-projected-logits 2000000000
```

These are illustrative finite allowances, not measured memory requirements or a
promise that every input fits. The remaining native-work, graph, mask and output
limits still apply. A compatible local current-candidate INT8 artifact and an
`asupersync-runtime` build are required. No artifact is implicitly selected,
downloaded, authenticated or promoted to a release by this command.

## Input and mode isolation

Input is one bounded JSON snapshot, not an NDJSON stream:

```json
{
  "documents": [
    {"id": "report-a", "text": "Alice discussed the project in London."},
    {"id": "report-b", "text": "Alice later joined the review."}
  ],
  "options": {
    "blocking": "ascii_word_overlap",
    "context_scalars": 128,
    "minimum_margin_milli": 1000
  }
}
```

The optional top-level `ner` member accepts the existing complete typed NER
options. Omission selects the existing person, organization and location types.
The explicit `options` member supplies the resolution blocking/context/margin
policy; its margin is not calibrated confidence. Original IDs must be unique,
bounded and free of control characters. Original text is not normalized or
trimmed. Per-document mentions, scores, chunk instructions and execution
identities cannot be supplied in discovery mode.

Without `--discover-entities`, the existing supplied-mention input remains
unchanged. With discovery but without `--chunked`, the existing single-context
NER path remains unchanged. `--chunked` requires discovery, and all new chunk
options require `--chunked`; they cannot be silently ignored in another mode.

An empty snapshot (`"documents": []`) can produce a genuine empty result without
loading weights. Configuration, local artifact metadata and task identities are
still checked. Empty individual documents are refused in chunked mode, rather
than being represented as successful NER without an actual nonempty partition.
A nonempty snapshot cannot use the model-free finalization path.

## Original-document semantics

Every original document is partitioned losslessly with the pinned source encoder.
The same code-owned NER schema and template fragments used by actual task planning
price the scaffold. Chunk token capacity is the remaining prompt/context room
after reserving the exact scaffold and all possible output tokens. There is no
byte/token estimate, substituted tokenizer, placeholder prompt or source
truncation. Every actual chunk is independently compiled before the first neural
call in the snapshot.

Preparation retains only the original strings and compact private witnesses:
chunk extents, complete execution-identity hashes, prompt lengths and native-work
commitments. Execution rebuilds one grammar at a time and requires its identity,
prompt length and work to match the corresponding witness. It does not retain an
array of native transcripts, grammars or map outputs. The witness hashes are not
published as content-derived identifiers.

Each NER result is independently checked against that chunk's exact original
bytes. All reported occurrences and byte/scalar coordinates must agree with a
fresh bounded occurrence scan, including duplicated proposals before deduplication.
Checked offsets are lifted into the original document with checked arithmetic.
Gaps, overlaps, broken UTF-8 boundaries, scalar drift and incomplete partitions
refuse the snapshot. A proposal in one chunk does not label identical text in
another chunk that did not itself propose that entity.

Only after all discovery succeeds is the resolution graph prepared. Its mentions
retain original document IDs and original byte/scalar offsets. Pair context comes
from the original document, so it may cross a NER chunk boundary. The candidate
set includes applicable same-document cross-chunk pairs as well as cross-document
pairs. The graph is not built from concatenated documents or synthetic chunk IDs.

The existing resolver is reused unchanged: lexical overlap only selects candidate
comparisons; both model presentation orders must support a match under the fixed
margin policy; complete-link clustering refuses a merge when a required cross-pair
is missing, abstains or conflicts. Cluster identifiers remain snapshot-local.
Increasing document size does not authorize lexical-only merging or partial
candidate enumeration.

## Resource scopes

| Option | Meaning in chunked discovery |
| --- | --- |
| `--max-ner-tokens` | NER output tokens including EOS per chunk; distinct from pair-label candidate depth |
| `--max-ner-mask-node-visits` | Grammar-mask allowance per chunk |
| `--max-snapshot-mask-node-visits` | All discovery chunks across the entire snapshot |
| `--max-ner-chunks` | Chunks per original document; default 64, maximum 256 |
| `--max-snapshot-ner-chunks` | Additional whole-snapshot chunk ceiling; default 1024, maximum 16384 |
| `--max-ner-chunk-bytes` | Additional source-byte cap per chunk; default 4096, at least 4 bytes |
| `--max-ner-tokenizer-calls` | Partition tokenizer calls per document; default 8192, maximum 1000000 |
| `--max-input-bytes` | Complete input JSON bytes; still bounded by the existing CLI ceiling |
| `--max-expanded-bytes` | Original IDs/text plus expanded mention strings across the snapshot |
| `--max-mentions`, `--max-pairs` | The complete original-document graph, not a per-chunk quota |

The five existing native ceilings (`--max-forward-positions`,
`--max-projected-logits`, `--max-attention-pairs`, `--max-dot-products`,
`--max-multiply-accumulates`) cover the ENTIRE invocation: all discovery chunks
and subsequent pair scoring. Enabling chunking does not multiply or reset them.
The exact native reservation for every chunk is summed before inference. Pair
planning receives the remaining whole-snapshot allowance after subtracting all
reserved NER work, intersected with its existing stage cap. Early EOS does not
refund the conservative NER reservation to pair scoring.

The initial document-count mask check is only a lower bound in chunked mode.
Exact preflight checks the actual chunk count and its complete mask reservation;
a snapshot can pass its wire/input checks and still fail before weights are
loaded. Occurrence recovery shares one fields/matches/scan budget across all
chunks and documents. Graph validation has its existing separately bounded scan
and complete candidate/clustering budgets. Any exhausted axis refuses the
operation rather than silently dropping a document, chunk, occurrence or pair.

## Host, memory and publication

One resident strict-INT8 model and one native engine serve discovery and pair
scoring. Full resident KV capacity is checked against both stages' authority;
logical KV must be clean between native operations. A single process-owned
blocking invocation supplies cancellation and the native resource ledger. There
is no per-chunk model reload, second scheduler, retry, checkpoint resume or
incremental clustering service.

The CLI reserves preparation and graph/witness headroom before building the
snapshot. Compact witness capacity contributes to the graph floor; the host
independently charges actual retained original-string and witness-vector
capacities. Temporary host storage prices one NER result and token vector plus
the complete expanded mention/pair graph. The preflight charge conservatively
overlaps the host charge until the consumed prepared snapshot has drained.
Pinned planners, compilation and allocator headroom remain explicit preparation
commitments. These are modeled reservations, not allocator interception or
measured or operating-system-enforced RSS limits.

Successful output is one completed candidate JSON object. Its new execution
wrapper contains `chunk_profile`, `document_chunks` with each original ID and
its NER chunk count, and `output`, the existing aggregate discovery/resolution
result. Discovery work and mention counts aggregate all chunks belonging to
each original document. `output.resolution` contains the complete graph result.
The wrapper uses execution
`portable-int8-chunk-ner-original-document-resolution-v1` and includes explicit
chunk-boundary and unestablished-quality warnings. The outer candidate evidence
remains `non_authoritative`.

No partial discovery graph or intermediate NER text is published if a later
chunk, occurrence check, pair plan, native call or completion check fails.
Metadata, document counts, chunk geometry totals and reserved work must agree
before delivery. The complete result, including wrapper overhead, is checked
against the result-byte ceiling, and the hosted output guard survives final
serialization, writing and flushing. A transport error can still leave partial
external bytes and is an error, not successful delivery.

Embedding applications use
`corpus::entities_int8::long::prepare_int8_document_entities` with
`Int8DocumentEntityConfig`, then transfer the consumed
`PreparedInt8DocumentEntityCorpus` and pinned vocabulary to
`NlpEngine::execute_int8_document_entities`. The existing single-context
`prepare_int8_entities` and `execute_int8_entities` APIs remain unchanged.

## Detection limits and validation status

NER remains chunk-local. A boundary may split an entity or remove context needed
to recognize it. Recovered source coordinates prove an occurrence exists, not
that the proposed type is correct or that all entities were found. Original-document
pair context does not restore an entity omitted during discovery. Resolution
quality and recall remain unestablished; no global cross-snapshot entity identity
or semantic-equivalence guarantee is introduced.

This extension adds 24 model-free regression-test definitions: 13 core,
7 CLI contracts (including one feature-disabled test), and 4 runtime integration
fixtures. They cover real pinned preparation, original-document cardinality and
contexts, Unicode lifting, repeated occurrences, complete snapshot budgets,
identity rebuild checks, malformed or corrupt later chunks, typed cancellation,
mode isolation, genuine empty finalization and completion rejection. Private
synthetic NER results are fault-injection fixtures, not native-success evidence.

Source/API, ownership and targeted GitHub diff review were performed. Compilation,
Cargo tests, scripts, Actions, DSR and full-model execution were NOT run under the
repository's controller-owned validation policy. A passing build, model-present
parity, recall, performance and release qualification are not claimed.
