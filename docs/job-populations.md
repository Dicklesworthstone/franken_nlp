# Complete NDJSON job populations

`jobs::population::JobPopulation` reads a bounded NDJSON corpus into one owned,
immutable in-memory population. It opens no job files, persists no inputs,
loads no model, and executes no task. Call it inside the embedding host's
admitted invocation; a memory/storage limit is not itself a memory permit.

`read_ndjson` takes the existing job secret and random job ID, `JobLimits`,
separate `PopulationReadLimits`, and the caller's existing cancellation control.
It retains the exact bytes before each LF, INCLUDING an optional CR; only empty
LF and CRLF records are ignored. A final LF is optional. Whitespace-only records,
flush commands, duplicate JSON keys (including nested arguments), invalid UTF-8,
unknown envelope fields and invalid IDs fail the whole population. Arguments
remain in the original bytes, never a reserialized JSON value. The typed native
argument contract is checked again by `JobRunner` before storage is opened.

The full snapshot has one keyed duplicate-ID index, not the live batch epoch
window. Input bytes, all physical lines (including blanks), record bytes,
items, complete-snapshot bytes and the minimum attempt count are independently
bounded with checked arithmetic. The snapshot accounting matches `FrozenManifest`:
ID bytes plus original and normalized lengths. Both representations borrow one
retained envelope in this identity-normalization profile. The transport limits
are ingestion ceilings, not meaning-affecting recipe fields; LF delimiters and
ignored blank lines are not part of a retained envelope.

`inputs` borrows exact records, `borrowed_inputs` creates the bounded reference
index accepted by `JobRunner`, and `freeze` applies the existing complete
execution/recipe/population contract. No mutable view, Clone, Debug, or
Deserialize path exposes a replaceable population. Any read/validation failure
returns no usable prefix, and no inference occurs until ingestion has finished.
This intentionally rejects populations that exceed the in-memory bounds;
external-sort indexing and protected persistent input spooling remain separate.

Cancellation is checked at most 8192 copied transport bytes apart. This is not
preemption of arbitrary blocking `BufRead`, JSON parsing, allocation, or hashing.
The host must price the reader's retained buffers, record capacities, keyed
indices, borrowed index, canonical JSON trees and preparation/allocator overhead.

Eleven regression-test functions are supplied for fragmented reads, exact
record replay, duplicate/invalid later inputs, byte/line/item/attempt budgets,
IO errors, and mid-record cancellation. They are source fixtures only: no Rust
compilation, test execution, native inference or platform qualification was run
for this checkpoint; repository build authority remains as stated in WIRING.md.
