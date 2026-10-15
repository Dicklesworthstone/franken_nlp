# INT4 group reuse across admitted batches

The packed INT4 batch projection now traverses selected output row, quantization
group, then activation tile. One group is unpacked once and borrowed by every
four-row integer tile in the admitted batch (up to the existing 64-row limit).
Previously the tile loop enclosed the group loop, unpacking every group again
for each tile. For 64 rows this changes group preparations from 16 to one per
selected output row/group. That is a source-level operation-count reduction,
not a measured speedup or claim about DRAM traffic.

## Existing execution paths

Both `GroupedInt4Linear::project_batch_f32_into` and `project_batch_bf16_into`
use this traversal. Existing `strict_int4` layer-major prompt prefill and ragged
sequence-cohort decoder/head calls already use those projection APIs; they do
not need another model, kernel selector, worker pool or CLI flag. It applies to
the scalar and explicitly selected SIMD integer leaves alike. Single-row
projection retains its existing group-by-group floating program. No artifact,
quantization recipe, group scale, sampler or semantic execution identity changes.

`PackedInt4Kernel::prepare_range` validates a bounded logical nibble range and
unpacks it into caller-owned scratch. `PreparedInt4Group` holds an immutable
borrow of that scratch and the existing hardware-checked leaf. Repeated
`dot_rows_into` calls do not read packed bytes again. Safe borrowing prevents
scratch replacement while the prepared group remains in use. The prepared
handle does not borrow the original packed source or serialize model weights.
Existing dense and one-shot range calls use the same unpack implementation.

## Independent sparse projections

`GroupedInt4Linear::project_cohort_f32_into` and `project_cohort_bf16_into` now
accept up to 64 `Int4ProjectionRow` members, independently of the four-row
hardware tile width. Each member supplies its own activation, sorted output
selection, output slice and nonrenewable ledger. A bounded cursor merge visits
only the union of requested output rows. A weight group is prepared once per
union row and then reused by four-row tiles containing ONLY interested members.
Different selected-row counts, sparse overlaps and odd active populations do
not require padding, a dense union-result matrix or projection of unwanted rows.

Every member's complete shape, selection, budget and ledger-overflow check
finishes before any member is charged or any scratch/output is changed. A
request with too little budget cannot borrow another request's surplus. Shared
execution retains independent scales, output indices and full reservations.
Cancellation during any tile fails the whole epoch, including members whose
cells were completed earlier. Metadata and per-column totals use fixed arrays
bounded by 64 members; unpacking scratch is still one group, not 64 groups.

This is an explicit library projection capability. It does not add automatic
request batching, dynamic sequence refill or a grammar/sampling scheduler.
The full `Int4CohortSession::logits_group` API still accepts one common selection;
this change does not claim a new heterogeneous full-session logits interface.

## Numerical and resource boundaries

Each output keeps its own increasing-group f32 sum, activation scale and group
scale. Each integer group is dequantized in the existing fixed scale order
BEFORE addition; groups with unequal scales are never combined as integers.
BF16 is cast once after all groups. Sparse row selection remains exact: only
requested output rows are visited. Odd row/group boundaries and tail groups
keep the original low-nibble-first layout without padding.

The complete logical group-dot/MAC reservation is charged before preparation
or execution and is never refunded on failure. Every activation, output shape,
scratch bound and applicable budget is validated first. Polling occurs before
first execution and before a tile would exceed 32 group dots since the previous
checkpoint, even while one group is borrowed across many tiles. A final poll
remains mandatory. Cancellation before the first tile does not mutate scratch.

The uniform batch retains only one output column's 64 f32 totals: 256 fixed
stack bytes, not an M*N intermediate or heap allocation. Unpacking scratch
remains one `min(group_width,K)` i8 range regardless of batch size. The store/
checkpoint order can differ from the earlier cohort-major loop; output remains
private staging, and any cancellation, native error or cast failure invalidates
ALL of it. No output-prefix success, hidden retry, cache retention or runtime
is added. These are logical storage bounds, not measured peak-RSS results.

## Validation status

Six prepared-group, five batch and four wide-cohort regression functions were
added. They cover nibble starts, source-buffer reuse, bounded vector/model
lengths, shape-before-write refusal, independent floating scale order, sparse
output rows, 64-row admission, batch permutation, per-member budget isolation
and cancellation during a prepared group's reuse. Existing batch regressions
and the original four-row full-model cohort fixture population are retained.

An independent Python model passed 768 schedule shapes, 160 f32 comparisons
(uniform and heterogeneous selections), and 1,200 cursor-merge cases. It checked
per-cell group order, exact selected-row coverage, dot/MAC totals, one preparation
per union row/group and at-most-32-dot checkpoint gaps. This was NOT execution
of the Rust source or an ISA/model test. Rust compilation, Rust tests, end-to-end
model parity and benchmarks were not run. The controller-owned DSR and bead-
specific numerical/performance gates remain outstanding.
