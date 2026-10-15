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

## Numerical and resource boundaries

Each output keeps its own increasing-group f32 sum, activation scale and group
scale. Each integer group is dequantized in the existing fixed scale order
BEFORE addition; groups with unequal scales are never combined as integers.
BF16 is cast once after all groups. Sparse row selection remains exact: only
requested output rows are visited. Odd row/group boundaries and tail groups
keep the original low-nibble-first layout without padding.

The complete logical group-dot/MAC reservation is charged before preparation
or execution and is never refunded on failure. Every activation, output shape,
scratch bound and aggregate budget is validated first. Polling occurs before
first execution and before a tile would exceed 32 group dots since the previous
checkpoint, even while one group is borrowed across many tiles. A final poll
remains mandatory. Cancellation before the first tile does not mutate scratch.

Only one output column's 64 f32 totals are retained: 256 fixed stack bytes,
not an M*N intermediate or heap allocation. Unpacking scratch remains one
`min(group_width,K)` i8 range regardless of batch size. The store/checkpoint
order can differ from the earlier cohort-major loop; output remains private
staging, and any cancellation, native error or cast failure invalidates ALL of
it. No output-prefix success, hidden retry, cache retention or runtime is added.

## Validation status

Six prepared-group and five batch-regression functions were added. They cover
nibble starts, source-buffer reuse, bounded vector/model lengths, shape-before-
write refusal, independent floating scale order, sparse output rows, 64-row
admission, batch permutation and cancellation at every checkpoint. Existing
batch regressions remain unchanged.

An independent Python scheduling/arithmetic model passed 768 schedule shapes
and 160 f32 comparisons (uniform and heterogeneous selections), preserving
per-cell group order, dot/MAC totals and at-most-32-dot checkpoint gaps. This
was NOT execution of the Rust source or an ISA/model test. Rust compilation,
Rust tests, end-to-end model parity and benchmarks were not run. The controller-
owned DSR and bead-specific numerical/performance gates remain outstanding.
