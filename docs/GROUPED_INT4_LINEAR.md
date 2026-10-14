# Executable group-scaled INT4 matrices

`native_engine::portable_int8::int4` adds an explicit packed-weight numerical
candidate over the existing dynamic INT8 activation rail. It does not change
`StrictInt8Engine`, the candidate CLI's model loader, or artifact activation.
There is no new `--int4` CLI switch and no whole-model INT4 qualification claim.

## Conversion and execution

`convert::GroupedInt4Matrix::from_bf16_rows` reads one BF16 row at a time.
`from_bf16` applies exactly the same program to a caller-owned contiguous slice.
Both require explicit result-plus-scratch payload, source-value and group-count
ceilings. The row callback is never invoked before those checks and allocation.
An incomplete row, nonfinite source, source error or cancellation returns no
matrix. The callback is synchronous; cancellation cannot preempt a blocked read.

The recipe `bf16-s4-symmetric7-group-f32-rne-v1` uses one f32 scale per row/group,
`max(abs(weight))/7`, with scale one for an all-zero group. Rounding is nearest
even into [-7,7]. The imported packed execution surface also accepts -8, but
this particular producer does not emit it. Group width is always explicit.

Weights are low-nibble-first two's-complement, with no row or group padding.
The complete weight count must be even. Odd row widths and odd group widths
are supported, including shared bytes across those boundaries. All scales are
positive and finite. The owned result produces a borrowed `as_linear()` view;
weights are not expanded into another dense integer or float matrix.

```rust
use franken_nlp::native_engine::{
    decode::{DecodeCancellationKind, DecodeStepControl},
    portable_int8::{ActivationBuffer, LinearRows, ProjectionLedger,
        int4::convert::{GroupedInt4Matrix, Int4ConversionBudget, Int4ConversionRequirement}},
    tensor::Bf16,
};

// Finite demonstration only. A real host supplies its admitted epoch control.
struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}
fn example() -> Result<(), Box<dyn std::error::Error>> {
    let source = [1.0, -2.0, 3.0, 0.5, -1.0, 2.0, 0.0, 4.0].map(Bf16::from_f32);
    let need = Int4ConversionRequirement::for_shape(2, 4, 2)?;
    let budget = Int4ConversionBudget {
        max_payload_bytes: need.peak_payload_bytes().ok_or("payload overflow")?,
        max_source_values: need.source_values,
        max_groups: need.groups,
    };
    let matrix = GroupedInt4Matrix::from_bf16(2, 4, 2, &source, budget, &mut Continue)?;
    let linear = matrix.as_linear();
    let mut input = ActivationBuffer::try_new(4)?;
    input.encode_f32(&[0.5, -1.0, 2.0, 0.25])?;
    let mut ledger = ProjectionLedger::new(linear.planned_work(LinearRows::All)?);
    let mut scratch = vec![0_i8; linear.scratch_len()];
    let mut output = [0.0_f32; 2];
    linear.project_f32_into(&input, LinearRows::All, &mut output,
        &mut scratch, &mut ledger, &mut Continue)?;
    // Only now may this primitive's successful output be consumed by its caller.
    Ok(())
}
```

## Numerical and resource contracts

Every group has its own exact integer dot, then the canonical activation,
row-one, column-one and group-scale epilogue. Those f32 partials are summed in
increasing logical group order. BF16 is cast once, after the complete row.
Summing raw group integers and applying one scale is incorrect. Casting each
group to BF16 before summation is also a different numerical program.

The existing projection ledger counts actual group dots, not output rows:
`selected_rows * ceil(K / group_width)` dots and `selected_rows * K` MACs.
No padded tail MACs are charged. The complete requested work is reserved before
execution and is never refunded on cancellation or partial evaluation.

Execution scratch is `min(K, group_width)` signed bytes plus fixed-size native
and cohort stack tiles. Conversion payload admission includes packed bytes,
f32 group scales and one BF16 source row; it excludes the caller's source,
allocator slack and process/runtime memory. Neither statement is an RSS bound.

## Independent shared-weight cohorts

`cohort::Int4ProjectionRow<T>` supplies one activation, one `LinearRows` mask,
one output slice and one ledger per member. `project_cohort_f32_into` and
`project_cohort_bf16_into` accept up to `MAX_INT4_COHORT_ROWS` members. Masks can
differ. A fixed-space sorted merge visits only their union and unpacks each
weight group once for the lanes that actually need it. Each lane retains its
own activation scale, group order and output ordering. Serial and cohort paths
call the same floating epilogue and BF16-cast helpers.

All members are checked before any ledger is charged or output is touched.
An execution failure invalidates the entire epoch's output, including earlier
completed cells; all full reservations remain charged. The host supplies the
meet-composed epoch control and owns guards, supervision and publication. These
leaves do not spawn, form a runtime, load a model, or implement hidden retries.

Scalar is the default. `with_kernel(PackedInt4Kernel::detect(kind)?)` permits an
explicit real-host-checked native leaf. No measured dispatch table or fastest
kernel is inferred. Matrix `content_sha256()` binds recipe, shape, partition,
packed bytes and little-endian scale bits; it is not a model digest, source
provenance certificate, or publisher authentication.

Rust regressions accompany the code. Native execution, compilation, complete
model conversion/binding, identity-bound task integration, quality comparisons,
DSR and target-host performance remain separate gates; this document awards none.
