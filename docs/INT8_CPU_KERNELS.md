# Explicit INT8 CPU execution

The current candidate engine can use the scalar reference, AVX2 X3a
low-7/high-bit decomposition, or AVX2 X3b signed-i16 widening. The AVX2
implementations are experimental code-first candidates, not measured dispatch
winners, model-quality certificates, or release qualification. Scalar remains
the default. Other ISA tiers and SIMD INT4 execution remain separate work.

## Candidate CLI

Use an explicitly selected local candidate artifact with the existing
`asupersync-runtime` build. No model is downloaded or activated by this option.

```sh
fnlp candidate --int8-kernel x3b_avx2_widened_i16 generate prompt.txt \
  --model local.fnlpq --memory-mib 8192

fnlp candidate --int8-kernel x3a_avx2_low7_high_bit batch \
  --task ner --model local.fnlpq --memory-mib 8192 < documents.ndjson
```

The option is available throughout the candidate command tree, including
streaming, classification, extraction and native corpus/job routes. It selects
the integer leaf in every decoder projection and the vocabulary head, including
selected-row and grouped projections. It does not alter the tokenizer, task
compiler, sampling policy, source validation, integer-work budget, fixed scale
order, BF16 cast program, or output schema. The stderr selection line identifies
the explicit kernel; it is not a benchmark claim. Unsupported CPUs are refused
before input/model access. Unknown names never degrade to scalar.

Use `--int8-kernel s_scalar_i32` for an explicit scalar comparison, or omit the
option for the unchanged default. There is no `auto` or `fastest` option. Both
AVX2 implementations check actual OS-aware CPU capabilities; a fabricated
feature report cannot authorize instructions.

## Embedding

Before binding the first model, an embedding host can call
`native_engine::simd::config::configure_process_kernel(Int8KernelKind::Avx2WidenedI16)`.
The process default is immutable after its first selection or model binding;
repeating the same selection is harmless, while a late conflicting request is
an error. `configured_process_kernel()` only observes, without freezing a default.
A host performing explicit per-instance comparisons can instead apply the
hardware-checked `Int8Kernel` to `Int8WeightView::with_kernel` before engine
construction. Both options borrow the existing weights rather than copying or
expanding them. Leaf kernels neither spawn workers nor inspect the environment.

## Evidence boundary

The Rust differential definitions exercise scalar/i64 agreement, every byte
pair, full model reduction sizes, unaligned vector tails, scalar/BF16 projection
bits, selected-row behavior, grouped execution and cancellation/work accounting.
They have not been compiled or executed in the implementation environment.
An independent local C translation of the two instruction algorithms completed
82,541 differential cases with zero mismatches under UBSan, including protected
page endings for every length from zero through 10,752. That is an arithmetic
and memory-access cross-check of the C translation, not Rust, DSR, model parity,
Apple/Zen host performance, or a Bead/release gate receipt. Those gates remain
independent and must be run against the actual Rust source and target hosts.
