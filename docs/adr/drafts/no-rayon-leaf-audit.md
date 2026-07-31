# No-Rayon leaf audit (pre-write)

**Bead:** `franken_nlp-xmx`
**Audit target:** `/dp/frankentorch` at
`523aaf827faf538aa541126ee222fcd7af348410`
**Status:** pre-write analysis; this document does not select a new Suite pin or
land a FrankenTorch implementation.

## Decision boundary

FrankenNLP's release graph must contain neither `rayon` nor `rayon-core`.
The reason is ownership, not merely oversubscription: an inference leaf must
not install, enter, or enqueue work onto a scheduler outside the
asupersync-owned request/epoch CPU team. Setting `RAYON_NUM_THREADS=1` does not
satisfy that contract: the dependency and pool/job API remain present.

At the audited source pin, `ft-kernel-cpu` cannot be consumed as that leaf
surface. This is a source-level finding. The release-graph and no-spawn tests
named below remain required at the selected new pin.

## Audited source and manifest facts

The audited checkout was clean at the revision above. Its root
[`Cargo.toml`](/dp/frankentorch/Cargo.toml) has one `[patch.crates-io]` entry:

```toml
block = { git = "https://github.com/Dicklesworthstone/rust-block", rev = "b39ae859d1ee8e8cb5eef6a516471f1578d26b96" }
```

That patch is the reviewed `metal 0.33.0` / `block 0.1.6` closure repair. It
does not patch, configure, or make Rayon optional. Any selected FrankenTorch
pin must record this root patch closure exactly in `SUITE_ROOT_PATCHES.lock`; a
new leaf route must not accidentally drop or float it.

[`crates/ft-kernel-cpu/Cargo.toml`](/dp/frankentorch/crates/ft-kernel-cpu/Cargo.toml)
currently has these release dependencies:

```toml
[dependencies]
ft-core = { workspace = true }
libm = { workspace = true }
matrixmultiply = "0.3"
rayon = "1.10"
wide = "0.7"
```

There is no `[features]` section in that manifest. Therefore
`default-features = false` at a downstream call site cannot remove the direct,
unconditional Rayon edge. Making the dependency optional is also insufficient
until every unconditional import and every selected call path is separated from
it.

The current implementation begins with `use rayon::prelude::*;` at
`crates/ft-kernel-cpu/src/lib.rs:11`; its `gemm` module repeats that import at
line 21. The source has no existing feature partition that can compile the
needed primitives without those imports.

## Exact scheduling edges found

The following entrypoints are in the requested ADOPT/ADAPT set, or are direct
dependencies of that set. Each currently reaches Rayon scheduling machinery or
a wrapper that does so.

| Current function and source location | Current Rayon/scheduling edge | Required no-Rayon replacement |
| --- | --- | --- |
| `gemm::sgemm` (`src/lib.rs:1148`) and `gemm::sgemm_bt` (`:1268`) | Shape-dependent `par_chunks` / `into_par_iter`; `gemm::should_parallelize` reads `rayon::current_num_threads` (`:89-98`). | Caller-scheduled `leaf_sgemm_block_into` and `leaf_sgemm_bt_block_into`, derived from the existing serial `gemm::sgemm_block` (`:1384`) and `gemm::sgemm_bt_block` (`:1362`) rather than the public wrappers. |
| `quantize_per_output_channel_i8` (`:30842`) | Local Rayon prelude import and `par_chunks_mut` threshold path. | `quantize_per_output_channel_i8_serial` plus an explicit output-row range/into form. Preserve channel order and quantization bytes exactly. |
| `quantize_rows_i8` (`:30885`) | Local Rayon prelude import and `par_chunks_mut` path. It is used by both dynamic-linear routes. | `quantize_rows_i8_range_into`; caller supplies the row interval and destination scratch. |
| `linear_int8_dynamic_f32` (`:30966`) | Calls `quantize_rows_i8`; both the AArch64 SDOT and portable paths use `out.par_chunks_mut`. | `linear_int8_dynamic_f32_rows_into`, with `[row_start, row_end)` and caller-owned output/scratch. It must retain the current scalar/SDOT/VNNI algorithm choices and exact i32 accumulation contract. |
| `linear_int8_dynamic_prepacked_f32` (`:31138`) | Calls `quantize_rows_i8`; its SDOT and fallback paths both use Rayon chunks. | `linear_int8_dynamic_prepacked_f32_rows_into`, also range-owned by the caller. `pack_int8_weights_nr4` (`:30922`) is already serial and is the appropriate packing building block. |
| `rms_norm_forward_f32` (`:5680`) | The row operation has a serial branch, but the public function selects a Rayon `par_chunks_mut` branch at `:5705-5708`. | `rms_norm_forward_f32_rows_into`, exposing the existing row math over a caller-owned row interval without a scheduler decision. |
| `sdpa_forward_f32` (`:4630`) | Its block calls public `gemm::sgemm_bt` (`:4659`) and `gemm::sgemm` (`:4688`); its outer head execution reads Rayon thread count and uses `out.par_chunks_mut` in both branches (`:4692-4702`). | `sdpa_forward_f32_head_query_range_into`, built only on serial leaf GEMM/range primitives. The caller owns head and query-tile assignment and output storage. |
| `matmul_rhs_transposed_contiguous_f32` (`:12340`) and `matmul_tensor_contiguous_f32` (`:30758`) | The former calls `gemm::sgemm_bt` (`:12363`); the latter reaches the scheduling GEMM wrappers. | Serial/range matmul primitives that call only the serial leaf GEMM functions. |
| `softmax_dim_tensor_contiguous_f32` (`:32193`) | The contiguous row path uses `par_chunks`; other paths consult `rayon::current_num_threads`. | `softmax_dim_tensor_contiguous_f32_serial` or a range form that preserves the present scalar/libm reduction and NaN behavior. |
| `argmax_dim_tensor_contiguous_f32` (`:32443`) | The wide and per-lane paths consult the pool and use Rayon chunks. | `argmax_dim_tensor_contiguous_f32_serial`, preserving tie/first-index and NaN semantics. |
| `silu_tensor_contiguous_f32` (generated at `:30135`) | It is generated through `define_unary_f32!` and `unary_contiguous_f32` (`:29468`), which uses `window.par_iter()` at its parallel threshold. | `silu_tensor_contiguous_f32_serial`, or a clearly separate serial-unary primitive; it must not funnel back through `unary_contiguous_f32`. |
| `pairwise_sum_f32_maybe_par` (`:30540`) | The parallel implementation uses `rayon::join`. | Leaf reductions call only the existing serial `pairwise_sum_f32` (`:30527`) and expose deterministic partitions to their caller. |

`sdpa_forward_f32` is particularly important: a nominally serial-by-head choice
in the current public function still calls `par_chunks_mut`, and its GEMM
helpers are scheduling wrappers. It is not a no-spawn leaf as-is.

## Required upstream surface

The preferred implementation is a narrow, reviewed `ft-kernel-cpu-leaves`
crate (or equivalently isolated leaf module compiled as a separate package)
that depends on the needed numeric surfaces but has no `rayon` or
`rayon-core` dependency. Existing Rayon-backed `ft-kernel-cpu` entrypoints may
remain only in a non-release/dev-reference package or behind an explicit
`rayon-runtime` feature that is absent from the release package selection.

An alternative feature split inside `ft-kernel-cpu` is acceptable only if all
of the following are true:

1. `rayon-runtime = ["dep:rayon"]` is explicit and disabled for the leaf
   package selection;
2. every current Rayon import, scheduling wrapper, and Rayon-dependent helper
   above is excluded from that compilation unit; and
3. the serial/range API is compiled independently, rather than merely taking a
   runtime "serial" branch.

A dependency toggle alone, a global-pool size setting, or a wrapper around the
current public functions is rejected by this audit.

Every exposed leaf must meet this call contract:

```text
input ranges/tiles + caller-owned output/scratch
    -> finite arithmetic only
    -> returns before any caller-scheduled sibling begins a conflicting write
```

Specifically, it creates no OS thread, task, worker, pool job, Rayon scope, or
allocator-owned work queue. It chooses no parallel partition itself. The
asupersync scoped CPU team owns the split into rows, heads, and query tiles;
the leaf receives an already assigned range. `*_into`/range APIs are required
to make that ownership and lifetime boundary auditable and to avoid a hidden
allocation/scheduling wrapper.

The leaf implementation must preserve the current profile-specific arithmetic:
the int8 paths retain their scalar/SDOT/VNNI baselines and exact accumulation
rules, while the floating dense attention/softmax path retains the existing
scalar `libm` profile unless a separately named numerics decision changes it.
This pre-write does not authorize a polynomial-exp or other fidelity change.

## Pin, graph, and proof requirements

`franken_nlp-b2p` owns the lock tooling prerequisite. After it lands, this
bead's implementation must record the reviewed upstream revision in
`SUITE.lock` and the root patch closure (including the exact `block` revision
above) in `SUITE_ROOT_PATCHES.lock`. No floating branch or local unreceipted
checkout is consumable.

The selected release feature set must then provide all of the following
artifacts and checks:

| Artifact/check | Required result |
| --- | --- |
| Feature-matrix compile | Leaf surface builds with its release features; every Rayon-backed public entrypoint is unavailable or belongs to a non-release package. |
| `tests/no_spawn_leaves.rs` | Calls every exposed leaf under the thread-inventory watchdog, logs leaf name and before/after thread ids, and ends `NO_SPAWN RESULT=PASS leaves=<n> spawns=0`. |
| `tests/fixtures/release_graph.golden` | Diff-reviewed `cargo tree -e features` release-graph snapshot contains neither `rayon` nor `rayon-core`, including transitively. |
| Metadata graph walk | Independently rejects either Rayon node and prints the offending dependency path. |
| `scripts/check.sh` | Enforces the no-Rayon assertion; the dependency-policy known-blocked entry is deleted rather than suppressed. |
| Pin closure audit | `SUITE.lock`, `SUITE_ROOT_PATCHES.lock`, `Cargo.lock`, and `cargo metadata` agree at the selected revision and patch closure. |

No Cargo graph claim is made in this document: FrankenNLP has no crate scaffold
or `Cargo.toml` at the time of this audit, and the required lock tooling is
blocked on `franken_nlp-b2p`. This is a precise implementation handoff, not a
substitute for those gates.

## Implementation handoff

The future implementation must import only the newly reviewed no-Rayon leaf
surface. It must not call the present `ft-kernel-cpu` public wrappers listed
above in production inference. Existing Rayon code can remain useful as a
development or out-of-process reference baseline, but it must be
feature/package-partitioned so a release dependency walk cannot reach it.

The follow-on G0 seam proof owns the caller side: one asupersync
`spawn_blocking` crossing and a fixed `scoped_cpu` team form the work-sharing
region. This audit owns the complementary guarantee that the leaf cannot add a
second scheduler beneath that team.
