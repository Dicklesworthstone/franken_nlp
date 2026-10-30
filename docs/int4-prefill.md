# Explicit grouped-INT4 prompt prefill

`fnlp candidate source-int4 generate`, `chat`, `classify`, `sentiment` and
`judge` accept `--prefill-rows N` for bounded layer-major prompt execution.
Scored tasks group each independent head's prompt. The value is 1 through
64. Without this flag the existing sequential path is unchanged. This is an
explicit non-certified candidate, not automatic dispatch, a benchmark result,
a numerical qualification, or an INT8 fallback.

## CLI

```sh
fnlp candidate source-int4 generate prompt.txt \
  --source-dir /models/nanbeige-source \
  --source-manifest /models/pinned-source-manifest.json \
  --memory-mib 8192 --group-width 128 --prefill-rows 4

fnlp candidate source-int4 chat conversation.json \
  --snapshot /models/nanbeige-g128.int4snap \
  --expected-model-sha256 "$MODEL_CONTENT_SHA256_FROM_TRUSTED_CONVERSION" \
  --memory-mib 8192 --group-width 128 --prefill-rows 4 \
  --kernel x3b_avx2_widened_i16
```

These commands require the `asupersync-runtime` feature. The second example
requires the named AVX2 implementation on the actual host. Kernel selection
and prompt scheduling are independent; scalar remains the kernel default and
unsupported explicit kernels fail rather than fall back. The example memory
ceiling is a ledger allowance, not measured RSS.

Source verification, independent snapshot pins, optional create-new snapshot
saving, tokenizer/template binding, input validation and the completed JSON
envelope remain unchanged. A snapshot failure never starts source conversion.
Prefill does not add a stream, a multi-request batch command or a new runtime.
One-off classification, sentiment and judgment use the same explicit scheduling
option with complete task and native work admission. Scored corpus/job commands,
resolution, extraction and source tasks still reject it, including
programmatically constructed requests. There is no silently ignored option.

For an existing digest-pinned INT4 snapshot and a typed classification request:

```sh
fnlp candidate source-int4 classify request.json \
  --snapshot /models/nanbeige-g128.int4snap \
  --expected-model-sha256 "$MODEL_CONTENT_SHA256_FROM_TRUSTED_CONVERSION" \
  --memory-mib 8192 --group-width 128 --prefill-rows 4
```

The `sentiment` and `judge` commands accept the same option with their existing
typed request files. Their complete multi-head preparation remains independently
charged; every head must pass admission before the first model forward.

## Native execution and numerical contract

The native `Int4Session::append_layer_major` method validates the entire exact
prompt, remaining context, work ceilings and extra-workspace limit before
mutating state. Long prompts are divided into bounded morsels. Each morsel
runs the existing `LoopRunner` through both 22-layer passes and both final
norms, preserving all 44 independent logical KV slots. A complete prefix may
be extended without reinterpreting its position coordinates.

Grouped projection visits retain low-nibble-first signed INT4 storage,
per-row activation scales, each matrix's group partition and unpadded tails.
Up to four activation rows share each packed-group integer reduction. Every
lane dequantizes and accumulates groups in the original order, then casts to
BF16 once. Weights are not expanded into a dense INT8 or floating matrix.

Attention appends one token's key/value row and immediately attends to that
exact prefix. It does not append a whole morsel before attention; earlier rows
cannot see future tokens. Post-prompt decoding remains token-by-token. Only
the final prompt position runs the language-model head, never intermediate
morsel boundaries. `prefill_layer_major` also supports the existing selected
head-row contract for native callers.

The generation adapter reuses the existing cursor, sampler addressing, stop
and penalty policy. Prompt positions are reconciled without rerunning the
model or consuming random draws. Chat reuses the pinned tokenizer and the same
independent result finalizer. Model fingerprints, semantic execution identity,
profile and completed result fields do not change with prompt scheduling.
Algorithmic group-dot/MAC/attention counts still describe the same work, not a
claim about retired instructions, memory traffic or speed.

Finite-candidate scoring groups only the initial prompt of each head. The shared
candidate scorer still owns every continuation, terminal EOS edge, probability
denominator and trie branch. Backtracking retains one live KV branch, and heads
clear their KV state before the next head. Classifications, independent sentiment
axes, both pairwise judgment orders, rubric criteria and faithfulness evidence
retain their original finalizers and exact whole-bundle work allowances. The
same completed result contract applies to serial and grouped execution.

## Memory and cancellation

`Int4PrefillLimits::required_extra_scratch_bytes(rows)` is the authoritative
additional payload bound: `rows * (32 * intermediate_size * sizeof(f32) + 4096)`.
It prices the finite prompt-row rails, activations and descriptors separately
from base native scratch, KV capacity, RoPE, sampler and allocator allowances.
The engine's one existing group-unpacking buffer is not duplicated per row.
The bound is conservative, not a peak-RSS measurement.

The CLI adds this bound only to the execution stage, then takes the maximum of
loading, optional saving and execution storage. It does not charge all three
stages as though they coexist. The host independently reserves the additional
payload in `ActivationScratch` before any native allocation. That reservation,
the model and all other native guards remain owned until physical completion.
The hosted result and CLI preparation guards remain live during publication.

The whole prompt is preflighted, and complete projection work is charged before
its first output write. Cooperative cancellation is polled between bounded
native operations; failed work is not refunded or automatically retried.
Failures after mutation poison the normal session; RAII cleanup clears every
KV slot. No token event is emitted by the new generation adapter until prompt
prefill and its final head complete. Library stream events remain provisional
until the full task finalizer succeeds; the CLI still emits one completed JSON
object, not provisional events.

## Embedding APIs

The existing sequential APIs remain available with their original signatures.
The explicit alternatives are:

- `Int4Session::append_layer_major` / `prefill_layer_major`.
- `Int4GenerationPlan::execute_layer_major` / `execute_layer_major_with_sink`.
- `PreparedInt4Chat::execute_layer_major` / `execute_layer_major_with_sink`.
- `NlpEngine::execute_int4_chat_layer_major` for charged hosted execution.
- `Int4CandidatePlan::execute_layer_major` for exact finite-language scoring.
- `PreparedInt4Classification`, `PreparedInt4Sentiment` and `PreparedInt4Judge`
  expose `execute_layer_major_with_control`.
- `NlpEngine::execute_int4_classify_layer_major`,
  `execute_int4_sentiment_layer_major` and `execute_int4_judge_layer_major`
  reserve the extra scratch before entering the existing blocking invocation.

Lower-level limits describe payload ceilings; they are not a substitute for a
resource-domain reservation. The hosted method shares one implementation with
sequential execution, including actual model/domain checks, native work bounds,
output admission and physical cleanup. It cannot import uncharged weights or
mint a successful task result from synthetic model output.

## Validation status

Regression source covers packed row/scalar equivalence across group widths,
odd reductions, tail groups and available explicit kernels; causal attention
against token-at-a-time replay; work and memory admission; cancellation and
cleanup; cursor output/score/policy reconciliation; and CLI source/snapshot
modes and stale scheduling admission. Synthetic logits exercise the real
cursor, not model quality or full-model numerical equivalence.

The ignored test
`model_present_snapshot_text_prefill_matches_sequential_complete_results`
compares the complete CLI response for sequential and 1/4/64-row prefill,
separately for generation and multi-turn chat, using seeded decoding and
log-probabilities. It requires controller admission and these explicit inputs:
`FNLP_INT4_PREFILL_SNAPSHOT` (an existing group-width-128 snapshot),
`FNLP_INT4_PREFILL_MODEL_SHA256` (its independently retained model-content pin),
`FNLP_INT4_TEST_MEMORY_MIB`, and `FNLP_INT4_TEST_KERNEL`. It only reads that
snapshot; it never saves, replaces, deletes or downloads model files.

Rust compilation/tests and this full-model test have not been run during this
implementation: the repository-required RCH runner is unavailable in the working
environment. Independent Python packed-group and memory/work arithmetic checks
and exact source-blob verification do not establish Rust correctness, numerical
qualification or measured performance. Keep the path opt-in pending those gates.
