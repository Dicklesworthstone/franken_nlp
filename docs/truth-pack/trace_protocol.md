# Nanbeige reference trace protocol

`scripts/gen_reference_fixtures.py --trace` is the only Phase -1 producer of
the L2 trace fixture.  It runs inside the CPU-only, hash-locked oracle closure
and never changes the archived remote-code source.  The command has no meaning
until `franken_nlp-ilz` has recorded `docs/truth-pack/oracle_env.json`.

```bash
scripts/gen_reference_fixtures.py --trace \
  --model-source /owner-controlled/Nanbeige4.2-3B/f56ec5a9650268aa098496734743c25ea778bd2d \
  --profile hf-bf16-eager --output tests/fixtures/reference
```

The generator loads `NanbeigeForCausalLM` with `trust_remote_code`,
`use_fast=False`, CPU placement, inference mode, and the named attention
backend.  `hf-bf16-eager` is the HF-match authority; `diagnostic-f32` is the
structural bisect profile; `hf-bf16-sdpa` is explicitly variance-only.

## Hooked spans and non-perturbation rule

The trace collector resolves the model's decoder stack, embedding, final
RMSNorm, and lm head from the loaded object.  It installs forward pre/post
hooks on each of the 22 physical decoder modules; an invocation counter labels
the first call as loop 0 and the second as loop 1.  It also records optional
post-attention and post-MLP hooks when those child modules exist.

The 22 physical modules must each be invoked exactly twice.  The resulting 44
slots use `(loop, layer)`, and their KV index is `layer + loop * 22`.  The
collector separately records the two final-RMSNorm outputs and fails unless
the loop-1 norm tensor equals the captured loop-2 layer-0 input.  It captures
both the prefill cache and one-token append cache, each with all 44 KV slots.

Ordinary hooks only detach/copy after observing a tensor and always return
`None`; they never mutate, cast, or replace a live value.  After each captured
prompt the generator replays an untraced forward/greedy path and requires
bitwise-equal logits and the same greedy tokens.  `--trace-selftest` adds one
test-only hook that deliberately changes layer 0; its mismatch must be
detected before the command can report `PASS`.

## Serialized form

Each prompt/profile directory contains `trace.json` plus raw tensor sidecars.
`trace.json` tags every record with its phase (`prefill` or `append`), tap,
optional `(loop,layer)`, dtype, shape, element size, raw byte length, sidecar
path, and SHA-256.  It also binds profile, attention backend, variance-only
status, oracle-closure digest, generator commit, prompt digest, and pinned
model revision.  Raw floating tensor digests are fixture integrity records,
not cross-host portability claims.

The terminal line is machine-readable:

```text
TRACE_HARNESS RESULT=PASS|FAIL|SKIPPED_NO_MODEL taps=<n>/44 norms=<n>/2 perturbation=<none|first-diverging-layer-0>
```

Execution receipts, their exact command lines, and a no-perturbation run
receipt are added only after the locked oracle closure is available.  No
placeholder receipt is authority.
