# Reference fixture format and verification contract

`scripts/gen_reference_fixtures.py --generate` materializes the complete
profile-tagged Phase -1 comparison corpus.  It is model-gated: a missing source
closure reports `SKIPPED_NO_MODEL`; it never downloads a model.  Generation
requires the immutable `oracle_env.json` closure record and the measured
`oracle_floor.json`; a fixture cannot freeze a greedy stream before the floor
has declared its stable prefix.

```bash
scripts/gen_reference_fixtures.py --generate --profile all \
  --model-source /owner-controlled/Nanbeige4.2-3B/f56ec5a9650268aa098496734743c25ea778bd2d \
  --corpus tests/fixtures/reference_inputs.json \
  --oracle-floor docs/truth-pack/oracle_floor.json \
  --output tests/fixtures/reference

scripts/gen_reference_fixtures.py --verify \
  --output tests/fixtures/reference \
  --oracle-floor docs/truth-pack/oracle_floor.json
```

## Matrix and authority

The generator writes `hf-bf16-eager`, `diagnostic-f32`, and
`hf-bf16-sdpa`.  Every trace index names its profile, dtype, and attention
backend.  Eager bf16 owns HF-match claims; diagnostic f32 is a structural
bisect surface only; SDPA carries `variance_only: true` and cannot redefine
eager semantics.

The repository-authored input corpus covers ordinary/multilingual/code/marker
tokenization and the chat-template mode matrix: default system text, thinking
off, thinking preserved, and tool JSON.  Adding a case is a versioned corpus
change: it changes the corpus digest and requires a new fixture generation.
No third-party licensed prose belongs in this input file.

## Layout

```text
tests/fixtures/reference/
  manifest.json
  auxiliary.json
  <profile>/prompt-<n>/trace.json
  <profile>/prompt-<n>/tensors/*.bin
```

`trace.json` contains every L2 item: post-embedding, all 44 post-layer states,
both post-loop norms, pre-lm-head state, logits, optional post-attention/MLP
bisect points, and prefill plus one-token-append KV values for all 44 slots.
Each tensor descriptor records exact CPU-storage bytes, dtype, shape, element
size, byte length, relative sidecar path, and SHA-256.  The index binds the
pinned revision, profile/backend, oracle closure digest, generator commit, and
prompt digest.

The `greedy_contract` contains the SHA-256 of `oracle_floor.json` and the
stable prefix length for that prompt.  `oracle_floor.json` provides a
`stable_prefixes` object keyed by the prompt SHA-256, with an integer prefix
length.  A fixture with an unbound or mismatched floor digest is rejected; it
cannot be presented as an exact golden.

`manifest.json` hashes every trace index and `auxiliary.json`.  The verifier
rejects an unknown profile tag, unexpected attention backend, stale digest,
unsafe/duplicate sidecar path, malformed shape/byte length, incomplete
44-slot/2-norm/KV inventory, missing frozen prefix, or a variance file without
its explicit tag.  It has no model dependency and ends with:

```text
REF_FIXTURES RESULT=PASS|FAIL|SKIPPED_NO_MODEL fixtures=<n> missing=<list>
```

`--self-test` is also model-free.  It proves raw-sidecar tampering and an
unknown profile tag are rejected.  The execution receipts and generated binary
fixtures remain blocked on the locked oracle closure and are not fabricated by
this pre-write.
