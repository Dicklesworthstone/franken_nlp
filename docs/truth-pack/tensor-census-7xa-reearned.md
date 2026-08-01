# Tensor Census Re-Earn Transcript — 7xa

This is a retained code-first replay record for `franken_nlp-7xa`. It does not
close the bead and does not substitute for a controller-owned DSR checkpoint.

## Scope

- Model: `Nanbeige4.2-3B`
- Revision: `f56ec5a9650268aa098496734743c25ea778bd2d`
- Generator: `scripts/gen_tensor_census.py`
- Archived census: `docs/truth-pack/tensor_census.json`
- Source closure: `/Users/jemanuel/.cache/franken_nlp/source/Nanbeige4.2-3B/f56ec5a9650268aa098496734743c25ea778bd2d`

The source closure was read only. The real-index command authenticates both
`config.json` and `model.safetensors.index.json` before deriving shapes and
checking the 201 generated names.

## Commands

Run from the repository root on `2026-08-01T02:47:57+00:00`:

```text
python3 scripts/gen_tensor_census.py --self-test
python3 scripts/gen_tensor_census.py --check-artifact --output docs/truth-pack/tensor_census.json
python3 scripts/gen_tensor_census.py --check --index /Users/jemanuel/.cache/franken_nlp/source/Nanbeige4.2-3B/f56ec5a9650268aa098496734743c25ea778bd2d/model.safetensors.index.json --config /Users/jemanuel/.cache/franken_nlp/source/Nanbeige4.2-3B/f56ec5a9650268aa098496734743c25ea778bd2d/config.json --output docs/truth-pack/tensor_census.json
```

All three commands exited zero and emitted
`TENSOR_CENSUS RESULT=PASS missing=0 mismatched=0 extra=0`. The self-test
also exercised the OQ-1 `mhc_probe.weight` negative fixture, which raised the
required design-assumption error internally before the self-test reported its
pass result.

## Retained identities

| Input or artifact | Expected SHA-256 | Observed SHA-256 | Expected bytes | Observed bytes |
| --- | --- | --- | ---: | ---: |
| Generator source | `5c18ac04550527cdf00577c9c65e529bbff1f2ed3c4869a8e7631215f0053500` | `5c18ac04550527cdf00577c9c65e529bbff1f2ed3c4869a8e7631215f0053500` | n/a | n/a |
| Pinned `config.json` | `f6cb15b22847664f3a6049dc4b58fdd10f1650d112ac99a1da3d051f17c2ca19` | `f6cb15b22847664f3a6049dc4b58fdd10f1650d112ac99a1da3d051f17c2ca19` | 1,019 | 1,019 |
| Pinned safetensors index | `30d8da0fa8b97abc6d9eddfd017a0cb7a649bcbd58caa57804c56c767db5c0f1` | `30d8da0fa8b97abc6d9eddfd017a0cb7a649bcbd58caa57804c56c767db5c0f1` | 16,519 | 16,519 |
| Archived generated census | `ebe71a912723366e4def38cc8e558c0b09dba028a44bb7892872ff04d5d0f982` | `ebe71a912723366e4def38cc8e558c0b09dba028a44bb7892872ff04d5d0f982` | 34,795 | 34,795 |

`--check-artifact` compares the generated canonical bytes directly with the
archived bytes; a same-schema but differently formatted file also fails. The
real-index replay independently generated the same 34,795 bytes from the
authenticated configuration and found zero missing, shape-mismatched, or extra
index names.

## Drift-guard disposition

`scripts/check.sh` now runs the hermetic command below as mandatory section
`tensor-census`, and `.github/workflows/ci.yml` invokes that repository
entrypoint. `scripts/test_check.sh` retains a whitespace-only artifact-drift
case and requires the `tensor-census` section to fail before `cargo-check`:

```text
python3 scripts/gen_tensor_census.py --check-artifact
```

This ordinary CI guard never emits `SKIPPED_NO_MODEL`. The source-dependent
`--check` replay instead fails loudly if its authenticated pinned index is
absent, directing ordinary runs to the hermetic guard. No Cargo, DSR, or
GitHub Actions run is claimed by this transcript.

The negative read-only command below was also run on
`2026-08-01T02:48:48+00:00`; it exited nonzero with
`RESULT=FAIL`, rather than reporting a successful skip:

```text
python3 scripts/gen_tensor_census.py --check --index tests/fixtures/reference/absent-model.safetensors.index.json --output docs/truth-pack/tensor_census.json
```
