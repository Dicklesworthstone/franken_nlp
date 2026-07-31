# Fetch-model fixture transcript

This is the retained local-fixture evidence for `franken_nlp-mzr`. It exercises
the provisioning scripts only; it does not download model weights and is not an
inference or release-installer test.

## Unix fixture run

- Timestamp: `2026-07-31T10:00:47Z` through `2026-07-31T10:02:23Z`
- Host-local command: `sh -n scripts/fetch_model.sh && sh -n scripts/test_fetch_model.sh && scripts/test_fetch_model.sh`
- Exit code: `0`
- Retained fixture directory: `/var/folders/vt/n2xyn_s51b97_j3yh2qbqcnc0000gn/T/fnlp-fetch-model-test.Q71H8c`

The test creates a two-file catalog behind a loopback HTTP fixture and records
the detailed per-file logs, journals, partials, and quarantine entries in that
retained directory. Its transcript ended as follows:

```text
2026-07-31T10:01:17Z FETCH_MODEL_TEST CASE=1 RESULT=PASS detail=fresh-download
2026-07-31T10:01:39Z FETCH_MODEL_TEST CASE=2 RESULT=PASS detail=journal-resume
2026-07-31T10:01:50Z FETCH_MODEL_TEST CASE=3 RESULT=PASS detail=unbound-partial-quarantine
2026-07-31T10:01:50Z FETCH_MODEL_TEST CASE=4 RESULT=PASS detail=symlink-refusal
2026-07-31T10:02:01Z FETCH_MODEL_TEST CASE=5 RESULT=PASS detail=corrupt-final-quarantine
2026-07-31T10:02:12Z FETCH_MODEL_TEST CASE=6 RESULT=PASS detail=check-only-tamper
2026-07-31T10:02:12Z FETCH_MODEL_TEST CASE=7 RESULT=PASS detail=untrusted-revision-refusal
2026-07-31T10:02:22Z FETCH_MODEL_TEST CASE=8 RESULT=PASS detail=interrupted-resume-guidance
2026-07-31T10:02:23Z FETCH_MODEL_TEST CASE=9 RESULT=PASS detail=check-only-no-model-skip
2026-07-31T10:02:23Z FETCH_MODEL_TEST FETCH_MODEL_TESTS RESULT=PASS cases=9 failed=none
```

`CASE=2` also verifies the safe fallback for a fixture server that declines a
Range request: the journal-bound partial is quarantined and a fresh regular
partial is verified before activation. `CASE=8` verifies that an interrupted
resume prints the exact same `--dest` re-invocation and its journal directory.
`CASE=9` is the model-gated no-asset behavior: `--check-only` emits
`CHECK_ONLY RESULT=SKIPPED_NO_MODEL files=0/2`, rather than treating an absent
closure as a cache hit.

## Windows fixture run

`scripts/test_fetch_model.ps1` is committed with equivalent numbered loopback
fixture cases. This host had neither `pwsh` nor `powershell` on `PATH` at the
time of this transcript, so its execution is intentionally not claimed here.
It remains a required Windows Phase -1 validation before this bead can close.

## Real closure status

No real-network model closure was armed or downloaded in this run. The
model-gated result is therefore `SKIPPED_NO_NETWORK`; a later evidence run must
invoke both `fetch_model.sh --check-only` and `fetch_model.ps1 -CheckOnly`
against the clean revision-scoped ten-file closure and bind the observed values
to `nanbeige4.2-3b.source.json`.
