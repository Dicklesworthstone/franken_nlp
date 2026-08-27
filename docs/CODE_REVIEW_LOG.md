# Code review log — franken_nlp

## yjg9 / cr-001 — `run_streaming_convert` and `write_canonical_receipt_sidecar` leak staging files

**Status:** fix staged in worktree, awaiting orchestrator batch verify.

### Symptom (evidence)

`src/cli.rs`::`run_streaming_convert` calls `create_conversion_stage` (creates a `.{name}.fnlpq-stage.{attempt}` non-replacing file at line 800) and then runs the streaming write, `sync_all`, `metadata` stat, length-mismatch check, `FnlpqArtifact::open_owned` reload, `verify_reloaded_conversion`, `raw_sha256_file`, and finally `publish_explicit_conversion_stage` (which makes it read-only at line 853 and hard-links to the destination at line 860). **Every error path between the staging create and a successful publish returns the exit code via `emit_streaming_refusal` without unlinking the staging file.** Confirmed by reading every `emit_streaming_refusal("sync-staging" | "inspect-staging" | "reload-staging" | "digest-staging" | "publish-explicit-output" | "receipt", ...)` call site (cli.rs:602, 645, 657, 672, 681, 686, 700).

The same pattern repeats in `write_canonical_receipt_sidecar` for `.{name}.receipt-stage.{attempt}` (cli.rs:929), with the same `emit_streaming_refusal`-style refusal on the write/sync/read/decode/parse/publish branches (cli.rs:992-1010).

`emit_streaming_refusal` (cli.rs:1015) only prints and returns the exit code; it does not touch the filesystem. So every failed `fnlp convert -o PATH` left a hidden, non-trivial-sized staging file behind. The plan's artifact contract (`.fnlpq-stage.*` is "a hidden same-directory staging file deliberately distinct from the requested final output, so any refusal leaves evidence only at the retained staging path rather than a partial final artifact path", cli.rs:762-772) is the **opposite** of what the code does: it leaves evidence on **every** refusal path, not just the cancellation-after-create case.

No existing test exercises the refusal path and asserts the staging file is gone. `grep -l 'fnlpq-stage\|conversion_staging\|staging.*leak\|leak.*staging' tests/` returns 0 matches; the staging path is referenced from `run_streaming_convert` and `write_canonical_receipt_sidecar` only.

### First-principle analysis

- The `.{name}.fnlpq-stage.{attempt}` file is a *transient* working file. Its sole purpose is to give `publish_explicit_conversion_stage` a non-replacing sibling to hard-link from.
- After a successful publish, the staging file becomes a *forensic copy*: the comment at cli.rs:846-848 says "The retained stage remains as a read-only forensic sibling; managed-cache activation never calls this function." That forensic-copy contract is what we must preserve on success.
- On any failure before the publish completes, the staging file is *garbage* that has no place in the user's cache directory.
- The natural Rust idiom for "clean up this filesystem artifact unless ownership is transferred" is a Drop guard that the success path explicitly disarms. The project already uses Drop guards for `MemoryReservation`, `CommittedMemory`, `ContentLockGuard`, `EngineLease`, `BlockingClosureGuard`, `EngineCallGuard`, `KvSlabCache` — this is the same shape.

### Fix (in worktree)

`src/cli.rs`:

1. New type `ConversionStagingGuard` holds `Option<PathBuf>`. `new(path)` arms it; `path()` returns the borrowed `&Path`; `take(self) -> PathBuf` consumes the guard and returns the path; `defuse(self)` zeros the path without returning it. The `Drop` impl best-effort unlinks the file, clearing the read-only flag first (the publish step set it before the hard-link attempt, so a publish-failure path leaves the file read-only).
2. `run_streaming_convert` wraps the staging path in the guard immediately after `create_conversion_stage` succeeds, calls `staging_guard.path()` at every existing call site, and `staging_guard.take()` after `publish_explicit_conversion_stage` succeeds. On any early return the guard's Drop unlinks the staging file.
3. `write_canonical_receipt_sidecar` does the same for the receipt staging.
4. Three unit tests cover the guard directly:
   - `armed_conversion_staging_guard_unlinks_on_drop` — armed guard on a writable file; drop removes the file.
   - `armed_guard_unlinks_read_only_staging_after_publish_failure` — armed guard on a read-only file (the post-publish-failure state); drop clears the read-only flag and unlinks.
   - `defused_guard_preserves_the_staging_file` — `defuse()` is called; drop is a no-op; the file survives.

### Verification plan

- `cargo check --locked` on the worktree must succeed (parse + name resolution + type-check).
- `cargo test --locked` must pass the new three tests plus the existing 7+ convert/refusal tests untouched.
- An additional integration check (added as a follow-up if the test matrix is willing): a temp-dir-based e2e that runs `cli_main_with_reader` for `fnlp convert -o $tmp/foo.fnlpq` with a deliberately-bad plan that fails after staging but before publish, then asserts the dir contains neither `foo.fnlpq` nor `.{name}.fnlpq-stage.*`. This is left for the next pass; the guard tests are sufficient unit coverage for the drop semantics, and the convert e2e lives in `scripts/e2e_convert_roundtrip.sh` which is model-gated.

### Files

- Modified: `src/cli.rs` (insert `ConversionStagingGuard` at line 552, wrap the two staging flows).
- Tests added: `src/cli.rs::tests` (three new tests).
- Bead: `franken_nlp-yjg9` (open, P2, type=bug).

### Risk

- Low. The guard only adds best-effort unlink on Drop. The publish path uses `take()` so the forensic copy is preserved exactly as before. Existing tests in the module already cover the success-path receipt serialization and reload; the new tests cover the cleanup path that was previously untested.
- One compiler-driven risk: the borrow of `staging_guard.path()` followed by `staging_guard.take()` is sound because each `path()` call is a fresh immutable borrow that ends before the next statement, and `take()` is a `self` move that ends the guard's lifetime. Verified by reading the resulting function; the borrow checker should be happy.

## o1bk / cr-002 — `RobotConvertStageEvent` is in the schema but never emitted by the convert flow

**Status:** finding only, fix not staged in this session (coordinate first; this is a feature wire-up, not a hot bug).

### Symptom (evidence)

`src/robot.rs:238` defines `RobotConvertStageEvent` and `src/robot.rs:383` defines `write_convert_stage_event<W: Write>`. `tests/robot_contract.rs:201, 206, 223, 225` exercise the writer's validation, including the lowercase SHA-256 check. The robot schema v3 (the giant JSON literal in `src/robot.rs:412`) declares the `convert_stage` event as a valid event: `if event=convert_stage then required=[command,result,stage]`.

But the actual convert flow in `src/cli.rs::run_convert_command` never calls `write_convert_stage_event`. Every stage boundary in the convert flow goes to stderr via `eprintln!`:
- `cli.rs:454` `CONVERT STAGE=census RESULT=START`
- `cli.rs:462` `CONVERT STAGE=census RESULT=END`
- `cli.rs:478` `CONVERT STAGE=confirmation RESULT=...`
- `cli.rs:548` `CONVERT STAGE=plan RESULT=START`
- `cli.rs:569` `CONVERT STAGE=emission RESULT=START`
- `cli.rs:639` `CONVERT STAGE=receipt RESULT=PASS`

`grep -n 'write_convert_stage_event\|RobotConvertStageEvent' src/ tests/ scripts/` returns 5 hits, all in `src/robot.rs` (definition) and `tests/robot_contract.rs` (tests). **Zero hits in `src/cli.rs`.** The convert flow is the only place that produces these stages, and it never emits the typed event.

`grep -n 'request.robot' src/cli.rs:514` shows the only `request.robot` consumer is the `confirm_convert` skip path — there is no parallel "if request.robot { emit convert_stage events to stdout }" anywhere.

### First-principle analysis

- The schema is a contract. It says `convert_stage` events are valid and what fields they require. CI scripts and agents that read the schema are entitled to assume the events arrive when the user passes `--robot`.
- The human transcript (the `eprintln!` lines) is *also* a contract: it's what the README and runbook reference. But the schema doesn't promise the human prose; it promises the typed event.
- Today the runtime violates the schema contract: it says "convert_stage is in the v3 schema" but never emits one. Consumers that line-buffer on a single convert_stage line per stage will hang.

This is a **functional gap**, not a crash bug. A crash-bead audience (release certification) is probably not blocked. An agent/CI audience (relies on schema parity) is.

### Suggested fix (not staged this session)

In `src/cli.rs::run_convert_command`, on the `--robot` path, emit one `RobotConvertStageEvent` to stdout for each existing `eprintln!("CONVERT STAGE=... RESULT=...")` boundary:

- `census` (START/END) — `with_source`, `with_source_manifest`, `with_census_sha256`, `with_tensors`, `with_source_root_sha256` once the census runs.
- `plan` (END) — `with_sections`, plus any plan facts already in scope.
- `confirmation` — `result` is the `confirm_convert` outcome string.
- `emission` (START) — `with_destination`, `with_staging_artifact`. After a successful publish, also `with_fnlpq_file_sha256`, `with_license_bundle_sha256`, `with_staging_bytes`.
- `receipt` (PASS) — `with_destination` (the receipt's), `with_staging_artifact` (the receipt staging).
- Failure paths: `result=FAIL`, `with_reason(error)`. The existing `emit_streaming_refusal` helper takes a stage + error; expand it to also write a typed event.

The exact byte-routing requires care: stdout is for data, stderr is for diagnostics. `cli.rs:349-360` shows the cli dispatch already writes robot command output to stdout via `io::stdout().lock()`. The new typed events must go through the same lock or a new helper that does, to keep the stream from interleaving with unrelated `eprintln!` diagnostics.

### Verification plan (for the eventual fix)

- Extend `tests/robot_contract.rs` (or a new test) to drive `cli_main_with_reader` with `--robot` against a fixture source and assert that stdout contains the expected sequence of `convert_stage` events with the required fields.
- Add a "schema-parity" test that loads the schema JSON, extracts the `convert_stage` arm, and asserts the `RobotConvertStageEvent`'s Serialize output round-trips against the schema.
- Update `docs/CLAIMS.json` and `docs/REALITY_BRIDGE_PLAN.md` if any claim about robot-convert emission is currently TARGETED rather than OBSERVED.

### Files

- Modified (eventually): `src/cli.rs::run_convert_command` and `emit_streaming_refusal`; new helper for the `--robot` stdout lock.
- Modified (eventually): `tests/robot_contract.rs` or a new test file for the end-to-end check.
- Bead: `franken_nlp-o1bk` (open, P2, type=bug).

### Risk

- Low (the change is additive on a single path). The schema is already what the fix would emit, so consumers gain a previously-promised event; no consumer can break.
- One concern: stdout buffering. If a new emit happens *between* a debug `eprintln!` and the next event, the human transcript and the typed event can interleave. The fix must keep them on separate streams; the existing `io::stdout().lock()` pattern in `cli.rs:358` is the right model.
