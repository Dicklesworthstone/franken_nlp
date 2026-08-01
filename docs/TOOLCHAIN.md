# Toolchain Pin and Compiler Identity

FrankenNLP is pinned to `nightly-2026-07-20`, never floating `nightly`.
`rust-toolchain.toml` is the installation authority and
`ci/toolchain.expected.json` is the comparison fixture consumed by the
controller-authorized DSR checkpoint. A toolchain change is a versioned
evidence event: update both files, explain the affected proof re-runs, and
retain the DSR observation of `rustc -Vv`, LLVM, the build target, enabled
target features, and compiler feature catalogue.

Ordinary swarm panes run no Cargo, RCH, DSR, or GitHub Actions command. After
code-first contention quiesces, the controller selects one clean immutable SHA;
only an occasional DSR job running `scripts/check.sh` against the explicitly
named `production` graph can create build evidence. Direct RCH and GitHub
Actions results are non-authoritative. The receipt binds this toolchain fixture,
the clean SHA, exact graph, DSR recipe/version, host/target, and literal
`PASS|FAIL`. Until that receipt exists, toolchain/build proof is `BLOCKED`.

The selected dated nightly reports `rustc 1.99.0-nightly`, commit
`9f36de775bc636c8e88c31a173c2bcb6995956a0` (2026-07-19), and LLVM 22.1.8.
It exposes AArch64 `dotprod`, `i8mm`, and `neon` in the target-feature
catalogue, which is the required compiler capability for the later stdarch
kernel islands.  The initial cross-target scaffold checks and compile probes
remain DSR batch-verification work; that pending state is recorded explicitly
in the expectation file rather than implied as completed evidence. Model-run,
target-host performance/platform, and human-review gates remain separate from
this compiler identity record.
