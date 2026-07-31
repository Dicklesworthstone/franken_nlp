# Toolchain Pin and Compiler Identity

FrankenNLP is pinned to `nightly-2026-07-20`, never floating `nightly`.
`rust-toolchain.toml` is the installation authority and
`ci/toolchain.expected.json` is the CI comparison authority.  A toolchain
change is a versioned evidence event: update both files, explain the affected
proof re-runs, and retain the CI observation of `rustc -Vv`, LLVM, the build
target, enabled target features, and compiler feature catalogue.

The selected dated nightly reports `rustc 1.99.0-nightly`, commit
`9f36de775bc636c8e88c31a173c2bcb6995956a0` (2026-07-19), and LLVM 22.1.8.
It exposes AArch64 `dotprod`, `i8mm`, and `neon` in the target-feature
catalogue, which is the required compiler capability for the later stdarch
kernel islands.  The initial cross-target scaffold checks and compile probes
remain batch-verification work; that pending state is recorded explicitly in
the expectation file rather than implied as completed evidence.
