<!-- fnlp-ledger-schema: negative-evidence/v1 -->

## NE-FIXTURE-MISSING-CPU-001

- Claim ID: none
- Evidence: exact scalar/i64 comparison fixture
- Fixture hashes: sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
- Command + environment: `fnlp bench`; threads=1; allocator=system; profile=diagnostic-f32
- Disposition: rejected
- Hypothesis: candidate should improve the measured kernel.
- Five-pass loop: baseline, one lever, parity, thermal A/B, revert.
- Loss basis: slower than baseline.
- Revert proof: no source landed; candidate remains out of tree.
- Re-evaluation conditions: repeat only with a new non-saturating algorithm.
