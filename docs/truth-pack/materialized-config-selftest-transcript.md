# Materialized-config self-test transcript

**Bead:** `franken_nlp-3pe`
**Generator:** `scripts/gen_materialized_config.py`
**Working directory:** `/Users/jemanuel/projects/franken_nlp`

The hermetic self-test exercises the required named-finding assertions and the
changed-raw-config negative fixture. The live `--check` was also invoked. It
was not given a verified local pinned source closure, so its declared
`SKIPPED_NO_MODEL` result is retained rather than being treated as successful
materialized-config regeneration evidence.

```text
$ python3 scripts/gen_materialized_config.py --self-test
2026-07-31T10:27:52+00:00 MATERIALIZED_CONFIG phase=self-test start
2026-07-31T10:27:52+00:00 MATERIALIZED_CONFIG RESULT=PASS fields=0

$ python3 scripts/gen_materialized_config.py --check
2026-07-31T10:27:57+00:00 MATERIALIZED_CONFIG RESULT=SKIPPED_NO_MODEL fields=0 detail=--source pinned closure not supplied
```

`SELFTEST PASS` covers only the hermetic generator assertions. A future
source-verified invocation must run `--check --source <verified-pinned-closure>`
and byte-compare the committed materialized record before this truth-pack item
can be closed.
