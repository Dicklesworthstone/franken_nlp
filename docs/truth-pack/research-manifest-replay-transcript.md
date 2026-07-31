# Research-manifest replay transcript

The evidence closure was replayed on 2026-07-31 against the immutable files
under `docs/truth-pack/research/`. This documents the validator evidence only;
the research manifest remains outside converter and runtime inputs.

`--source-repo` is deliberately a stricter optional replay: it accepts a complete
pinned upstream repository tree, not `scripts/fetch_model.*`'s smaller conversion
closure. That closure omits research-only paths and retains fetch journals, so
passing it to the complete-census comparison must fail rather than weakening the
census to fit the conversion role.

```text
$ python3 scripts/validate_research_manifest.py --self-test
2026-07-31T13:18:13+00:00 RESEARCH_MANIFEST self_test verdict=PASS checks=6
RESEARCH_MANIFEST RESULT=PASS files=0 mismatches=

$ python3 scripts/validate_research_manifest.py \
    --manifest docs/truth-pack/nanbeige4.2-3b.research.json \
    --archive-root docs/truth-pack \
    --conversion-manifest docs/truth-pack/nanbeige4.2-3b.source.json
2026-07-31T13:18:13+00:00 RESEARCH_MANIFEST separation verdict=PASS shared_revision=true file_overlap=0
RESEARCH_MANIFEST RESULT=PASS files=12 mismatches=
```

## Current replay

The current archive-only replay repeated the same separation and digest gate
without treating the narrower conversion cache as a complete upstream
repository checkout.

```text
$ python3 scripts/validate_research_manifest.py --self-test
2026-07-31T15:51:32+00:00 RESEARCH_MANIFEST self_test verdict=PASS checks=6
RESEARCH_MANIFEST RESULT=PASS files=0 mismatches=

$ python3 scripts/validate_research_manifest.py \
    --manifest docs/truth-pack/nanbeige4.2-3b.research.json \
    --archive-root docs/truth-pack \
    --conversion-manifest docs/truth-pack/nanbeige4.2-3b.source.json
2026-07-31T15:51:51+00:00 RESEARCH_MANIFEST separation verdict=PASS shared_revision=true file_overlap=0
RESEARCH_MANIFEST RESULT=PASS files=12 mismatches=
```
