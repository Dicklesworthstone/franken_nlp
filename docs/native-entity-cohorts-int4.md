# Native INT4 entity discovery and resolution cohorts

`fnlp candidate source-int4 resolve` and `resolve-corpus` can group independent
NER discovery tasks and complete candidate pairs on the existing native
grouped-INT4 cohort engines. The two schedules are explicit and independent.
They retain the same original-source verification, both-order finite scoring
language, and complete-link clustering as serial entity resolution.

## Select the phases

| Option | Allowed width | Scope |
| --- | --- | --- |
| `--cohort-pairs N` | 1 through 32 complete pairs | Supplied mentions, raw-document discovery, and raw-document corpora |
| `--cohort-chunks N` | 1 through 64 source tasks | Requires `--discover-entities`; works with or without `--chunked` |

Both presentation orders of one pair always share a native window, so a
32-pair window has at most 64 independent scoring heads. A short final window
uses only its actual heads. Discovery follows canonical document/chunk order
and can put chunks from different original documents in the same window.
Without `--chunked`, each admitted complete document supplies one source task.

Omit either option to keep that phase serial. Omit both for the existing serial
route and result shape. Width one explicitly selects a cohort route; it is not
an alias for omitting the option. Zero and oversized widths reject before model
or document IO. These commands continue to reject `--cohort-heads`, prompt
prefill overrides, generation controls, and caller-supplied scheduling metadata
in source records.

## Resolve supplied mentions

Use the existing bounded JSON snapshot with original documents, source-anchored
mentions, and an explicit resolution policy:

```sh
fnlp candidate source-int4 resolve anchored-snapshot.json \
  --cohort-pairs 4 \
  --snapshot /models/nanbeige.group128.snapshot \
  --expected-model-sha256 "$TRUSTED_MODEL_CONTENT_SHA256" \
  --group-width 128 --memory-mib 32768 --context-tokens 2048
```

The complete lexical graph and both orders of every candidate pair must fit.
Windows change native scheduling only: they do not truncate the graph, change
the full-vocabulary score denominator, or allow a cluster to succeed with
missing pair evidence. All future pair windows preflight before the first
native pair allocation.

## Discover mentions and resolve the complete original snapshot

For the existing raw `{documents:[{id,text}],options,ner}` JSON shape:

```sh
fnlp candidate source-int4 resolve documents.json \
  --discover-entities --chunked --cohort-chunks 4 --cohort-pairs 2 \
  --snapshot /models/nanbeige.group128.snapshot \
  --expected-model-sha256 "$TRUSTED_MODEL_CONTENT_SHA256" \
  --group-width 128 --memory-mib 32768 --context-tokens 2048 \
  --max-input-bytes 1048576 --graph-reserve-mib 128
```

Remove `--chunked` when each complete document fits its NER context. Remove
`--cohort-pairs` to group only discovery, or remove `--cohort-chunks` to group
only the later pair phase. Source conversion uses the same schedules: replace
`--snapshot` and its independent digest pin with `--source-dir` and
`--source-manifest`.

Every original NER chunk is rebuilt and checked against its retained private
witness before the first native allocation. The shared collector verifies
returned occurrences against the original source and lifts chunk coordinates
to original byte and scalar offsets. Its metadata is bounded by document count
plus one native window; it does not copy or flatten all source witnesses.

The complete pair graph is planned only after discovery finishes. All planned
NER work remains reserved when computing the remaining pair allowance. Early
EOS, empty discoveries, and short windows grant no additional pair work.
Failure in any later window prevents a completed snapshot result.

## Keep one model resident across corpus epochs

Use the existing fixed NER/resolution policy file and raw-document NDJSON:

```sh
fnlp candidate source-int4 resolve-corpus documents.ndjson \
  --discover-entities --chunked --policy entity-policy.json \
  --cohort-chunks 4 --cohort-pairs 2 \
  --snapshot /models/nanbeige.group128.snapshot \
  --expected-model-sha256 "$TRUSTED_MODEL_CONTENT_SHA256" \
  --group-width 128 --memory-mib 32768 --context-tokens 2048 \
  --max-input-bytes 1048576 --graph-reserve-mib 128 \
  --max-documents 256 --max-records 1000 --max-snapshots 1000 \
  > resolved-epochs.ndjson
```

An explicit `{"flush":true}` completes the current epoch, including an empty
epoch. EOF completes a remaining nonempty epoch. The selected schedule and
physical reservation belong to one invocation; flush, empty epochs, and window
boundaries do not replenish native work, masks, or grounding allowances.
Physical counters accumulate alongside the existing lifetime ledger.

The model stays resident. Cohort windows allocate their required native
contexts, then drain before the dependent phase begins. The cohort route does
not keep an unused scalar engine beside its selected cohort phases. An
unselected nonempty phase still executes serially and drains before the next
phase. Empty snapshots and empty pair graphs allocate no unnecessary engine.

Resolution remains local to an epoch. A flush separates populations that will
not be compared, and cluster IDs have no durable corpus-global identity.
Publication retains the existing write-and-flush acknowledgement boundary;
arbitrary stdout delivery does not create durable resumable job state.

## Admission and work authority

The shared CLI/host quote covers retained original input, pinned planning,
graph and collection storage, additional simultaneous NER plans, native
contexts and scratch, bounded task staging, output, and cleanup. Successive
native phases use a componentwise peak because they drain in order. Additional
source-plan preparation scales with the selected NER width. Output owners stay
charged through validation, serialization, write, and flush.

The original task context and per-head KV ceiling never become the aggregate
cohort KV ceiling. Cohort storage has a separate physical allowance. All five
native work axes retain their whole-snapshot or whole-invocation ceilings,
including projected logits, attention, grouped dot products and multiply
accumulates. Source mask and grounding budgets retain their independent caps.

Example memory limits are declarations, not a promise that every policy or
population fits. Larger widths cost more simultaneous memory. A request that
does not fit rejects explicitly; the implementation does not silently narrow
its width or retry with a different numerical backend.

## Result shapes

Serial one-off responses retain their existing shape. With a selected cohort
schedule, the CLI's existing `output` contains a physical cohort wrapper and
that wrapper's `output` contains the unchanged semantic resolution or entity
result. Optional `cohort_pairs` and `cohort_chunks` provenance fields identify
the selected CLI widths without exposing source paths or private identities.

A supplied-mention wrapper reports pair cohort, group-step and projection-group
counts. A raw-entity wrapper reports its explicit schedule plus separate NER
and pair counters. A phase that remains serial has zero cohort counters.
These physical counters do not replace semantic work or its reserved limits.

Corpus epoch frames keep the existing semantic `output` and add an optional
sibling `cohort` object. The terminal summary includes accumulated cohort
counters for the fixed schedule. Serial frames omit these optional fields.
Only complete validated results are published, with no intermediate discovery
transcript or partial cluster result.

## Library entry points

`PreparedInt4Resolution::execute_cohorts_with_control` executes complete pair
windows. `Int4ResolutionCohortMemoryLimits::requirements` provides its pure
physical quote. `PreparedInt4EntityCorpus::execute_cohorts_with_control` uses
`Int4EntitySchedule` to select NER, pairs, or both; the schedule's `requirements`
method prices sequential phases and simultaneous source preparation.

The resource host exposes `NlpEngine::resolve_int4_cohorts` and
`NlpEngine::execute_int4_entities_cohorts`. Their corresponding
`Int4ResolveCohortHostRequirements::for_config` and
`Int4EntityCohortHostRequirements::for_config` functions share the CLI's
admission calculation. Resident corpus callers set
`Int4EntityCorpusHostConfig.schedule` and use
`Int4EntityCorpusHostRequirements::for_config_with_schedule`.

## Validation status and semantic limits

This is candidate implementation code. Focused regression definitions cover
schedule bounds, complete admission, source-coordinate parity, work conservation,
physical receipts, cancellation, publication and corpus flush behavior.
Actual-model fixtures include serial/cohort and scalar/SIMD comparisons.
These definitions are **UNRUN**; compilation, actual-model equality, quality,
and performance require the repository's controller-owned immutable-SHA
production validation and named model gates. No measured speedup is claimed.

Existing task limits remain: NER can omit or mistype mentions, chunk boundaries
can split entities, lexical blocking can miss aliases, and pair scores are
uncalibrated. Cohort execution does not establish identity accuracy or create
canonical model releases.
