# Candidate finite-scoring judgment

`fnlp candidate judge` connects the existing native strict-INT8 judge to the
executable. The input is one bounded JSON object, with `mode` set to `pairwise`,
`rubric` or `faithfulness`. Every mode requires an explicit policy. The command
uses complete finite-candidate scoring with full-vocabulary denominators and
terminal EOS, not free generation followed by a JSON parser or label coercion.

```sh
fnlp candidate judge request.json --model ./model.fnlpq --memory-mib 8192
```

Omit the input path or use `-` for stdin. The binary needs `asupersync-runtime`
and a compatible explicitly selected local current-candidate artifact. The
command does not download or activate models, authenticate a publisher, or
promote artifact, numerical-fidelity or task-quality evidence.

## Pairwise comparison: both presentation orders

```json
{
  "mode": "pairwise",
  "criterion": "Prefer the answer that directly addresses the question.",
  "a": "The package is due on Tuesday.",
  "b": "Packages arrive on business days.",
  "policy": {
    "minimum_margin_milli": 0,
    "maximum_order_disagreement_milli": 1000
  }
}
```

The native planner compiles both A/B and B/A presentations. Both must complete;
the finalizer maps their log odds back to the original A/B identities before
combining them. Output preserves both complete score sets, order disagreement,
combined margin, candidate-conditional weight and prefer-A/prefer-B/tie/abstained
decision. Threshold units are thousandths of natural-log score ratios, not ppm
probabilities. The values above are an illustrative explicit policy, not a
calibrated recommendation. No failed order becomes a tie or abstention.

## Rubric: every independent criterion

Rubric requests contain `mode: "rubric"`, `document`, `rubric` and `policy`.
The rubric is the existing `RubricDefinition` shape:

* `schema_version`: 1; a bounded `revision` identifier;
* `declared_origin_digest`: the caller's 64-lowercase-hex SHA-256 declaration;
* `scale_maximum`: an integer from 1 through 10;
* `criteria`: one through 16 objects with `id`, `description` and `weight`.

Criterion IDs must be unique valid identifiers; descriptions must be nonempty.
Weights are integers in 1 through 1000000. The caller's origin digest and
revision are retained as declarations, not verified publisher provenance. The
CLI does not invent them or label a user rubric as a qualified shipped preset.

The explicit policy contains `minimum_peak_weight_ppm` and
`maximum_normalized_entropy_ppm`, each in 0 through 1000000. Every criterion is
scored independently over the complete integer scale 0 through `scale_maximum`.
The existing finalizer computes the weighted aggregation; it does not omit
failed or abstained criteria to manufacture an aggregate success. Results keep
all criterion distributions, weights, diagnostic means and decision disclosures.

## Faithfulness: whole source plus all evidence windows

```json
{
  "mode": "faithfulness",
  "source": "Alice moved to Paris. Bob stayed in Berlin.",
  "claim": "Alice moved to Paris.",
  "policy": {
    "minimum_candidate_weight_ppm": 0,
    "minimum_margin_milli": 0,
    "evidence_window_bytes": 128,
    "max_evidence_windows": 31,
    "max_evidence_spans": 31
  }
}
```

The source and claim remain exact UTF-8 data. The full-source head must fit the
admitted context; this command does not silently substitute retrieval, chunk
summaries or a truncated source. The evidence partition covers all source
bytes. It allows at most 31 windows plus the whole-source head. A source that
fits a single window reuses the whole-source head rather than scoring it twice.

`max_evidence_spans` is positive and cannot exceed `max_evidence_windows`.
Impossible or excessive partitions are rejected before opening model metadata.
The native finalizer retains uncertain and contrary windows, verifies reported
quote coordinates against the original source and can explicitly abstain. A
native failure is not the model relation `unsupported`. Source membership proves
that a quotation occurs; correlated model reads are not a factuality certificate.
All score distributions and judgments remain uncalibrated.

## Admission, delivery and limits

Shared finite-scoring options match `candidate classify` and `sentiment`:
2048 context tokens, 16 maximum candidate tokens including EOS, 65536 input
bytes and a 1 MiB complete native-result cap by default. Input is capped at
1 MiB; candidate provenance adds at most 4096 bytes to completed output.

All five computational-work limits cover the ENTIRE request across its heads:
`--max-forward-positions`, `--max-projected-logits`, `--max-attention-pairs`,
`--max-dot-products` and `--max-multiply-accumulates`. They retain the shared
scored-command defaults. Context is the largest live head, not the sum of all
forward work. Repeated prompt preparation remains covered by the existing
modeled preparation floor; the ledger is not an operating-system RSS limit.

Duplicate JSON keys, escaped duplicate aliases, unknown fields, invalid modes,
missing policies and invalid task data are refused before model metadata access.
Request JSON cannot supply a TaskBudget, tokens, templates, model identities,
work receipts, score-space overrides or tools. Source/criterion strings never
enter the trusted template renderer. The CLI supplies the finite task budget.

The same candidate Session reserves preparation before input or tokenizer
allocation. The pinned planner seals the complete request and checks all work
axes before weight loading. Loading compares the actual retained model facts;
execution uses the existing process host, resident model and native judge.
One failed comparison order, criterion or evidence head fails the whole request.

One cooperative elapsed budget follows input, preparation, loading and native
execution. Bounded compiler operations and blocking IO are not preempted midway.
Native and loader checkpoint allowances remain separately declared. Complete
JSON is staged before publication, and hosted output/preparation ownership
survives writing and flushing. A failed sink is not successful task delivery.
No provisional heads, private prompt fingerprints or nested parser diagnostics
are printed. Output retains `scope=real-artifact-current-candidate` and
`evidence=non_authoritative`.

This is a single-request command. Existing map, durable-job, source/extraction
batch and score-batch routes are preserved; judge batch is not added here.
Without `asupersync-runtime`, execution refuses before input or model IO.

## Validation scope

Sixteen added regression definitions cover strict parsing, policy units,
original text, rubric/evidence constraints, feature-disabled no-IO behavior,
all three real pinned planners, complete head counts, identity binding, context
reuse, every computational-work axis and cancellation. They are test sources,
not executed tests. Rust compilation/tests, full-model inference, numerical
parity, task quality, benchmarks and controller DSR were not run here.
