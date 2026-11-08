# Experimental INT4 summary and answer review

`fnlp candidate source-int4 summarize-semantic` and `answer-semantic` generate
the existing cited source-task result and then assess its complete statements
with the native faithfulness judge. One resident model and one native engine
perform both stages. These commands require `asupersync-runtime` and explicit
experimental opt-in through a complete `--semantic-options FILE` contract.

## What the result means

The combined result retains the original source task, including generated
tokens, text, citations, byte and Unicode-scalar offsets, work counts, and its
existing `semantic_support: "not_assessed"` field. A separate semantic receipt
reports the second reader's assessment:

| Generated result | Statement sent to the judge | Evidence |
| --- | --- | --- |
| Cited summary | Each complete bullet's exact text, in original order | The complete original document |
| Answered question | A versioned statement containing the exact question and answer as separately escaped canonical JSON fields | All original passages joined in their original order |
| Abstained answer | No claim; explicit `not_checked` with `answer_abstained` | No judge execution |
| Empty summary | No claim; explicit `not_checked` with `empty_summary` | No judge execution |

Each attempted statement reports `entailed`, `contradicted`, `unsupported`, or
`judge_abstained`, with the full faithfulness result. Judge abstention and a
source task that generated no claim remain distinct. An empty summary or
abstained answer never becomes an implicitly supported statement.

The same model supplies generation and review, so the receipt explicitly marks
`same_model_correlated: true` and
`calibration: "experimental_uncalibrated_diagnostic_only"`. These assessments
have no acceptance or repair authority. They do not establish independent
truth, calibrated confidence, atomic-claim coverage, or that a displayed
citation alone entails its associated prose. The evidence scope is the
**complete original source**, including all passages for QA. Original citation
validation continues to establish structural source membership.

For QA, the question is claim data and cannot become evidence. Passage IDs and
their byte/scalar ranges are included in the review's evidence layout; joining
bytes are not attributed to a passage. The exact original request, including
question, passage IDs/order, task options and budget, must match the sealed
source plan. A caller cannot attach different evidence to a prepared task.

## CLI usage

Summarization reads exact UTF-8 document text:

```bash
fnlp candidate source-int4 summarize-semantic document.txt \
  --snapshot nanbeige.int4.snapshot \
  --expected-model-sha256 "$EXPECTED_MODEL_SHA256" \
  --memory-mib 8192 --context-tokens 2048 \
  --semantic-options source-semantic.json
```

QA uses the existing question/passages input format:

```json
{"question":"Where does Alice work?","passages":[{"id":"p1","text":"Alice works in Shanghai."}]}
```

```bash
fnlp candidate source-int4 answer-semantic question.json \
  --source-dir ./pinned-source \
  --source-manifest ./conversion-source.json \
  --memory-mib 8192 --context-tokens 2048 \
  --semantic-options source-semantic.json
```

Both commands accept the existing task-specific `--options FILE`. Input may
come from `-`; configuration files must be bounded local files. The snapshot
and source-conversion routes use the same composition. The initial review
route uses sequential full-vocabulary execution and byte-fallback input.
Compute teams, packed prefill, selected output rows and Safe-BPE are not
enabled on these composed commands. Ordinary source tasks, batches, maps,
synthesis and jobs retain their own interfaces.

## Complete review options

`source-semantic.json` is a complete
`tasks::source_planning::grouped_int4::semantic::Int4SourceSemanticOptions`
object. Duplicate keys, unknown fields, partial options and missing explicit
opt-in are refused. This example accommodates the default eight-bullet summary
ceiling and a 2,048-token resident context:

```json
{
  "schema_version": 1,
  "experimental_opt_in": true,
  "judge_policy": {
    "minimum_candidate_weight_ppm": 0,
    "minimum_margin_milli": 0,
    "evidence_window_bytes": 8192,
    "max_evidence_windows": 8,
    "max_evidence_spans": 8
  },
  "judge_budget": {
    "max_input_tokens": 1985,
    "max_output_tokens": 64,
    "max_output_bytes": 65536,
    "max_grammar_states": 4096,
    "max_kv_bytes": 369098752
  },
  "limits": {
    "max_statements": 8,
    "max_source_bytes": 1048576,
    "max_claim_bytes": 16384,
    "max_total_claim_bytes": 131072,
    "max_total_prompt_tokens": 16384,
    "max_total_projected_logits": 100000000,
    "max_output_bytes": 4194304
  },
  "max_model_work": {
    "forward_positions": 100000,
    "projected_logits": 200000000,
    "attention_pairs": 1000000000000,
    "projections": {
      "dot_products": 10000000000000,
      "multiply_accumulates": 1000000000000000
    }
  }
}
```

These values are explicit resource ceilings and an uncalibrated example
policy. They are not measured performance or recommended acceptance
thresholds. A byte allowance does not enlarge the model context: the complete
source, claim, trusted prompt and every required evidence window must also fit
their token and work budgets. Oversized evidence is refused without
truncation or selective retrieval.

`--max-result-bytes` bounds the original source-task result.
`limits.max_output_bytes` bounds the complete combined result. A summary's
`max_bullets` must fit `limits.max_statements`; QA's exact rendered question and
answer must fit both claim-byte ceilings. The judge's KV budget covers the
complete configured resident context, not a smaller per-head allocation.

## Execution and resource contract

All statement judges are compiled and preflighted after source generation and
before the first review forward. Complete prompt storage is divided into fixed
shares across attempted statements, and projected-logit limits accumulate
across every review head. The maximum original source-generation reservation
plus all review work must fit the five-axis `max_model_work` ceiling. Early EOS
does not refund that original reservation to the second reader.

The hosted path reserves memory for full resident KV, native workspace, all
retained source/claim copies and judge plans, simultaneous judge results and
the combined output. It runs both stages within one dispatch deadline and
cancellation context. A failed plan, exceeded ceiling, cancellation, incomplete
judge result or invalid final accounting yields no combined success result.
There is no partial-review publication or automatic retry.

Optional source `--save-snapshot NEW_FILE` follows fixed-policy and memory
admission. Claims are available only after generation, so later dynamic review
refusal can leave a valid saved snapshot. Snapshot persistence does not imply
that the subsequent semantic task completed.

Library callers construct a native `PreparedInt4SourceTask` and pass it with
the exact original `SourceTaskRequest`, an `Int4JudgePlanner`, and explicit
options to `PreparedInt4SourceSemantic::new`. They then call
`NlpEngine::execute_int4_semantic_source` with
`Int4SourceSemanticHostLimits`. The shared `requirements` method supplies the
modeled complete memory quote before execution. Prepared plans are sealed;
caller-supplied result JSON cannot authorize the second stage.

This implementation and its regression sources have undergone source review.
Rust compilation, controller-run tests, real-model behavior, performance and
incremental error-catching value remain unverified until their separate
execution and evaluation gates run.
