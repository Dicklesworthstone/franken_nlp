# Faithfulness and semantic extraction: native/library source implementation

This change implements the missing judge faithfulness mode (plan 7.6) and an
explicit experimental extraction second reader (plan 7.1), reusing the same
finite-candidate scorer, pinned tokenizer and admitted native prefix engine.
It does not activate inference CLI commands or bypass model/process admission.
The planned `extract --verify-semantic` CLI switch remains unwired.

Compilation, builds, tests, harnesses, DSR, conformance, real-model evaluation
and benchmarks have NOT been run for this change. Source-level implementation
is not qualification. All judgments remain uncalibrated model diagnostics.
No second read upgrades source membership to entailment proof, grants an
acceptance certificate, repairs an extraction, or authorizes a policy action.

## Full-source faithfulness and evidence

`JudgeRequest::Faithfulness` carries original `source`, `claim`, an explicit
`FaithfulnessPolicy`, and a TaskBudget. It shares the strict bounded JSON parser,
`JudgePlanner`, sealed `PreparedJudge`, generic provider and native execution
routes with pairwise/rubric judging. The low-level `FaithfulnessPlan` accepts
pinned SourceDocument values and matching TaskPlans rather than unrelated
text alongside caller-invented token arrays.

The six-segment prompt is:

    global / instruction / source-data / instruction / claim-data / scaffold

Trusted instructions define E as support for every material part of the claim,
C as explicit contradiction of a material part, and U as insufficient source
information. Absence is not contradiction; outside knowledge is not requested.
Source and claim are separately byte-preserving untrusted data. E/C/U use exact
ordinary byte-fallback continuations, each including an explicitly scored EOS.
All vocabulary rows contribute to sequence-log-probability denominators.

Head zero receives the COMPLETE source. The deterministic partition makes
nonoverlapping UTF-8 windows, preferring newline or sentence-like delimiters
in each window's latter half. It never normalizes text or drops a tail. At
most 31 windows plus the whole-source head fit the common 32-head limit.
If the complete source is one window, head zero also serves as that window:
it is not rerun or presented as independent corroboration.

For multiple windows every window is scored, including uncertain and contrary
windows. There is no retrieval filter that can hide inconvenient source text.
The response retains all window assessments, complete head score sets and work.
Source over-context, excessive windows, an unfit UTF-8 scalar, projection limits
or excessive selected evidence produce a typed error, not a truncated success.

The explicit candidate-weight and log-margin thresholds are UNCALIBRATED.
Exact argmax ties always abstain even when a caller chooses zero thresholds.
The global judgment cannot be replaced by a more confident selected window:

- Ambiguous global distribution: `ambiguous_distribution` abstention.
- Accepted global support/contradiction but no matching evidence window:
  `insufficient_evidence` abstention.
- An accepted contrary window, or local support/contradiction against global
  unsupported: `conflicting_evidence` abstention.

Otherwise an accepted model relation is entailed, contradicted or unsupported.
Unsupported means insufficient information according to the model; it does
not stand in for cancellation, failed inference, or unknown execution state.
Those execution failures abort the entire result.

Accepted support/contradiction carries every agreeing policy-passing window
within the declared evidence cap. Quotes come from original source slices;
byte and Unicode-scalar offsets are checked by the independent span validator.
Unsupported carries no invented quote. Abstentions preserve diagnostics rather
than presenting selected quotes as supporting an asserted relation.

Evidence is a bounded window, NOT necessarily a minimal clause. Splits can
separate premises, anaphora, negation or other context; the policy may abstain
on valid multi-window reasoning rather than fabricate a supporting span.
The delimiter heuristic is not a qualified multilingual sentence splitter.
A byte-valid quote only establishes membership. Model support judgments and
agreement between correlated heads still require locked factual-support evals.

## Schema-aware extraction claims

`ExtractPlan::prepare_semantic_verification` is the explicit second stage after
ordinary extraction. Its inputs include the original TaskPlan, pinned complete
SourceDocument, ExtractResult, extraction identity, SemanticVerificationSpec,
JudgePlanner, judge PlanContext and declared budgets. It checks the extraction's
TaskIR identity and single exact source segment; unrelated source substitution
or after-the-fact concatenation is refused. It reruns the existing independent
extraction finalizer, including schema, scalar-length and source-membership
checks, and requires the result envelope to match that finalization.

A specification must name schema_version=1, experimental_opt_in=true, a revision,
claim rules, faithfulness policy and judge budget. There is no default-on route.
The bounded parser rejects duplicate/unknown keys. The caller explicitly
supplies field meaning; a bare field name is not treated as a factual claim.

A ClaimRule contains an ID, typed schema path, scalar kind, prefix and suffix.
Rendering is exactly prefix + canonical JSON scalar + suffix, not arbitrary
code or a model-generated template. Example rule (library request data only):

```json
{
  "id": "paid-amount-v1",
  "path": [{"kind": "property", "name": "amount"}],
  "value_kind": "number",
  "prefix": "The paid amount was ",
  "suffix": "."
}
```

`each_item` is a distinct path step, not the property spelling `*`; array indices
remain in per-value result pointers. Pointers follow the validator convention:
root `$`, child `/name`, with `~0`/`~1` escaping. Rules must follow the declared
schema and match a non-verbatim string, integer, number or boolean leaf.
Duplicate identifiers/paths, mismatched kinds and absent schema paths refuse
before inference. Only present fields are traversed; missing optional values
are not invented. Empty containers receive explicit unchecked records.

Numbers use the independent exact Decimal representation and canonical spelling,
including its 38-significant-digit domain, never serde/f64 conversion. Strings
retain JSON quoting/escaping. The inserted value cannot silently become an
executable template. String-escape admission uses a conservative six-times-byte
upper bound; it can refuse a claim whose actual escaped spelling would fit.
This is an explicit resource restriction, not data truncation.

Every present leaf or empty container has a status. Configured model-judged
claims may be entailed, contradicted or unsupported. Verbatim fields remain
`not_checked` with `verbatim_membership_only`; existing occurrence evidence is
preserved, not promoted to semantic truth. Other explicit unchecked reasons
are `no_claim_rule`, `null_value`, `empty_container`, and `judge_abstained`.
A judge abstention retains the full diagnostic result and its spent work.
It is not relabeled unsupported. Infrastructure failures are errors, never
unchecked fields silently dropped from a successful composite.

The receipt retains the rendered claim, rule ID, task/rendering versions,
specification revision, coverage counts, complete faithfulness diagnostics and
same-model correlation. Every checked claim uses the EXACT public faithfulness
request/planner/scorer, not an alternate hidden verification prompt. The original
ExtractResult, exact JSON text and source-field occurrence evidence remain
unchanged in the composite result. No automatic edit or acceptance decision is
made. An empty claim-rule set makes no model calls and reports zero checked
fields; it does not claim all fields are verified.

## Identity, native execution and costs

V1 requires the same declared admitted logical model, artifact/packing recipe,
tokenizer, backend, numerics, KV type, thinking/tool modes and execution profile
as extraction. Task-specific templates and prompts necessarily differ. The
caller still owns artifact/model admission; supplying an identity is not itself
permission to activate weights or bypass process/resource controls.

All field judges are prepared before any second-reader callback. Their full
identities are retained privately and every supplied admitted identity is
checked before the first field executes, including the identities of later
fields. Private binding commitments include extraction identity/result,
specification, limits, claim-rendering version and all prepared judge identities.
Unkeyed content commitments and private execution identities are not emitted in
the result or public telemetry. Claims, quotes and field values are private task
output and must follow the caller's ordinary output-handling policy.

The native APIs are `PreparedSemanticVerification::execute_eager` and
`execute_eager_with_control`. Use the original extraction's now-empty admitted
engine and the complete ordered set of separately admitted judge identities.
The API preflights every field's context and complete engine KV reservation
before any forward. It sums exact planned native work and checks the caller's
ONE aggregate second-reader PrefixBudget; each field receives only its exact
precharged share. A full allowance is never renewed at every field.

Within a head the existing prefix engine reuses continuation prefixes. Between
heads/fields logical KV clears while preserving buffers and weights. No model
clone, second KV cache, worker pool, new runtime, hidden retry or cross-field
prefix-sharing performance claim is introduced. Nonempty caller state is
refused without clearing it. Typed native cancellation/deadline causes survive
wrapping; a failed later window or field yields no successful composite result.

Source and claim prompt costs repeat for each actual head. Logical scoring
work includes all attempted field/head projections. The native wrapper reports
ONLY second-reader native work; original extraction work remains separately
inside the unchanged extraction output. Full response ceilings include original
extraction, rendered claims, all field diagnostics, quotes and native metadata.
Template/walk/field/claim limits are checked explicitly. Aggregate prompt
admission uses conservative equal deterministic shares across configured claims;
projection allowance is decremented rather than reset between compiled claims.

Integration remains two-stage and explicit: obtain the ordinary extraction,
construct an opt-in specification and matching judge PlanContext, prepare all
semantic field plans, admit their retained identities, then execute with the
same empty admitted engine and an aggregate second-reader budget. This note is
not a claim that the `fnlp` CLI currently exposes those library calls.

## Source regression coverage and remaining gates

Twenty-six additional source regression cases cover UTF-8 coverage/offsets,
evidence limits, conflicting/insufficient support, whole-source ambiguity,
one-head reuse, full-window work and failure, identity/source/policy drift,
exact decimals, typed nested paths, explicit unchecked values, claim limits,
real field orchestration with synthetic logits, later-field atomic failure,
zero-callback identity refusal, opt-in, complete output limits, parser strictness,
aggregate native budgets and typed deadline propagation. The prior twenty
pairwise/rubric tests are preserved. ALL of these cases remain UNRUN.

Remaining gates include compilation/test cleanup, independent conformance,
locked real-model faithfulness and per-domain incremental-error-catching evals,
calibration and decision-policy qualification, throughput/resource measurements,
and inference CLI/admission integration. Passing source membership or repeating
this model is not a substitute for any of those gates.
