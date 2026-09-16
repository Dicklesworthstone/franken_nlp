# Grammar-first native JSON execution

`native_engine::constrained_sparse::decode_json_eager_sparse` is an explicit
native decoding entrypoint over an already-admitted empty eager engine, exact
prompt tokens, compiled JsonProgram and tokenizer-derived VocabMaskOracle.
It reuses EagerPrefixSession rather than loading weights or creating a runtime.
The original `constrained::decode_json_eager` full-projection reference and its
existing extraction callers remain unchanged; this new strategy is opt-in.

At each output step, the decoder materializes the complete legal grammar mask,
removes excluded template/control tokens, and includes the explicit EOS only
when the state accepts. It then projects exactly those ascending token rows.
The greatest finite legal logit wins, with lowest token ID breaking ties and
positive/negative zero treated equally. Greedy selection computes no probability
or full-vocabulary denominator and does not claim one.

The entire prompt executes once without intermediate lm_head projections or
L2 tap copies. Every selected nonterminal token is fed back through all 44 KV
slots before the next projection, even for a one-bit mask or already-accepting
JSON. Grammar certainty never becomes a hidden-state skip. Accepting numeric
prefixes still score EOS against legal longer continuations. EOS itself is
scored but never fed into KV. Final success requires independent whole-JSON
validation; no partial or EOS-less result is returned.

SparseJsonLimits prices the maximum legal rows per step and the entire response
byte envelope. A row-limit failure refuses the COMPLETE legal set; it never
prunes options to fit. JsonWorkBudget continues to bound actual forward work,
full engine KV reservation, aggregate projected rows and mask work. Dynamic
legal-row sets are charged before native projection. A long prompt does not
pay a fictitious vocabulary-wide projection at every prefill position.

Cancellation uses the one caller-owned control state. Grammar/selection polls
use the generated-token index; native prefill/layer/head polls use the prefill
channel. The temporary shared adapters do not create independent deadlines or
budgets. A failed native session is poisoned before mutation; all session exit
paths clear logical KV while retaining the admitted engine buffers. Completed
native work is checked against returned forward, projection and token counts.

The response names `eager-grammar-first-selected-rows-v1` and nests the ordinary
JsonDecodeOutput with actual work counts. This is an execution-strategy record,
not a model-quality or performance receipt. Unlike the full-row reference,
this route checks finite values only for requested rows; it does not inspect or
certify unused vocabulary rows. Callers must retain that explicit distinction.

Seven source regression tests cover legal-row selection, EOS and accepting
numeric continuations, exclusions, no-pruning budgets, finite values and signed
zero ties, complete output limits, cancellation and aggregate mask work.
Compilation, tests, DSR, real-model equality and performance measurements have
NOT been run under the current code-first instruction. Artifact activation,
source-trust and process admission gates remain unchanged.
