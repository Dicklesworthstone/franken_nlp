# Native candidate redaction

`fnlp candidate redact` runs source-bound native INT8 NER plus the existing
rule detectors, applies transactional edits, and by default reruns BOTH detector
families on the transformed text. It requires `asupersync-runtime`, an explicit
compatible local model and a process memory ceiling. It does not activate a
catalog, fetch weights or certify model quality. The existing model-free
`fnlp redact --rules-only` command is unchanged.

```sh
fnlp candidate redact document.txt --model ./model.fnlpq --memory-mib 8192
fnlp candidate redact document.txt --action placeholder \
  --model ./model.fnlpq --memory-mib 8192
```

Input is exact UTF-8 from a file or omitted/`-` for stdin. One completed JSON
response contains candidate provenance and the native redaction result. Only
final transformed text is returned: intermediate NER transcripts, original
values and rejected partially redacted documents are not exported. Output and
preparation charges survive complete serialization, writing and flushing.

## Scope and verification

Default NER types are person, organization and location. `--ner-options FILE`
accepts a complete bounded `NerOptions` object (`types`, `max_entities`,
`max_mention_scalars`), capped at 16 KiB. Unknown fields, repeated types and
invalid limits fail before model metadata access. The default rules detect
ASCII contact shapes, HTTP(S) URLs, IP literals and Luhn-valid card shapes.
`--rules email,phone,url,ip-address,credit-card,date` explicitly replaces that
set; dates are opt-in. These rules do not normalize every Unicode obfuscation.

The default `--action mask` replaces each detected scalar with `*`, so masks
do not expand source byte length. `placeholder` emits typed replacements.
`pseudonymize` uses the full HMAC digest described below. Overlapping findings
are combined using the existing union policy; incompatible mixed regions are
masked instead of inventing a type. `--include-map` opts into sensitive
original/transformed coordinates inside the result, not telemetry.

Verification sees the actual edited text, not old offsets or original NER
results. Residual detections, a failed native pass, cancellation, or exceeded
bounds yield no successful output. `--no-verify` is an explicit opt-out and
retains the native `not_requested` status. A clean declared union is NOT a
claim that all PII was found. Model omissions and correlated verification errors
remain possible; source membership proves occurrence, not recognition accuracy.

## Full-length pseudonyms

```sh
fnlp candidate redact document.txt --action pseudonymize \
  --key-stdin --key-id rotation-1 --namespace dataset-1 \
  --model ./model.fnlpq --memory-mib 8192 < private.key
```

The caller supplies authorized high-entropy binary key material (32..4096 bytes).
It is read directly through the existing bounded secret reader, never argv or a
UTF-8 String. Key length checks do not establish entropy. With `--key-stdin`,
the document must use a separate file. No new key-file permission mechanism or
key-generation source is introduced. Keep shell traces and output access under
your own privacy controls. Secret buffers use the existing best-effort cleanup,
not a compiler-proof zeroization guarantee.

The same key, namespace, PII type and exact UTF-8 value produce the same
full-256-bit pseudonym. No per-document 128-bit collision preflight is claimed.
An optional `--expected-key-commitment` is checked before reading the document;
a different key fails rather than silently changing identities. Result metadata
retains the existing nonsecret key/scope commitments. Pseudonyms remain linkable
and are not anonymization. The native host owns its key share until physical
completion, including cancellation and failure.

## Bounded native execution

The shared source options retain 2048 context tokens, 512 maximum output tokens
per NER pass, 65536 original input bytes and a 1 MiB complete result by default.
The actual original NER plan is checked before weight materialization. A verified
request reserves its exact first-pass work plus one full admitted-context second
pass, summing all five independent work counters rather than concatenating
attention contexts. At most two passes run, on one engine and resident model.
`--max-mask-node-visits` is explicitly per pass; the total allowance is its
checked product with one or two passes. Failed passes do not renew it.

The second pass cannot be compiled until edits exist. Placeholder/pseudonym
expansion may exceed its context even when the original fits; this is a refusal,
never truncation, rules-only fallback or an implicit verification opt-out.
Transformed byte admission is widened only up to the already priced result cap;
exact native context checks remain independent. Complete-result size includes
policy metadata and any opted-in coordinate map, not just edited text.

Rule scans have a separately finite per-scan `--max-rule-work`. Detection and
edit counts share `--max-detections` (4096 default, at most 16384). The hosted
edit reserve (`--edit-reserve-mib`, 64 default) has a checked floor for region
metadata and replacement staging. Shared preparation defaults to 512 MiB.
These are modeled ledger commitments, not OS-enforced or measured RSS bounds.
One cooperative elapsed limit follows key/input reads, planning, loading and
execution; blocking I/O and individual tokenizer operations are not preempted.
Without the runtime feature, refusal precedes key, document, options and model
I/O. Failures use closed diagnostics and no candidate result prefix is printed.

## Validation status

Regression sources cover command routing, detector scope, verification opt-out,
secret-source conflicts, binary-key bounds and commitments, limits, actual pinned
NER preparation, independent work accounting and cancellation. They are not
neural-success fixtures. Rust compilation/tests, model inference, PII recall,
benchmarks and controller DSR were not executed for this implementation.
All successful outputs retain `evidence=non_authoritative`.
