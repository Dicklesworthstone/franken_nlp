# Redaction command contract

`fnlp redact` now dispatches to the rule/union/action/verification library path.
This is implemented code, not a release certification: compilation, Rust tests,
real-model inference and platform durability qualification have not been run
for this change under the current code-first policy.

## Scope and basic use

```sh
fnlp redact --rules-only --verify document.txt
fnlp redact --rules-only --action mask --json < document.txt
fnlp redact --rules-only --rules email,phone,date document.txt
```

`--rules-only` is mandatory. It does not run NER and does not find arbitrary
people, organizations or locations. The default rules are email, phone, URL,
IP address and Luhn-valid credit-card shapes. Dates are opt-in. The versioned
rules do not normalize or universally catch Unicode obfuscations.

The default action is `placeholder`; `mask` and `pseudonymize` are also
implemented. Verification reruns the selected rules on the transformed text by
default; `--verify` makes that intent explicit. `--no-verify` is an explicit
opt-out and the result says `not_requested`, never a clean pass. A clean scan
is a result about this declared rule set, not anonymity or compliance.

Plain output preserves all untouched bytes and adds no final newline. `--json`
emits one canonical result record with a final newline. Pre-output failures
leave stdout empty; structured, content-free error records go to stderr.
An output-stream failure can of course occur after a partial OS write.

## Keyed pseudonyms

Supply raw, high-entropy binary key material, 32 through 4096 bytes; byte
length checks cannot prove entropy. No key-value argv option exists. `--key-id`
is a public rotation identifier, not the secret. `--namespace` is required.

```sh
fnlp redact --rules-only --action pseudonymize \
  --key-stdin --key-id rotation-1 --namespace corpus-1 document.txt < secret.key

fnlp redact --rules-only --action pseudonymize \
  --key-file private/secret.key --key-id rotation-1 --namespace corpus-1 \
  --json < document.txt
```

Key stdin and document stdin cannot be combined. A file used as the key cannot
also be opened as the document under an inode alias. Keys are not trimmed,
logged, stored in maps, or used as random seeds. `--expected-key-commitment`
accepts a saved lowercase HMAC commitment and refuses the wrong key before
reading document input. An inherited-descriptor CLI option is not implemented.

By default, the complete document's pseudonymized values are preflighted before
any 128-bit token is emitted. Mixed-type overlap regions are conservatively
masked. A single invocation is a single-document job, not a corpus-wide
preflight. A larger job that cannot preflight its entire value set must select
`--full-digest` from the start. No automatic short-to-long mode switch occurs.
Preflight exhaustion fails; external-sort overflow is not implemented.

## Protected local files

```sh
mkdir -m 700 private-output
fnlp redact --rules-only --verify document.txt \
  -o private-output/redacted.txt --map-out private-output/coordinates.json
```

File keys, `-o` and `--map-out` currently have an explicit Linux x86_64/aarch64
implementation requiring procfs. Parent directories must already exist, be
owned by the process user, and have mode 0700. No symlink path components or
`..` traversal are accepted. Key files must be owner-only regular files with
one link. Final output paths must not exist; there is no overwrite flag.
`-o -` means stdout. `--map-out -` is refused. Other platforms refuse protected
file operations rather than substitute weaker path-based checks; stdin and
stdout redaction remain available.

The implementation pins directory handles, creates exclusive 0600 staging
files, writes/syncs complete bytes, and publishes with a same-directory,
no-replace hard link followed by directory sync. The staging namespace
`.fnlp-redact-*` is reserved. It does not grant model-root/cache authority.
This profile assumes a trustworthy procfs/local filesystem; same-UID/root
attackers and malicious mounts are outside its scope. No network-filesystem
or power-loss durability certification is claimed.

The coordinate map contains original and transformed byte/scalar intervals,
action/provenance, verification state, and public key/policy commitments. It
contains neither original matched values nor secret keys and is not a
re-identification dictionary. With `--map-out`, coordinates are emitted only
to that file, not duplicated in JSON stdout or diagnostics.

Both files are fully staged before either is published. They are **not an
atomic two-file transaction**: the map is published first and the main output
last. If the main output fails, the error reports `map_published: true` and the
map is retained. A failed publication/sync may leave a complete final file;
that is reported as uncertain publication, never success. No rollback deletes
final files. Interrupted cleanup may leave a private staging file.

## Bounds and failures

`--max-input-bytes` bounds document reads before buffering. Only one byte past
the limit is consumed to detect overflow. `--max-output-bytes` bounds the
library's complete result and the aggregate emitted document plus map bytes,
including JSON record newlines. Both byte settings have a 64 MiB CLI ceiling.
Detection counts and each complete rule scan have separate finite budgets.
`--max-preflight-values` and `--max-preflight-bytes` bound truncated-mode
preflight; use full digests explicitly rather than relying on a silent fallback.

Exit codes use the shared `ErrorCode` authority: invalid arguments 2, document
read/decode failure 4, rule work exhaustion 5, checked resource/file refusal 9,
and residual findings, collision or key-commitment mismatch 10. An output
stream/serialization failure uses 1. Failures never echo source or key bytes.
