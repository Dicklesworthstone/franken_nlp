# Resumable local release reconstruction

`fnlp models reconstruct` assembles an explicit local release package into a
checked Generic `.fnlpq` file. The manifest must be canonical, compatible and
active, and its ordinary SHA-256 must come from an independently trusted
channel. The command reads local parts only. The resulting file is
not an activated model or a release-qualification receipt.

`--manifest-sha256` and the report's `release_manifest_sha256` name the
standard SHA-256 of the exact canonical manifest file bytes, matching
`sha256sum model.manifest.json` or `shasum -a 256 model.manifest.json`.
There is no domain prefix or JSON normalization in the hash. Pins from the
earlier domain-prefixed candidate deliberately fail with a manifest-digest
mismatch; obtain the corrected standard checksum through the same independent
trusted channel. Computing a checksum over an untrusted download alone does
not establish publisher identity.

## Start and resume

Use caller-controlled directories whose paths and contents remain stable for
the invocation. Staging and final output must have the same parent directory;
the final output must be a new pathname.

```sh
fnlp models reconstruct \
  --manifest /release/model.manifest.json \
  --manifest-sha256 "$INDEPENDENT_RELEASE_MANIFEST_SHA256" \
  --parts-dir /release \
  --staging-output /models/model.fnlpq.stage \
  --output /models/model.fnlpq
```

The default creates a new staging file. If an invocation stops, retain the
manifest, all release parts and staging, then repeat the command with
`--resume`. Resume requires existing regular staging and never truncates it.
It compares every retained byte against its source part and verifies that
entire part, including the suffix of a partially copied part, before the first
append. A mismatching prefix, changed part or oversized staging file refuses
without repairing or extending the retained file. Starting with a new staging
pathname remains available when the retained file is unusable.

If earlier part files are on removable media or have been archived, add
`--resume --verify-staged-parts`. This hashes each **complete** retained part
directly from staging against the pinned manifest. Supply the source file for
a partially retained part and all remaining parts in `--parts-dir`. A complete
staging file can be verified and published with an empty parts directory and
`--max-copy-bytes 0`. Any damaged retained part stops the operation before
the first append. The equivalent library entry point is
`resume_local_release_from_staged_parts`.

For an explicit amount of new copying per invocation, add a byte limit:

```sh
fnlp models reconstruct \
  --manifest /release/model.manifest.json \
  --manifest-sha256 "$INDEPENDENT_RELEASE_MANIFEST_SHA256" \
  --parts-dir /release \
  --staging-output /models/model.fnlpq.stage \
  --output /models/model.fnlpq \
  --resume --max-copy-bytes 1073741824
```

Exhausting the copy limit returns exit code **5**, leaves the exact copied
prefix in staging and publishes no successful result. Repeat with `--resume`
to continue. The limit counts newly appended bytes; prefix authentication and
complete-artifact verification require additional reads. A zero-byte allowance
can finish an already complete staging file after all checks pass.

## Completion and ownership

Before publication, reconstruction syncs staging, rehashes the entire file,
checks its total length and whole-artifact digest, then validates the container
and manifest-bound identities through a clone of the held file descriptor.
It publishes with a no-replace hard link. An existing final file is never
overwritten. Staging remains after success as well as failure.

The success report contains `resumed_bytes`, `copied_bytes`, total artifact
bytes, part count, artifact SHA-256 and the trusted manifest SHA-256. Success
means complete checked local reconstruction. Directory durability remains
`unconfirmed`. This operation does not provide a hostile shared-root
transaction, publisher authentication, network acquisition, native packing or
automatic model activation. A process crash or filesystem failure can leave
uncertain publication; inspect retained paths rather than assuming deletion or
retry with overwrite semantics.

## Library use

`reconstruct_local_release(&request)` keeps the create-new default.
`reconstruct_local_release_with_options(&request, &options, checkpoint)` adds
`LocalReconstructOptions { resume, max_new_bytes }` and cooperative checkpoints.
The callback receives `LocalReconstructProgress` for `VerifyPrefix`, `Copy`,
`VerifyArtifact` and `Publish`, and can return a typed `DecodeCancellationKind`.
Cancelled operations return `PullError::Cancelled` with progress and no success
report. Checkpoints run between bounded reads and writes; synchronous operating
system calls cannot be preempted.

Regression fixtures cover real package assembly, interrupted/resumed byte
identity, complete staging, corrupt prefixes and parts, cancellation, budgets,
no-replace publication, symlink refusal and multipart boundaries. Independent
standard-checksum goldens and old-domain-pin refusals cover embedded manifests,
local reconstruction, final-cache reuse and HTTPS acquisition entrypoints. Compilation
and Rust tests remain pending the repository's controller-selected DSR
checkpoint; these source fixtures do not qualify a model or a release.
