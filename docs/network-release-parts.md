# Candidate HTTPS release-part acquisition

The canonical release pipeline now has an executable **pre-activation** network
slice under the `asupersync-runtime` feature:

`artifact::pull::http::download_release_parts`

This API is intentionally narrower than the eventual `fnlp pull`. It obtains
the immutable `.partNN` objects named by an independently pinned canonical
release manifest and leaves them in a caller-controlled workspace for the
existing `models reconstruct` path. It does **not** activate a model, write a
model root, or weaken the blocked platform transaction contract.

## Trust boundary

The caller provides:

- an explicit manifest pathname;
- the independently obtained domain-separated manifest SHA-256;
- an existing non-symlink parts directory;
- an explicit asupersync `Cx`.

Before network access, the implementation:

1. reads the bounded canonical manifest;
2. checks the independent manifest digest;
3. parses the closed release-manifest schema;
4. applies the installed binary's exact Nanbeige4.2-3B/int8 compatibility row;
5. refuses non-active lifecycle states.

A mirror URL never becomes its own trust root. Final success for a part requires
its full manifest SHA-256.

## Transfer contract

Every request is an explicit byte range and sends:

- `Range: bytes=<start>-<end>`;
- `Accept-Encoding: identity`;
- `User-Agent: franken_nlp-agent (+https://github.com/Dicklesworthstone/franken_nlp)`.

The pinned asupersync client is configured with no automatic redirects, no
automatic retries, no cookies, a finite request timeout and a body cap no larger
than the largest declared release part. Redirects are followed explicitly so
that every hop is checked **before** the next request is constructed. Every
target must be an absolute HTTPS URL with a plain ASCII hostname, no userinfo,
no explicit port (including `:443`), and no whitespace, controls, backslashes or
fragment. The check uses the exact selected asupersync revision's
[`ParsedUrl` parser](https://github.com/Dicklesworthstone/asupersync/blob/362dc5b174427f66cfa76ab2bdd68cce1a95c6cc/src/http/h1/http_client.rs)
after the stricter product authority checks. Policy refusal messages omit the
URL and its potentially sensitive signed query.

Redirect authority remains bound to the original manifest mirror throughout
the attempt:

- A custom mirror may redirect to its own exact hostname, compared without
  case sensitivity. Subdomains, sibling mirrors and suffix lookalikes do not
  inherit permission.
- A mirror originally on `github.com` may also redirect to the exact hosts
  `release-assets.githubusercontent.com`, `objects.githubusercontent.com` and
  `github-releases.githubusercontent.com`. These names come from GitHub's
  [documented download-domain inventory](https://docs.github.com/en/actions/reference/runners/self-hosted-runners#accessible-domains-by-function):
  the release-assets host and two existing GitHub download hosts. No wildcard
  `githubusercontent.com`, Azure or S3 permission is granted.
- An accepted asset-host hop cannot broaden the original policy. A custom
  mirror that uses another CDN must keep the request on its declared hostname;
  arbitrary cross-host private-CDN redirects are refused.

Signed paths and queries are forwarded byte for byte, including percent
escapes and query parameters. Redirect URLs have a separate 16 KiB limit, so
they can exceed the immutable manifest URL's 2 KiB cap without becoming
unbounded. Mirror URLs are attempted sequentially in manifest order; a refused
redirect follows the existing rollback and fallback path.

One timeout covers the **complete mirror attempt**: all redirect hops, the
response headers, and the streaming body. The effective deadline is the earlier
of that attempt's timeout and the caller's `Cx` deadline. Neither a redirect nor
a received body frame renews it. This explicit outer timer is necessary because
the pinned client's streaming request timeout ends at the response head.

While a transfer is pending, an owned runtime timer also checks the caller's
`Cx` cancellation every 100 ms. The same transfer stays alive between checks;
the timer neither restarts it nor extends its attempt deadline. Cancellation
drops the transfer before rollback and preserves the original cancellation
cause. The interval is subject to runtime scheduling; synchronous filesystem
work remains cooperative between bounded operations.

A data response must be exactly:

- status `206`;
- one canonical `Content-Range` equal to the requested range and declared full
  part length;
- one exact `Content-Length`;
- no `Transfer-Encoding`;
- no content encoding other than absent or `identity`;
- exactly the requested number of streamed body bytes;
- no trailers.

Malformed or duplicate authority headers are refused.

## Resume and failure behavior

The final part name is never overwritten. An existing final part is reused only
after its length and complete SHA-256 pass again.

Incomplete work is stored at `<part-name>.partial`. With resume enabled, the
entire retained prefix is re-hashed before the next request. A failed mirror is
rolled back to the exact pre-attempt length before the next mirror is tried.
The same rollback applies to a stalled body's timeout, after the body future
has been dropped. Received bytes from failed attempts remain charged against
the invocation's download allowance. Exhausting mirrors on a timeout preserves
the typed timeout and CLI exit category 5; a caller's cancellation retains its
original cause.

The release manifest contains whole-part digests rather than per-range digests,
so an incomplete prefix is deliberately **not** described as verified release
content. It becomes trusted only after the final complete part hash passes.

`max_new_bytes` bounds newly received bytes across one invocation. When that
allowance ends inside a part, the successfully received range remains in the
partial file and the operation returns `DownloadBudgetExceeded`; a later call
can resume it. Cancellation similarly produces no success report.

Final publication uses a no-replace hard link from the completed partial file.
The partial is retained as forensic/resume evidence.

## CLI surface

With `asupersync-runtime` enabled, the process-owned runtime exposes the same
transport through:

```sh
fnlp models download-parts \
  --manifest /release/model.manifest.json \
  --manifest-sha256 "$INDEPENDENT_MANIFEST_SHA256" \
  --parts-dir /release/parts \
  --memory-mib 256
```

The command installs one ordinary `EngineResources` host and runs the HTTP
future on that owned runtime; it does not create a second nested runtime.

For the complete remote-to-local-artifact candidate path with a reusable parts
workspace, use:

```sh
fnlp models acquire \
  --manifest /release/model.manifest.json \
  --manifest-sha256 "$INDEPENDENT_MANIFEST_SHA256" \
  --parts-dir /release/parts \
  --staging-output /models/model.fnlpq.stage \
  --output /models/model.fnlpq \
  --memory-mib 256
```

`models acquire` first admits 128 MiB of modeled local verification and
reconstruction workspace in the process memory ledger. It then authenticates
the supplied manifest pin and checks the final `--output` path. An existing
regular file is reused only after its complete length, SHA-256, checked
container structure, and manifest-bound semantic identities all pass again.
The held file descriptor is used throughout verification, and an observed
pathname replacement is refused. Corruption, truncation, symlinks, unreadable
files, or identity mismatches are errors; they never silently authorize a
replacement or download.

A verified final artifact succeeds without opening the parts directory or
staging path and without contacting any mirror. The receipt includes
`cache_hit=true`, zero network/copy work, and `local_workspace_bytes=134217728`.
This works with `--max-download-bytes 0 --max-copy-bytes 0`, including when the
original parts have been removed. Cache hits still require the
`asupersync-runtime` feature for the real process ledger.

When the final path is absent, acquisition completes the same verified part
download and then invokes local reconstruction against those final part files.
The receipt includes `cache_hit=false`. Network failure never enters
reconstruction; reconstruction keeps the forensic/resume semantics of
`models reconstruct`. Both paths retain the local workspace charge through
receipt publication. That charge models bounded metadata representations and
streaming scratch; it is neither a 128 MiB allocation nor an exhaustive RSS cap.

`--no-resume` disables reuse of both network partials and a pre-existing
reconstruction staging file. Otherwise, reconstruction resumes only when the
named staging file already exists; a fresh invocation uses create-new semantics.
This option does not disable reuse of a completely verified final artifact.
The network and reconstruction byte ceilings remain independent:
`--max-download-bytes` bounds newly received bytes and `--max-copy-bytes`
bounds newly appended reconstruction bytes.

`--network-reserve-mib` (default 16) creates an explicit process-ledger
reservation for HTTP/TLS/DNS/body bookkeeping. `--max-download-bytes` bounds
new network bytes for resumable invocations. `--no-resume` refuses an
existing partial file instead of extending it. `--request-timeout-seconds`
and `--max-redirects` are finite per-mirror controls. For `models acquire`, the
network reservation is additional to the 128 MiB local workspace; a cache hit
needs only the local reservation. An insufficient local memory ceiling is
refused before opening the manifest or artifact, with exit category 9.

A successful command prints metadata only to stderr. `download-parts` exits
zero after every part has a complete manifest digest; `acquire` exits zero only
after the complete final artifact is verified or reconstructed. Both retain
`activation_authority=false`.

## Acquire directly into one artifact

Add `--direct` when the goal is a complete checked artifact without retaining
a separate full-model parts workspace:

```sh
fnlp models acquire --direct \
  --manifest /release/model.manifest.json \
  --manifest-sha256 "$INDEPENDENT_MANIFEST_SHA256" \
  --staging-output /models/model.fnlpq.stage \
  --output /models/model.fnlpq \
  --memory-mib 256
```

Each bounded HTTPS response appends directly to the same staging file. There
is no second reconstruction write and no `.partNN` or per-part `.partial`
file. Staging and final output must have the same parent directory. Successful
publication creates a no-replace hard link, so the retained staging name and
final name refer to the same artifact storage. This avoids the separate
artifact-sized parts copy used by the default mode. It does not measure or
promise free disk space.

`--direct` rejects `--parts-dir` and `--max-copy-bytes` before IO. Without
`--direct`, the existing parts directory requirement and independent
download/copy limits remain. The direct mode uses `--max-download-bytes` for
newly received body bytes across every part and mirror attempt. Discarded
failed-mirror bytes still spend that allowance. A successful partial range
retains its bytes, returns exit category **5**, and can be resumed by repeating
the command. Rollback after a failed mirror preserves the exact pre-attempt
prefix, including all earlier complete parts.

An existing regular staging file is resumed by default. Before any request,
the command rehashes every retained complete part and compares it with the
pinned manifest. It also rehashes a retained incomplete part, but that tail is
not authenticated until its complete part digest passes. A corrupt retained
tail therefore cannot produce a successful artifact; failed completion keeps
the retained input for inspection rather than silently repairing it. Use a
new staging pathname to restart from fresh bytes. `--no-resume` refuses any
existing staging file. Oversized staging, symlinks, and observed pathname
replacement are refused.

A complete staging artifact can finish with `--max-download-bytes 0` and
without the original release parts. Before creating the final name, the
shared publication boundary syncs staging, rehashes its entire length,
checks the manifest's whole-artifact SHA-256, and validates the checked
container and all manifest-bound semantic identities. An already existing
final artifact is independently verified first and succeeds as a cache hit
without touching staging or contacting mirrors. An invalid existing final
file is never replaced.

The direct receipt includes `mode=direct`, `cache_hit`, `resumed_bytes`,
`network_downloaded_bytes`, and `retained_new_bytes`. The last two differ when
failed mirrors consumed bytes that were rolled back. A cache hit reports zero
for all three byte counters. Direct mode admits both the 128 MiB local
verification workspace and the configured network reserve before any IO,
including a cache hit; the default reserve therefore needs at least 144 MiB
of modeled memory allowance. The reservations model bounded metadata and
transport bookkeeping, not artifact-sized allocations or an exhaustive RSS
cap.

Library users can call
`artifact::pull::http::direct::acquire_release_artifact` with an explicit
`Cx`, `NetworkArtifactRequest`, and `NetworkPartsOptions`. The synchronous
`NlpEngine::acquire_release_artifact_candidate` adapter owns those local and
network reservations on the existing process runtime. Both return a
`NetworkArtifactReport` with `activation_authority=false` and
`directory_durability_confirmed=false`.

This remains a caller-controlled stable-path operation. It does not supply
the hostile model-root transaction or trusted activation journal. Cancellation
and synchronous filesystem failure can leave staging; cancellation detected
after publication can also leave a verified final pathname without a success
receipt. A retry verifies that pathname rather than overwriting it. The same
per-mirror timeout, cancellation, redirect, exact `206 Content-Range`, and
identity-encoding checks apply to both acquisition modes.

Source regressions cover multipart resume boundaries, corrupt prefixes and
tails, absolute-offset rollback, failed-mirror byte accounting, deterministic
timeout/cancellation, and the real current-format container's zero-network
completion/cache paths. Their execution remains pending the repository's
controller-selected DSR checkpoint; this candidate is not release evidence.

## What this does not claim

This code closes the previously absent transport path, but the project still
requires the G0 HTTPS matrix to qualify proxy/private-CA behavior and target
evidence. This candidate therefore makes no release-certification claim.

In particular:

- proxy environment-variable behavior is not claimed;
- private mirror credentials are not implemented;
- relative redirects are refused;
- activation remains blocked by the model-root platform-surface gate;
- directory durability is not claimed;
- no model is selected or made active.

The expected high-level sequence today is therefore:

1. acquire and independently pin the release manifest;
2. call the candidate HTTPS acquisition API to populate the local parts
   workspace;
3. run `fnlp models reconstruct` using those exact parts;
4. use candidate inference directly against an independently pinned artifact
   where supported.

The future authoritative `fnlp pull` may compose these stages only after the
remaining G0 network and model-root authority gates close.
