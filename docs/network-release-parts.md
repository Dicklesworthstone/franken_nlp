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
that every hop can be checked. Only absolute HTTPS redirect targets are
accepted. Mirror URLs are attempted sequentially in manifest order.

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
`--network-reserve-mib` (default 16) creates an explicit process-ledger
reservation for HTTP/TLS/DNS/body bookkeeping. `--max-download-bytes` bounds
new network bytes for resumable invocations. `--no-resume` refuses an
existing partial file instead of extending it. `--request-timeout-seconds`
and `--max-redirects` are finite per-mirror controls.

A successful command prints metadata only to stderr and exits zero after every
part has a complete manifest digest. It still has
`activation_authority=false`.

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
