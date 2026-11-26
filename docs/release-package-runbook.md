# Model release packaging runbook

This runbook packages one immutable Generic `.fnlpq` artifact for a draft GitHub Release. It never rewrites a release asset, uses no wildcard upload, and treats an old release as permanently downloadable by an offline client that already holds its manifest.

## Preconditions

- The converter output is a finished, independently verified Generic `.fnlpq` file.
- Supply the canonical receipt sidecar emitted by the converter. Packaging checks its `output_len` and `artifact_raw_sha256` against the bytes it splits. Standalone verification repeats the binding check against the reconstructed artifact.
- The three staged texts must reproduce the converter’s embedded license-bundle digest. The license directory contains exactly `APACHE-2.0.txt`, `ATTRIBUTION.txt`, and `MODIFICATION_NOTICE.txt`.
- The output staging directory does not exist. Packaging creates it once; a failed staging directory is retained for inspection and is never reused.

## Package and locally verify

```sh
fnlp release package-model \
  --artifact /absolute/path/nanbeige4.2-3b.fnlpq-v1.int8.generic.fnlpq \
  --staging-dir /absolute/path/release-staging/nanbeige4.2-3b-v1 \
  --logical-artifact-name nanbeige4.2-3b.fnlpq-v1.int8.generic.fnlpq \
  --conversion-receipt /absolute/path/CONVERSION_RECEIPT.json \
  --license-bundle-dir /absolute/path/license-bundle

fnlp release verify-model-package \
  --staging-dir /absolute/path/release-staging/nanbeige4.2-3b-v1
```

The package must contain only the ordered `.part00`, `.part01`, … files, `MODEL_ASSET_RECEIPT.json`, `SHA256SUMS`, `RECONSTRUCTION.txt`, `CONVERSION_RECEIPT.json`, and the three license-bundle files. Every non-final part is exactly 1,957,046,720 bytes. The verifier rehashes every member, rejects renamed/reordered/tampered parts, and computes the individual part and reconstructed artifact hashes from the same read of each part.

## Generate the manifest consumed by pull

After local verification, write the canonical manifest to a new file outside the package directory. The verifier checks the directory inventory exactly. The release tag and artifact ID in this example must match the proposed release.

```sh
fnlp release create-manifest \
  --staging-dir /absolute/path/release-staging/nanbeige4.2-3b-v1 \
  --artifact /absolute/path/nanbeige4.2-3b.fnlpq-v1.int8.generic.fnlpq \
  --source-manifest docs/truth-pack/nanbeige4.2-3b.source.json \
  --artifact-id nanbeige42-int8-generic-v1 \
  --release-tag fnlp-model-v1 > /absolute/path/new-release.manifest.json
```

On success, stdout contains the exact canonical bytes and stderr reports their standard SHA-256. The command checks the ordered parts and license bundle, reads the original artifact through the same file handle used for digest verification, and runs the container-identity checks used by pull. It derives the ten source records from the pinned source manifest and constructs the part URLs for this repository and tag. Treat the manifest as a local release candidate until the release is qualified and the digest is distributed through a trusted channel. Review the exit status before using the redirected file.

## Create a draft and upload explicit assets

Create the draft release first. Then copy the exact paths from `MODEL_ASSET_RECEIPT.json` into the upload command. Do not use a glob and do not use `--clobber`.

```sh
gh release create fnlp-model-v1 --draft --title 'Nanbeige4.2-3B artifact v1'

gh release upload fnlp-model-v1 \
  /absolute/path/release-staging/nanbeige4.2-3b-v1/nanbeige4.2-3b.fnlpq-v1.int8.generic.fnlpq.part00 \
  /absolute/path/release-staging/nanbeige4.2-3b-v1/nanbeige4.2-3b.fnlpq-v1.int8.generic.fnlpq.part01 \
  /absolute/path/release-staging/nanbeige4.2-3b-v1/nanbeige4.2-3b.fnlpq-v1.int8.generic.fnlpq.part02 \
  /absolute/path/new-release.manifest.json \
  /absolute/path/release-staging/nanbeige4.2-3b-v1/MODEL_ASSET_RECEIPT.json \
  /absolute/path/release-staging/nanbeige4.2-3b-v1/SHA256SUMS \
  /absolute/path/release-staging/nanbeige4.2-3b-v1/RECONSTRUCTION.txt \
  /absolute/path/release-staging/nanbeige4.2-3b-v1/CONVERSION_RECEIPT.json \
  /absolute/path/release-staging/nanbeige4.2-3b-v1/APACHE-2.0.txt \
  /absolute/path/release-staging/nanbeige4.2-3b-v1/ATTRIBUTION.txt \
  /absolute/path/release-staging/nanbeige4.2-3b-v1/MODIFICATION_NOTICE.txt
```

The example names the expected three parts of the int8-all artifact. Before running it, replace that list with every and only the explicit part paths named by the actual receipt; an artifact may have a different part count. Generate the final command from the receipt and review every expanded path before execution.

## Remote inventory and clean download

Query the draft inventory and compare it exactly to the local receipt. Download the package members into a newly created clean directory and run the same verifier there. Download the canonical manifest attachment separately, compare its bytes with the generated file, and verify its pinned digest.

```sh
gh release view fnlp-model-v1 --json assets

fnlp release verify-model-package \
  --staging-dir /absolute/path/clean-download/nanbeige4.2-3b-v1
```

Use `scripts/e2e_release_package_verify.sh --package-dir …` for the local package→verify→tamper-must-fail transcript. Publishing the draft is authorized only after both the local and clean-download verifiers pass and their receipts match.

## Revocation and supersession

Release assets are immutable. If a release is bad, publish a new artifact version and catalog entry; never replace bytes under the old tag/name. Revocation cannot recall a model that an offline client already downloaded or whose old embedded manifest still points at an immutable old release. State that limitation plainly in advisories and tooling output.
