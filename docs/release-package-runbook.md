# Model release packaging runbook

This runbook packages one immutable Generic `.fnlpq` artifact for a draft GitHub Release. It never rewrites a release asset, uses no wildcard upload, and treats an old release as permanently downloadable by an offline client that already holds its manifest.

## Preconditions

- The converter output is a finished, independently verified Generic `.fnlpq` file.
- The converter receipt is retained as a regular file.
- The license directory contains exactly `APACHE-2.0.txt`, `ATTRIBUTION.txt`, and `MODIFICATION_NOTICE.txt`.
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

The package must contain only the ordered `.part00`, `.part01`, … files, `MODEL_ASSET_RECEIPT.json`, `SHA256SUMS`, `RECONSTRUCTION.txt`, `CONVERSION_RECEIPT.json`, and the three license-bundle files. Every non-final part is exactly 1,957,046,720 bytes. The verifier rehashes every member, rejects renamed/reordered/tampered parts, and streams the ordered parts into the whole-artifact SHA-256 without materializing a second multi-GB file.

## Create a draft and upload explicit assets

Create the draft release first. Then copy the exact paths from `MODEL_ASSET_RECEIPT.json` into the upload command. Do not use a glob and do not use `--clobber`.

```sh
gh release create fnlp-model-v1 --draft --title 'Nanbeige4.2-3B artifact v1'

gh release upload fnlp-model-v1 \
  /absolute/path/release-staging/nanbeige4.2-3b-v1/nanbeige4.2-3b.fnlpq-v1.int8.generic.fnlpq.part00 \
  /absolute/path/release-staging/nanbeige4.2-3b-v1/nanbeige4.2-3b.fnlpq-v1.int8.generic.fnlpq.part01 \
  /absolute/path/release-staging/nanbeige4.2-3b-v1/nanbeige4.2-3b.fnlpq-v1.int8.generic.fnlpq.part02 \
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

Query the draft inventory and compare it exactly to the local receipt. Download every individual asset into a newly created clean directory, then run the same verifier there.

```sh
gh release view fnlp-model-v1 --json assets

fnlp release verify-model-package \
  --staging-dir /absolute/path/clean-download/nanbeige4.2-3b-v1
```

Use `scripts/e2e_release_package_verify.sh --package-dir …` for the local package→verify→tamper-must-fail transcript. Publishing the draft is authorized only after both the local and clean-download verifiers pass and their receipts match.

## Revocation and supersession

Release assets are immutable. If a release is bad, publish a new artifact version and catalog entry; never replace bytes under the old tag/name. Revocation cannot recall a model that an offline client already downloaded or whose old embedded manifest still points at an immutable old release. State that limitation plainly in advisories and tooling output.
