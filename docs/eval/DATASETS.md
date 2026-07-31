# Dataset registry

No dataset may enter CI or a scorecard without a complete record below. The
registry is deliberately machine-readable JSON inside Markdown so that the
human review surface and the validator share one authority. Production datasets
must name an immutable source, exact license terms, split ids, preprocessing
digest, and contamination risks before acquisition.

## fnlp-repo-authored-binary-v1

```json
{
  "acquisition": {
    "automated_access_allowed": true,
    "mode": "repo-authored-fixture",
    "user_supplied_allowed": true
  },
  "allowed_use": "CI and evaluation-harness mechanics only; this synthetic fixture is not evidence of model quality.",
  "contamination_risks": [
    "The fixture is public in this repository and must never be used for a quality or release claim.",
    "The fixture tests only split, digest, identity, and deterministic-metric plumbing."
  ],
  "dataset_digest_sha256": "028a6697cad85f58687cd0c0c7d68660734665a993ef6865b69d7fe6172e6e28",
  "dataset_id": "fnlp-repo-authored-binary-v1",
  "dataset_version": "1",
  "fixture_path": "tests/fixtures/eval/stub_binary_dataset.ndjson",
  "license_name": "LicenseRef-FNLP-Repo-Authored-Fixture-1.0",
  "license_text": "FrankenNLP repository-authored fixture license 1.0: maintainers grant permission to copy, modify, redistribute, and execute these synthetic rows solely to test FrankenNLP evaluation-harness mechanics. The rows contain no third-party corpus content and convey no benchmark-quality claim.",
  "preprocessing": "identity-preprocessing-v1: UTF-8 NDJSON rows are consumed without normalization, deduplication, or label transformation.",
  "preprocessing_digest_sha256": "159f60a59d2c64af6fbbb997a7a25d9dda1f616f38abbf40d27d16ec94077134",
  "redistribution": "Allowed under the exact repository-authored fixture license text above.",
  "source": "repo-fixture://franken_nlp/tests/fixtures/eval/stub_binary_dataset.ndjson#sha256=028a6697cad85f58687cd0c0c7d68660734665a993ef6865b69d7fe6172e6e28",
  "splits": {
    "calibration": ["cal-001", "cal-002"],
    "development": ["dev-001", "dev-002"],
    "test": ["test-001", "test-002", "test-003", "test-004", "test-005", "test-006"]
  },
  "unit_of_analysis": "One independent synthetic binary-label row identified by id."
}
```
