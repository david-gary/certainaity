# Certainaity — Testing Strategy

## Goals

1. **Correctness**: every component produces output within documented contracts.
2. **Forensic accuracy**: the ensemble achieves target F1/AUC on held-out benchmark datasets.
3. **Regression safety**: no change to a detection model degrades existing benchmark scores by more than 1 pp without explicit approval.
4. **API contract stability**: API responses match the published OpenAPI schema at all times.
5. **Chain-of-custody integrity**: SHA-256 hashes are computed correctly and never mutated.

---

## Test Taxonomy

```
tests/
  unit/
    test_ela.py
    test_noise.py
    test_cfa.py
    test_dct.py
    test_ingest.py
    test_report_generator.py
    test_auth.py
    test_rate_limiter.py
  integration/
    test_api_analyze.py
    test_api_reports.py
    test_pipeline_end_to_end.py
    test_resilience.py
  benchmark/
    eval_patchforensic.py
    eval_mantranet.py
    eval_spsl.py
    eval_inpainting.py
    eval_ensemble.py
  fixtures/
    authentic/          ← 20 known-authentic test images
    manipulated/        ← 20 known-tampered images with ground truth masks
    masks/              ← binary masks for the manipulated set
```

---

## Unit Tests

Unit tests are fast (< 100 ms each), isolated (no GPU, no disk I/O beyond small fixtures), and cover the logic of individual components.

### Feature Extractor Tests

**`test_ela.py`**

```python
def test_ela_authentic_low_signal():
    img = load_fixture("authentic/pristine_landscape.jpg")
    ela_map = compute_ela(img, quality=75)
    assert ela_map.mean() < 0.08, "ELA should be low for unmodified images"

def test_ela_spliced_high_signal():
    img = load_fixture("manipulated/splice_001.jpg")
    ela_map = compute_ela(img, quality=75)
    mask = load_fixture("masks/splice_001.png")
    # mean ELA inside manipulated region should be notably higher
    ela_in  = ela_map[mask == 1].mean()
    ela_out = ela_map[mask == 0].mean()
    assert ela_in > ela_out * 1.5

def test_ela_output_shape():
    img = Image.new("RGB", (512, 512))
    ela_map = compute_ela(img, quality=75)
    assert ela_map.shape == (512, 512)
    assert ela_map.min() >= 0.0 and ela_map.max() <= 1.0
```

**`test_noise.py`**
- Verify noise map is approximately uniform for authentic images with known sensor.
- Verify noise map shows discontinuity at splice boundary in manipulated images.
- Verify output dtype and normalization.

**`test_cfa.py`**
- Verify CFA artifact map is near-zero for images sourced from smartphones (no raw CFA visible post-JPEG).
- Verify autocorrelation peaks for a known raw-converted TIFF.

**`test_dct.py`**
- Verify DCT block similarity matrix is identity-like for an image with no repeated content.
- Verify high similarity scores between known copy-move source and destination blocks.

### Ingestion Tests

**`test_ingest.py`**

```python
def test_sha256_correctness():
    path = "fixtures/authentic/pristine_landscape.jpg"
    result = ingest_image(path)
    expected = subprocess.check_output(["shasum", "-a", "256", path]).split()[0].decode()
    assert result.sha256 == expected

def test_sha256_computed_before_processing():
    # Inject a mock that records call order
    order = []
    with patch("certainaity.ingest.compute_sha256", side_effect=lambda x: (order.append("hash"), "abc")[1]):
        with patch("certainaity.ingest.extract_exif", side_effect=lambda x: (order.append("exif"), {})[1]):
            ingest_image("fixtures/authentic/pristine_landscape.jpg")
    assert order[0] == "hash", "SHA-256 must be computed first"

def test_rejects_gif():
    with pytest.raises(UnsupportedFormatError):
        ingest_image("fixtures/test.gif")

def test_rejects_oversized():
    with pytest.raises(FileTooLargeError):
        ingest_image("fixtures/51mb_image.jpg")

def test_thumbnail_mismatch_detected():
    result = ingest_image("fixtures/manipulated/thumbnail_mismatch.jpg")
    assert result.thumbnail_mismatch is True
```

### Report Generator Tests

**`test_report_generator.py`**
- Mock analysis results; verify JSON output matches schema exactly (validate with `jsonschema`).
- Verify PDF file is non-empty and begins with `%PDF` magic bytes.
- Verify `sha256` in the report matches `ingest_image()` output for the same file.
- Verify `analysis_timestamp` is in ISO 8601 UTC format.

### Auth & Rate Limiter Tests

**`test_auth.py`**
- Verify expired JWTs are rejected.
- Verify tampered JWTs (modified payload) are rejected.
- Verify missing scope (`report:delete` token used for `/analyze`) is rejected with 403.
- Verify valid tokens pass.

**`test_rate_limiter.py`**
- Use a mock Redis client.
- Verify 60th request in a window succeeds.
- Verify 61st request returns 429.
- Verify window resets after 60 s.

---

## Integration Tests

Integration tests spin up the full FastAPI app (no network; using `httpx.AsyncClient` with `app=app`) and exercise real I/O against a GPU or CPU fallback.

**`test_api_analyze.py`**

```python
@pytest.mark.asyncio
async def test_full_analysis_authentic():
    async with AsyncClient(app=app, base_url="http://test") as client:
        with open("fixtures/authentic/pristine_landscape.jpg", "rb") as f:
            response = await client.post(
                "/v1/analyze",
                files={"file": f},
                headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            )
    assert response.status_code == 200
    data = response.json()
    assert data["report"]["manipulation_detected"] is False
    assert data["report"]["overall_confidence"] < 0.4

@pytest.mark.asyncio
async def test_full_analysis_spliced():
    async with AsyncClient(app=app, base_url="http://test") as client:
        with open("fixtures/manipulated/splice_001.jpg", "rb") as f:
            response = await client.post("/v1/analyze", ...)
    assert response.status_code == 200
    data = response.json()
    assert data["report"]["manipulation_detected"] is True
    assert data["report"]["overall_confidence"] > 0.75
    assert len(data["report"]["regions"]) >= 1

@pytest.mark.asyncio
async def test_returns_422_for_corrupt_image():
    ...

@pytest.mark.asyncio
async def test_sha256_in_report_matches_file():
    ...
```

**`test_resilience.py`**

Tests the anti-forensic resilience check end-to-end:
- Submit a highly manipulated image.
- Verify the response contains results for quality levels 70, 85, 95 in the report (when `resilience_test=true`).
- Submit an authentic image and verify `anti_forensic_warning` is `false`.

**`test_pipeline_end_to_end.py`**

Full pipeline test: ingest → feature extraction → all 4 models → ensemble → report. Asserts that:
- A known-authentic image produces `manipulation_detected: false`.
- A known-tampered image produces `manipulation_detected: true` with at least one region matching the ground truth mask (IoU > 0.3).
- Execution time on CPU < 15 s, on GPU < 1 s (soft assertion: warning, not failure).

---

## Benchmark Evaluations

Benchmarks are not run in CI (too slow). They run nightly on a dedicated GPU machine and post results to MLflow.

**Benchmark script pattern**:

```python
# eval_ensemble.py
dataset = ForensicDataset("data/processed/test/metadata.jsonl")
results = []

for image, mask in tqdm(dataset):
    report = run_pipeline(image)
    pred_mask = report_to_binary_mask(report, image.shape)
    results.append(compute_metrics(pred_mask, mask))

summary = aggregate_metrics(results)
mlflow.log_metrics(summary)
```

**Target benchmark metrics**:

| Model | Dataset | Pixel F1 | AUC-ROC |
|-------|---------|----------|---------|
| PatchForensic | CASIA v2 test | ≥ 0.80 | ≥ 0.88 |
| MantraNet | DEFACTO test | ≥ 0.77 | ≥ 0.85 |
| SPSL | COVERage test | ≥ 0.88 (pair AUC) | — |
| Inpainting Det. | Synthetic test | ≥ 0.94 (AUC) | ≥ 0.94 |
| **Ensemble** | CASIA + NIST 16 | **≥ 0.84** | **≥ 0.91** |

**Regression gate**: if any model's F1 drops by more than 1 pp from its baseline in `benchmarks/baseline.json`, the nightly run posts an alert to the `#certainaity-alerts` Slack channel and blocks the next deployment.

---

## Test Infrastructure

### CI (GitHub Actions)

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]

jobs:
  unit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -e ".[dev]"
      - run: pytest tests/unit/ -v --tb=short --cov=certainaity --cov-report=xml
      - uses: codecov/codecov-action@v4

  integration:
    runs-on: ubuntu-latest
    env:
      CERTAINAITY_USE_CPU: "1"   # run models on CPU in CI
    steps:
      - uses: actions/checkout@v4
      - run: pip install -e ".[dev]"
      - run: pytest tests/integration/ -v --tb=short -x
```

### Nightly Benchmarks (self-hosted GPU runner)

```yaml
# .github/workflows/benchmark.yml
name: Nightly Benchmark
on:
  schedule: [{ cron: "0 2 * * *" }]   # 2 AM UTC

jobs:
  benchmark:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v4
      - run: python -m certainaity.benchmark.run_all --log-mlflow
```

---

## Test Coverage Targets

| Layer | Target Coverage |
|-------|----------------|
| Feature extractors | 90% |
| Ingestion | 95% |
| Report generator | 85% |
| Auth / rate limiter | 95% |
| API routes | 80% |
| Pipeline orchestration | 75% |
| **Overall** | **≥ 85%** |

Coverage is measured with `pytest-cov`; the threshold is enforced in CI (`--cov-fail-under=85`).

---

## Chain-of-Custody Test Protocol

Because Certainaity may be used in legal proceedings, a specific protocol governs SHA-256 integrity:

1. A dedicated test `test_chain_of_custody.py` runs on every pull request.
2. It submits a fixture image, records the returned SHA-256, then independently computes the SHA-256 from the original file.
3. It then submits the same image again and asserts the SHA-256 is identical.
4. It submits a pixel-modified copy (1 px changed) and asserts the SHA-256 differs.
5. It inspects the generated PDF and asserts the embedded SHA-256 matches.

This test is marked `@pytest.mark.critical` and cannot be skipped.
