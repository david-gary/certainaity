# Contributing

## Development setup

```bash
git clone https://github.com/david-gary/forenscope.git
cd forenscope
pip install -e ".[dev,api]"
```

## Running tests

```bash
# Unit tests with coverage
pytest tests/unit/ -v --cov=forenscope --cov-fail-under=85

# Integration tests (no GPU required)
FORENSCOPE_USE_CPU=true \
FORENSCOPE_OUTPUT_DIR=/tmp/forenscope_test \
FORENSCOPE_JWT_PUBLIC_KEY_PATH=/tmp/nonexistent.pem \
pytest tests/integration/ -v

# Full CI equivalent
ruff check src/ tests/ && ruff format --check src/ tests/ && mypy src/
```

## Frontend

```bash
cd frontend
npm install
npm run dev        # Vite dev server on :3000, proxies /v1 to localhost:8000
npm run typecheck
npm run lint
npm run build
```

## Branch strategy

| Branch | Purpose |
|--------|---------|
| `main` | Production-ready; protected; requires passing CI |
| `feature/*` | New features; merge to main via PR |
| `fix/*` | Bug fixes; merge to main via PR |

## Commit style

Use the imperative present tense:

```
add ELA feature extractor
fix resilience test confidence calculation
update deployment guide for K8s HPA
```

No `Co-Authored-By` trailers.

## Adding a model

1. Create `src/forenscope/models/yourmodel.py` extending `ForensicModel`.
2. Implement `_load_weights()` and `_forward()`.
3. Add a `ModelName` enum variant in `base.py`.
4. Register the model in `Ensemble.__init__()` with a default weight.
5. Add unit tests in `tests/unit/test_models.py`.
6. Update `_DEFAULT_WEIGHTS` in `ensemble.py` so all weights sum to 1.

## Adding a feature extractor

1. Create `src/forenscope/features/yourfeature.py`.
2. Implement a `compute_*` function returning `(H, W) float32 ndarray in [0, 1]`.
3. Register it in `_extract_features()` in `tasks.py`.
4. Add unit tests in `tests/unit/`.

## Release process

1. Update `CHANGELOG.md` under `## [X.Y.Z] — YYYY-MM-DD`.
2. Bump `version` in `pyproject.toml`.
3. Merge to main.
4. Tag: `git tag vX.Y.Z && git push origin vX.Y.Z`.
5. The `publish.yml` workflow builds and pushes Docker images to GHCR and creates the GitHub Release automatically.
