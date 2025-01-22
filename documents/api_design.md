# ForenScope — API Design

## Overview

ForenScope exposes a REST API built with FastAPI (Python 3.11). The API is versioned under `/v1/`. Authentication uses JWT bearer tokens issued per organization. All endpoints return JSON; image upload uses `multipart/form-data`.

**Base URL (local dev)**: `http://localhost:8000`  
**Base URL (production)**: `https://api.forenscope.io` (TBD)

---

## Authentication

### Token Issuance

Tokens are issued out-of-band (admin CLI or web dashboard — out of scope for v1.0). Each token encodes:

```json
{
  "sub": "org_acme",
  "iat": 1700000000,
  "exp": 1731536000,
  "scopes": ["analyze", "report:read", "report:delete"]
}
```

Tokens are signed with RS256 (2048-bit RSA key). The public key is served at `/v1/.well-known/jwks.json`.

### Request Header

```
Authorization: Bearer <token>
```

Requests without a valid token receive `401 Unauthorized`.

---

## Endpoints

### `POST /v1/analyze`

Submit an image for forensic analysis.

**Request**

```
Content-Type: multipart/form-data

Fields:
  file         (required) Image file. JPEG, PNG, TIFF, or WebP. Max 50 MB.
  options      (optional) JSON string with analysis options (see below).
```

**Options object**

```json
{
  "resilience_test": true,         // run re-compression resilience test (default: true)
  "output_formats": ["json", "pdf"], // what artifacts to generate (default: ["json"])
  "webhook_url": null,             // if provided, POST result here when done (for large images)
  "priority": "normal"             // "normal" | "high" (high consumes extra quota)
}
```

**Synchronous response** (image ≤ 12 MP, estimated time ≤ 5 s)

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "request_id": "req_7a8f2b...",
  "status": "complete",
  "report": {
    "file_name": "evidence_001.jpg",
    "sha256": "a1b2c3...",
    "analysis_timestamp": "2025-01-22T09:14:33Z",
    "manipulation_detected": true,
    "overall_confidence": 0.92,
    "regions": [
      {
        "region_id": 1,
        "bbox": [100, 200, 300, 400],
        "type": "splicing",
        "confidence": 0.96,
        "evidence": "CFA mismatch (p=0.003), noise variance discontinuity"
      }
    ],
    "anti_forensic_warning": false,
    "models_used": ["PatchForensic", "MantraNet", "SPSL", "InpaintingDetector"],
    "execution_time_ms": 412
  },
  "artifacts": {
    "json_url": "/v1/reports/req_7a8f2b.../report.json",
    "pdf_url":  "/v1/reports/req_7a8f2b.../report.pdf"
  }
}
```

**Asynchronous response** (image > 12 MP or `priority=high` queue depth > 0)

```http
HTTP/1.1 202 Accepted
Content-Type: application/json

{
  "request_id": "req_9c1d4e...",
  "status": "queued",
  "estimated_completion_s": 12,
  "poll_url": "/v1/analyze/req_9c1d4e.../status"
}
```

**Error responses**

| HTTP Code | `error_code` | Description |
|-----------|-------------|-------------|
| 400 | `invalid_file_type` | File type not supported |
| 400 | `file_too_large` | File exceeds 50 MB |
| 400 | `image_too_small` | Image shorter side < 64 px |
| 401 | `unauthorized` | Missing or invalid token |
| 403 | `quota_exceeded` | Monthly request quota reached |
| 422 | `corrupt_image` | Image cannot be decoded |
| 429 | `rate_limited` | >60 requests/minute |
| 500 | `internal_error` | Unexpected server error |
| 503 | `models_unavailable` | GPU unavailable, try again |

```http
HTTP/1.1 400 Bad Request
Content-Type: application/json

{
  "error_code": "invalid_file_type",
  "message": "Only JPEG, PNG, TIFF, and WebP are accepted.",
  "request_id": "req_err_abc123"
}
```

---

### `GET /v1/analyze/{request_id}/status`

Poll the status of an asynchronous analysis.

**Response**

```json
{
  "request_id": "req_9c1d4e...",
  "status": "processing",   // "queued" | "processing" | "complete" | "failed"
  "progress_pct": 45,
  "estimated_remaining_s": 7
}
```

When `status == "complete"`, the `report` and `artifacts` fields are included (same schema as the synchronous 200 response).

---

### `GET /v1/reports/{request_id}/{filename}`

Download a generated artifact (JSON or PDF).

**Authorization**: token must have `report:read` scope.  
**Response**: `Content-Type: application/json` or `application/pdf` depending on filename.  
**Cache-Control**: `private, max-age=86400` (artifacts immutable for their lifetime).

---

### `DELETE /v1/reports/{request_id}`

Delete all artifacts for a request (chain-of-custody: caller is responsible for maintaining copies).

**Authorization**: token must have `report:delete` scope.  
**Response**: `204 No Content`

---

### `GET /v1/quota`

Return current quota usage for the authenticated organization.

```json
{
  "org": "org_acme",
  "plan": "professional",
  "period": "2025-01",
  "requests_used": 142,
  "requests_limit": 1000,
  "reset_date": "2025-02-01"
}
```

---

### `GET /v1/health`

Public endpoint (no auth). Returns server health and model load status.

```json
{
  "status": "ok",
  "version": "1.0.0",
  "models_loaded": {
    "PatchForensic": true,
    "MantraNet": true,
    "SPSL": true,
    "InpaintingDetector": true
  },
  "gpu_available": true,
  "queue_depth": 0
}
```

---

## Rate Limiting

Rate limits are enforced per API token using a sliding window (Redis-backed):

| Plan | Requests/minute | Requests/month |
|------|----------------|----------------|
| Free | 10 | 50 |
| Professional | 60 | 1,000 |
| Enterprise | 300 | Unlimited |

When rate-limited, the response includes:
```
Retry-After: 14
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1706000414
```

---

## Webhook Delivery

If `webhook_url` is specified in the analysis options, the server POSTs the same JSON body as the synchronous 200 response to that URL when analysis is complete. Webhook delivery is retried up to 5 times with exponential backoff (1 s, 2 s, 4 s, 8 s, 16 s). A shared secret (set in the org dashboard) is sent as the `X-Forenscope-Signature: sha256=<hmac>` header so the receiver can verify authenticity.

---

## FastAPI Route Definitions

```python
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from forenscope.auth import require_scope
from forenscope.schemas import AnalyzeOptions, AnalyzeResponse

router = APIRouter(prefix="/v1")

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile = File(...),
    options: str = "{}",
    token: dict = Depends(require_scope("analyze")),
):
    ...

@router.get("/analyze/{request_id}/status")
async def poll_status(
    request_id: str,
    token: dict = Depends(require_scope("analyze")),
):
    ...

@router.get("/reports/{request_id}/{filename}")
async def download_report(
    request_id: str,
    filename: str,
    token: dict = Depends(require_scope("report:read")),
):
    ...

@router.delete("/reports/{request_id}", status_code=204)
async def delete_report(
    request_id: str,
    token: dict = Depends(require_scope("report:delete")),
):
    ...

@router.get("/quota")
async def get_quota(token: dict = Depends(require_scope("analyze"))):
    ...

@router.get("/health", include_in_schema=False)
async def health():
    ...
```

---

## OpenAPI / Swagger

FastAPI auto-generates OpenAPI 3.1 docs. Available at:
- `/docs` — Swagger UI
- `/redoc` — ReDoc
- `/openapi.json` — raw schema

The schema is exported and committed to `docs/openapi.json` on each release.

---

## Versioning Policy

- The current API version is `v1`.
- Breaking changes (removed fields, changed types, removed endpoints) will be released under `v2` with 6-month parallel support for `v1`.
- Additive changes (new optional fields, new endpoints) are backwards-compatible and released in `v1` without a version bump.
- Deprecation notices are added to the OpenAPI schema as `x-deprecated: true` on affected endpoints/fields.
