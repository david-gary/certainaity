# REST API Reference

Base URL: `https://your-host/v1`

All protected endpoints require `Authorization: Bearer <jwt>`.

---

## `POST /v1/analyze`

Submit an image for forensic analysis.

**Auth required**: Yes  
**Rate limited**: Yes (default 60 req/min/IP; configurable via `FORENSCOPE_RATE_LIMIT_PER_MINUTE`)

### Request

`multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | binary | Yes | Image file (JPEG, PNG, TIFF, WebP; max 50 MB) |

### Response `202 Accepted`

```json
{
  "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "poll_url": "/v1/jobs/3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "message": "Analysis queued"
}
```

### Error responses

| Status | Cause |
|--------|-------|
| `400` | Invalid image format, image too small/large, corrupt file |
| `401` | Missing or invalid JWT |
| `413` | File exceeds `FORENSCOPE_MAX_FILE_BYTES` (default 50 MB) |
| `429` | Rate limit exceeded |

---

## `GET /v1/jobs/{job_id}`

Poll job status and stage metadata.

**Auth required**: Yes

### Response

```json
{
  "job_id": "3fa85f64-...",
  "state": "STARTED",
  "stage": "inference"
}
```

| Field | Values |
|-------|--------|
| `state` | `PENDING` · `STARTED` · `SUCCESS` · `FAILURE` · `RETRY` |
| `stage` | `ingest` · `features` · `inference` · `resilience` · `report` |

When `state` is `SUCCESS`:

```json
{
  "job_id": "3fa85f64-...",
  "state": "SUCCESS",
  "result": {
    "job_id": "3fa85f64-...",
    "sha256": "a3f4...",
    "overall_confidence": 0.87
  }
}
```

---

## `GET /v1/reports/{job_id}/json`

Retrieve the full JSON analysis report.

**Auth required**: Yes

### Response `200 application/json`

```json
{
  "job_id": "3fa85f64-...",
  "file_name": "photo.jpg",
  "sha256": "a3f4...",
  "analysis_timestamp": "2026-03-01T10:23:45.123456+00:00",
  "manipulation_detected": true,
  "overall_confidence": 0.87,
  "regions": [
    {
      "bbox": [120, 80, 200, 150],
      "type": "splicing",
      "confidence": 0.92,
      "evidence": "CFA anomaly=0.81; ELA=0.74"
    }
  ],
  "anti_forensic_warning": false,
  "models_used": ["PatchForensic", "MantraNet", "SPSL", "InpaintingDetector"],
  "execution_time_ms": 2341
}
```

---

## `GET /v1/reports/{job_id}/pdf`

Download the PDF forensic report.

**Auth required**: Yes

**Response**: `200 application/pdf` (binary)

---

## `GET /v1/health`

Liveness probe. No authentication required.

**Response**: `200 {"status": "ok"}`

---

## `GET /metrics`

Prometheus text-format metrics scrape endpoint. Not authenticated; excluded from OpenAPI schema.

See [Observability](../architecture/overview.md#observability) for the full metric list.
