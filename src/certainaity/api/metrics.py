"""Prometheus metrics for the Certainaity API.

All counters and histograms are module-level singletons; they accumulate
across the lifetime of the process and are scraped at GET /metrics.
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram

REQUESTS_TOTAL = Counter(
    "certainaity_http_requests_total",
    "Total HTTP requests handled by the API",
    ["method", "endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "certainaity_http_request_duration_seconds",
    "End-to-end HTTP request latency in seconds",
    ["endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

JOBS_SUBMITTED = Counter(
    "certainaity_jobs_submitted_total",
    "Total analysis jobs successfully enqueued to the Celery worker",
)

JOBS_REJECTED = Counter(
    "certainaity_jobs_rejected_total",
    "Total image submissions rejected before queuing",
    ["reason"],
)
