"""Celery application factory.

A single ``celery_app`` instance is imported by tasks and by the CLI
entrypoint (``celery -A certainaity.worker.app worker …``).
"""

from __future__ import annotations

from celery import Celery

from certainaity.config import get_settings


def _make_celery() -> Celery:
    settings = get_settings()
    app = Celery(
        "certainaity",
        broker=settings.redis_url,
        backend=settings.redis_url,
    )
    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
        task_default_queue="analysis",
        task_queues={
            "analysis": {
                "exchange": "analysis",
                "routing_key": "analysis",
            },
        },
        # Acknowledge only after the task completes so a crash triggers retry.
        task_acks_late=True,
        # One task at a time keeps GPU memory usage bounded on the worker.
        worker_prefetch_multiplier=1,
        # Store results for 24 hours, then Redis evicts them.
        result_expires=86_400,
    )
    return app


celery_app = _make_celery()
