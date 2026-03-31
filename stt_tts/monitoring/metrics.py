"""
metrics.py — Shared Prometheus instrumentation for all microservices.

Usage in any service:
    from monitoring.metrics import instrument_app
    instrument_app(app, service_name="gateway")

This adds:
  - /metrics endpoint (Prometheus format)
  - Request count, latency histogram, in-flight gauge per endpoint
  - Service info gauge with version label
"""
from __future__ import annotations

import time
from typing import Optional

from fastapi import FastAPI, Request, Response
from prometheus_client import (
    Counter, Gauge, Histogram, Info,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
    REGISTRY,
)


def _safe_metric(cls, name, description, registry, **kwargs):
    """Create a Prometheus metric, returning the existing one on duplicate."""
    try:
        return cls(name, description, registry=registry, **kwargs)
    except ValueError:
        # Already registered — look it up in the registry.
        # Counter adds _total/_created suffixes; strip to find the base name.
        base = name.removesuffix("_total").removesuffix("_created")
        for key, collector in registry._names_to_collectors.items():
            if key == base or key == name:
                return collector
        raise


def instrument_app(
    app: FastAPI,
    service_name: str,
    version: str = "1.0.0",
    registry: Optional[CollectorRegistry] = None,
):
    """Add Prometheus instrumentation to a FastAPI app."""
    reg = registry or REGISTRY
    prefix = service_name.replace("-", "_")

    # ── Metrics ──────────────────────────────────────────────────────────────
    REQUEST_COUNT = _safe_metric(
        Counter,
        f"{prefix}_http_requests_total",
        "Total HTTP requests",
        reg,
        labelnames=["method", "endpoint", "status"],
    )
    REQUEST_LATENCY = _safe_metric(
        Histogram,
        f"{prefix}_http_request_duration_seconds",
        "HTTP request latency in seconds",
        reg,
        labelnames=["method", "endpoint"],
        buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    )
    IN_FLIGHT = _safe_metric(
        Gauge,
        f"{prefix}_http_requests_in_flight",
        "Number of in-flight HTTP requests",
        reg,
    )
    UP = _safe_metric(
        Gauge,
        f"{prefix}_up",
        "Service is up (1) or down (0)",
        reg,
    )
    UP.set(1)

    INFO = _safe_metric(
        Info,
        f"{prefix}_build",
        "Service build info",
        reg,
    )
    INFO.info({"version": version, "service": service_name})

    # ── Middleware ────────────────────────────────────────────────────────────
    @app.middleware("http")
    async def _prometheus_middleware(request: Request, call_next):
        if request.url.path == "/metrics":
            return await call_next(request)

        method = request.method
        # Normalize path to avoid cardinality explosion
        path = request.url.path.rstrip("/") or "/"

        IN_FLIGHT.inc()
        start = time.monotonic()
        try:
            response = await call_next(request)
            status = str(response.status_code)
        except Exception:
            status = "500"
            raise
        finally:
            elapsed = time.monotonic() - start
            IN_FLIGHT.dec()
            REQUEST_COUNT.labels(method=method, endpoint=path, status=status).inc()
            REQUEST_LATENCY.labels(method=method, endpoint=path).observe(elapsed)

        return response

    # ── /metrics endpoint ────────────────────────────────────────────────────
    @app.get("/metrics", include_in_schema=False)
    async def _metrics_endpoint():
        body = generate_latest(reg)
        return Response(content=body, media_type=CONTENT_TYPE_LATEST)

    return {
        "request_count": REQUEST_COUNT,
        "request_latency": REQUEST_LATENCY,
        "in_flight": IN_FLIGHT,
        "up": UP,
    }
