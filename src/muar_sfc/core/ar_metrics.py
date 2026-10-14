"""Latency summaries and optional measured AR frame telemetry (no invented display data)."""

import math


def distribution(values):
    values = sorted(float(v) for v in values)
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Latencies must be finite and nonnegative")
    mean = sum(values) / len(values) if values else None

    def percentile(q):
        if not values:
            return None
        position = (len(values) - 1) * q
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        return values[lower] + (values[upper] - values[lower]) * (position - lower)

    return {
        "samples": len(values),
        "mean": mean,
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": max(values, default=None),
        "stddev": (
            math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)) if values else None
        ),
    }


def ar_frame_summary(frames=None, budget_ms=None):
    """Frames use motion_s/display_s in the same monotonic clock, or dropped=True.

    This is observed motion-to-photon latency, not the simulated SFC delay.
    A budget must be explicitly chosen for the target device/application.
    """
    if budget_ms is not None and (not math.isfinite(budget_ms) or budget_ms <= 0):
        raise ValueError("AR budget must be positive and finite")
    if frames is None:
        return {
            "status": "not_measured",
            "motion_to_photon_ms": distribution([]),
            "latency_jitter_ms": distribution([]),
            "display_interval_ms": distribution([]),
            "dropped_frames": None,
            "budget_ms": budget_ms,
            "budget_exceeded_frames": None,
            "comfort_validated": False,
        }
    latencies, displays, dropped, expected = [], [], 0, 0
    for frame in frames:
        expected += 1
        if frame.get("dropped", False):
            dropped += 1
            continue
        motion, display = float(frame["motion_s"]), float(frame["display_s"])
        if not math.isfinite(motion) or not math.isfinite(display) or display < motion:
            raise ValueError("Frame timestamps must be finite and display >= motion")
        latencies.append((display - motion) * 1000)
        if displays and display < displays[-1]:
            raise ValueError("Frames must be ordered by display timestamp")
        displays.append(display)
    return {
        "status": "measured" if expected else "no_samples",
        "motion_to_photon_ms": distribution(latencies),
        "latency_jitter_ms": distribution(
            [abs(b - a) for a, b in zip(latencies[:-1], latencies[1:], strict=True)]
        ),
        "display_interval_ms": distribution(
            [(b - a) * 1000 for a, b in zip(displays[:-1], displays[1:], strict=True)]
        ),
        "expected_frames": expected,
        "dropped_frames": dropped,
        "dropped_frame_percent": 100 * dropped / expected if expected else None,
        "budget_ms": budget_ms,
        "budget_exceeded_frames": (
            sum(v > budget_ms for v in latencies) if budget_ms is not None else None
        ),
        "comfort_validated": False,
    }
