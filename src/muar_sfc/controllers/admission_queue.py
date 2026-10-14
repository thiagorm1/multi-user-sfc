"""Bounded retries: queued requests are never counted as accepted."""

import heapq
import json
from dataclasses import dataclass, field


@dataclass
class AdmissionRequest:
    batch: list
    arrival: float
    deadline: float
    ready_at: float
    attempts: int = 0
    status: str = "pending"
    completed_at: float | None = None
    latencies_ms: list[float] = field(default_factory=list)


class AdmissionQueue:
    def __init__(self, enabled=False, max_wait=30.0, retry_interval=1.0, clock=None):
        self.clock = clock
        self.enabled = enabled
        self.max_wait = max_wait
        self.retry_interval = retry_interval
        self.requests = []
        self._pending = []
        self._sequence = 0

    def register(self, batch, now, arrival=None):
        arrival = now if arrival is None else arrival
        request = AdmissionRequest(batch, arrival, arrival + self.max_wait, now)
        self.requests.append(request)
        self._schedule(request)
        return request

    def _schedule(self, request):
        self._sequence += 1
        heapq.heappush(self._pending, (request.ready_at, self._sequence, request))

    def next_ready(self, now):
        if self._pending and self._pending[0][0] <= now:
            return heapq.heappop(self._pending)[2]
        return None

    @property
    def pending(self):
        return len(self._pending)

    @property
    def next_time(self):
        return self._pending[0][0] if self._pending else None

    def attempt(self, request, now, deploy, rollback=None):
        if self.enabled and now > request.deadline:
            request.status = "expired"
        else:
            request.attempts += 1
            success, latencies = deploy(request.batch)
            now = self.clock() if self.clock is not None else now
            if success and self.enabled and now > request.deadline:
                if rollback is None:
                    raise ValueError("Admissão tardia exige rollback físico.")
                rollback(request.batch)
                success = False
            if success:
                request.latencies_ms = latencies
                request.status = (
                    "accepted_immediate" if request.attempts == 1 else "accepted_after_wait"
                )
            elif self.enabled and now < request.deadline:
                request.ready_at = min(now + self.retry_interval, request.deadline)
                self._schedule(request)
                return False
            else:
                request.status = "expired" if self.enabled else "rejected"
        request.completed_at = now
        return True

    def snapshot(self):
        accepted = [r for r in self.requests if r.status.startswith("accepted")]
        latency = [value for r in accepted for value in r.latencies_ms]
        waits = [max(0.0, r.completed_at - r.arrival) for r in accepted]
        count = len(self.requests)
        return {
            "requests": count,
            "pending": self.pending,
            "accepted": len(accepted),
            "accepted_immediate": sum(r.status == "accepted_immediate" for r in accepted),
            "accepted_after_wait": sum(r.status == "accepted_after_wait" for r in accepted),
            "rejected": sum(r.status in ("expired", "rejected") for r in self.requests),
            "acceptance_pct": 100 * len(accepted) / count if count else 0,
            "mean_wait_seconds": sum(waits) / len(waits) if waits else 0,
            "accumulated_latency_ms": sum(latency),
            "mean_latency_ms": sum(latency) / len(latency) if latency else None,
            "records": [
                {
                    "ids": [s.id for s in r.batch],
                    "status": r.status,
                    "arrival": r.arrival,
                    "completed_at": r.completed_at,
                    "attempts": r.attempts,
                    "latencies_ms": r.latencies_ms,
                }
                for r in self.requests
            ],
        }

    def dump(self, path):
        path.write_text(json.dumps(self.snapshot(), indent=2, ensure_ascii=False))
