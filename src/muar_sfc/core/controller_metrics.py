"""Low-frequency Linux CPU telemetry, outside the admission decision path."""

import os
import threading
import time
from pathlib import Path


class ControllerCPUSampler:
    """Process CPU may exceed 100%: 100% means one logical core occupied."""

    def __init__(self, interval=0.5):
        if interval <= 0:
            raise ValueError("Sampling interval must be positive")
        self.interval = interval
        self.samples = []
        self.supported = Path("/proc/self/stat").exists()
        self._stop = threading.Event()
        self._thread = None

    @staticmethod
    def _process_ticks(pid):
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return int(fields[11]) + int(fields[12])

    def _snapshot(self):
        processes = {}
        for path in Path("/proc").iterdir():
            if not path.name.isdecimal():
                continue
            try:
                if path.name == str(os.getpid()) or (path / "comm").read_text().startswith(
                    "ollama"
                ):
                    processes[int(path.name)] = self._process_ticks(path.name)
            except (OSError, ValueError, IndexError):
                continue
        ticks = list(map(int, Path("/proc/stat").read_text().splitlines()[0].split()[1:9]))
        return time.perf_counter(), processes, sum(ticks), ticks[3] + ticks[4]

    def _run(self):
        previous = self._snapshot()
        hz = os.sysconf("SC_CLK_TCK")
        while not self._stop.wait(self.interval):
            current = self._snapshot()
            elapsed = current[0] - previous[0]
            delta = {
                pid: max(0, ticks - previous[1][pid]) / hz / elapsed * 100
                for pid, ticks in current[1].items()
                if pid in previous[1]
            }
            total = current[2] - previous[2]
            self.samples.append(
                {
                    "elapsed_s": elapsed,
                    "controller_cpu_percent": delta.get(os.getpid(), 0),
                    "ollama_cpu_percent": sum(v for pid, v in delta.items() if pid != os.getpid()),
                    "host_busy_percent": 100 * (1 - (current[3] - previous[3]) / total)
                    if total
                    else 0,
                }
            )
            previous = current

    def __enter__(self):
        if self.supported:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *_args):
        self._stop.set()
        if self._thread:
            self._thread.join()
