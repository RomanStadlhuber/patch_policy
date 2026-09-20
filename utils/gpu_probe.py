"""Sample GPU memory and utilization densely, report it sparsely.

nvidia-smi reports utilization over its own recent window, so a 5 s poll
averages short stalls away and a spot check misses peaks entirely. This keeps
one nvidia-smi streaming at a high rate and prints an aggregate per window,
which is what makes a peak visible without flooding a log.

Run it beside a training job:

    uv run python -m utils.gpu_probe --report-s 1.0 --out probe.npz

or use it from Python:

    with GpuProbe() as probe:
        train()
    print(probe.peak_memory_mib)
"""

import argparse
import signal
import subprocess
import sys
import threading
import time
from typing import List, Optional, TextIO

import numpy as np

QUERY = "memory.used,utilization.gpu"


class GpuProbe:
    """Streams samples from one nvidia-smi process into arrays."""

    def __init__(
        self,
        interval_ms: int = 100,
        gpu_index: int = 0,
        report_s: Optional[float] = None,
        stream: Optional[TextIO] = None,
    ) -> None:
        self.interval_ms = interval_ms
        self.gpu_index = gpu_index
        self.report_s = report_s
        self.stream = stream if stream is not None else sys.stdout

        self.timestamps: List[float] = []
        self.memory_mib: List[int] = []
        self.utilization: List[int] = []

        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._started = 0.0
        self._window_start = 0

    # -- lifecycle ------------------------------------------------------
    def start(self) -> "GpuProbe":
        self._proc = subprocess.Popen(
            [
                "nvidia-smi",
                f"--id={self.gpu_index}",
                f"--query-gpu={QUERY}",
                "--format=csv,noheader,nounits",
                f"--loop-ms={self.interval_ms}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self._started = time.perf_counter()
        self._thread = threading.Thread(target=self._consume, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def __enter__(self) -> "GpuProbe":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # -- sampling -------------------------------------------------------
    def _consume(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            parts = line.strip().split(",")
            if len(parts) != 2:
                continue
            try:
                memory, util = int(parts[0]), int(parts[1])
            except ValueError:
                # nvidia-smi prints "[N/A]" for a field it cannot read
                continue
            self.timestamps.append(time.perf_counter() - self._started)
            self.memory_mib.append(memory)
            self.utilization.append(util)
            if self.report_s is not None:
                self._maybe_report()

    def _maybe_report(self) -> None:
        window = self.timestamps[self._window_start :]
        if not window or window[-1] - window[0] < self.report_s:
            return
        end = len(self.timestamps)
        memory = np.asarray(self.memory_mib[self._window_start : end])
        util = np.asarray(self.utilization[self._window_start : end])
        print(
            f"[{window[-1]:7.1f}s] n={len(memory):4d}  "
            f"vram mean {memory.mean():6.0f} max {memory.max():6.0f} MiB  "
            f"util mean {util.mean():3.0f}% max {util.max():3.0f}% "
            f"p50 {np.median(util):3.0f}%",
            file=self.stream,
            flush=True,
        )
        self._window_start = end

    # -- results --------------------------------------------------------
    @property
    def peak_memory_mib(self) -> int:
        return max(self.memory_mib) if self.memory_mib else 0

    @property
    def mean_utilization(self) -> float:
        return float(np.mean(self.utilization)) if self.utilization else 0.0

    def summary(self) -> str:
        if not self.memory_mib:
            return "no samples"
        memory = np.asarray(self.memory_mib)
        util = np.asarray(self.utilization)
        return (
            f"{len(memory)} samples over {self.timestamps[-1]:.1f}s\n"
            f"  vram  mean {memory.mean():.0f}  p50 {np.median(memory):.0f}  "
            f"max {memory.max()} MiB\n"
            f"  util  mean {util.mean():.0f}%  p50 {np.median(util):.0f}%  "
            f"max {util.max()}%  time at >=95%: {(util >= 95).mean() * 100:.0f}%"
        )

    def save(self, path: str) -> None:
        np.savez(
            path,
            timestamps=np.asarray(self.timestamps),
            memory_mib=np.asarray(self.memory_mib),
            utilization=np.asarray(self.utilization),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval-ms", type=int, default=100)
    parser.add_argument("--report-s", type=float, default=1.0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--out", type=str, default=None, help="write samples to .npz")
    parser.add_argument(
        "--duration-s", type=float, default=None, help="stop after this long"
    )
    args = parser.parse_args()

    probe = GpuProbe(
        interval_ms=args.interval_ms, gpu_index=args.gpu, report_s=args.report_s
    )
    # a background job started from a non-interactive shell inherits SIGINT
    # ignored, so a launcher can only stop this with SIGTERM
    done = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: done.set())

    probe.start()
    try:
        done.wait(timeout=args.duration_s)
    except KeyboardInterrupt:
        pass
    finally:
        probe.stop()
        print(probe.summary(), flush=True)
        if args.out:
            probe.save(args.out)
            print(f"samples written to {args.out}", flush=True)


if __name__ == "__main__":
    main()
