from __future__ import annotations

import threading
import time
from collections import deque

from tqdm import tqdm


def duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class Progress:
    def __init__(
        self, metrics, workers: int, batch: int, interval: float, window: float, title: str
    ):
        self.metrics = metrics
        self.workers, self.batch = workers, batch
        self.interval, self.window = interval, window
        self.samples = deque([(metrics.started, 0)])
        self.stop = threading.Event()
        self.bar = tqdm(
            total=metrics.total,
            initial=metrics.skipped,
            desc=title,
            unit="img",
            dynamic_ncols=True,
            mininterval=interval,
        )
        self.thread = threading.Thread(target=self._loop, name="solo-progress", daemon=True)
        self.thread.start()

    def _refresh(self):
        now = time.perf_counter()
        with self.metrics.lock:
            new = self.metrics.success
            done = self.metrics.skipped + new
            elapsed = now - self.metrics.started
            failed, pending = self.metrics.failed, self.metrics.pending
        self.samples.append((now, new))
        while len(self.samples) > 2 and self.samples[1][0] < now - self.window:
            self.samples.popleft()
        span = now - self.samples[0][0]
        rolling = (new - self.samples[0][1]) / span if span > 0 else 0
        average = new / elapsed if elapsed > 0 else 0
        eta = (self.metrics.total - done) / average if average > 0 else 0
        self.bar.n = done
        self.bar.set_postfix_str(
            f"{100 * done / max(1, self.metrics.total):.1f}% rolling={rolling:.2f}img/s "
            f"avg={average:.2f}img/s elapsed={duration(elapsed)} "
            f"ETA={duration(eta) if average else '?'} "
            f"total={(elapsed + eta) / 3600:.2f}h queue={pending} "
            f"workers={self.workers} batch={self.batch} errors={failed}",
            refresh=False,
        )
        self.bar.refresh()

    def _loop(self):
        while not self.stop.wait(self.interval):
            self._refresh()

    def close(self):
        self.stop.set()
        self.thread.join()
        self._refresh()
        self.bar.close()
