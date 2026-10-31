"""A stack-sampling profiler for the free-fly viewer's threads (`--profile`): where each thread spends its time.

Every `interval` seconds it looks at the frame each thread of this process is executing (sys._current_frames())
and counts, per thread, the functions on its stack (inclusive) and the innermost one (exclusive). Waiting
counts too: a thread blocked in a socket read for SUMO, or for CARLA, shows the function that waits. `report()`
gives the top functions per thread since the last report, as shares of the samples.
"""

from __future__ import annotations

import collections
import os
import sys
import threading
import time


def thread_cpu_s(native_id: int) -> float | None:
    """CPU time (user + system, seconds) a thread of this process has used, by its native id; None if unknown."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            k32 = ctypes.windll.kernel32
            h = k32.OpenThread(0x0800, False, native_id)  # THREAD_QUERY_LIMITED_INFORMATION
            if not h:
                return None
            try:
                ft = [wintypes.FILETIME() for _ in range(4)]
                if not k32.GetThreadTimes(h, *(ctypes.byref(f) for f in ft)):
                    return None
                return sum((f.dwHighDateTime << 32 | f.dwLowDateTime) for f in ft[2:]) / 1e7
            finally:
                k32.CloseHandle(h)
        with open(f"/proc/self/task/{native_id}/stat", encoding="ascii") as f:
            parts = f.read().rsplit(")", 1)[1].split()
        return (int(parts[11]) + int(parts[12])) / os.sysconf("SC_CLK_TCK")
    except Exception:  # noqa: BLE001 - not available here
        return None


def _where(frame) -> str:
    code = frame.f_code
    return f"{os.path.basename(code.co_filename)}:{code.co_name}"


class StackSampler(threading.Thread):
    def __init__(self, interval: float = 0.005, depth: int = 40):
        super().__init__(daemon=True, name="stack-sampler")
        self.interval, self.depth = interval, depth
        self.stop = False
        self._lock = threading.Lock()
        self._reset()

    def cpu(self) -> dict[str, float]:
        """CPU seconds each thread used since the last call (Python threads only; CARLA's and TraCI's own
        C++ threads are left out), with "process": the whole process. A thread holding Python's lock is
        using CPU, so the threads' sum near 1 s a second means the lock is saturated."""
        now = {}
        for t in threading.enumerate():
            nid = getattr(t, "native_id", None)
            if nid is not None:
                c = thread_cpu_s(nid)
                if c is not None:
                    now[t.name] = c
        now["process"] = time.process_time()
        last = getattr(self, "_cpu_last", {})
        self._cpu_last = now
        return {k: round(v - last.get(k, 0.0), 3) for k, v in now.items()}

    def _reset(self) -> None:
        self.samples: collections.Counter = collections.Counter()           # thread -> samples
        self.inclusive: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        self.exclusive: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)

    def run(self) -> None:
        me = threading.get_ident()
        while not self.stop:
            names = {t.ident: t.name for t in threading.enumerate()}
            frames = sys._current_frames()
            with self._lock:
                for ident, frame in frames.items():
                    if ident == me:
                        continue
                    name = names.get(ident, str(ident))
                    self.samples[name] += 1
                    self.exclusive[name][_where(frame)] += 1
                    seen = set()
                    f, k = frame, 0
                    while f is not None and k < self.depth:
                        w = _where(f)
                        if w not in seen:
                            seen.add(w)
                            self.inclusive[name][w] += 1
                        f, k = f.f_back, k + 1
            del frames
            time.sleep(self.interval)

    def report(self, top: int = 12) -> dict:
        """{thread: {"samples": n, "inclusive": [(function, share)], "exclusive": [...]}} since last report."""
        with self._lock:
            out = {}
            for name, n in self.samples.items():
                out[name] = {"samples": n,
                             "inclusive": [(w, round(c / n, 3)) for w, c in self.inclusive[name].most_common(top)],
                             "exclusive": [(w, round(c / n, 3)) for w, c in self.exclusive[name].most_common(top)]}
            self._reset()
        return out


class CallCounter:
    """Counts the calls the viewer makes to the CARLA server, by kind (--profile): the server handles client
    calls for only a moment each picture, so each costly one puts the pose process's batch a picture late."""

    def __init__(self):
        self._lock = threading.Lock()
        self.calls: collections.Counter = collections.Counter()
        self.commands: collections.Counter = collections.Counter()

    def count(self, kind: str, commands: int = 0) -> None:
        with self._lock:
            self.calls[kind] += 1
            self.commands[kind] += commands

    def take(self) -> dict:
        with self._lock:
            out = {k: (n, self.commands[k]) for k, n in self.calls.most_common()}
            self.calls.clear()
            self.commands.clear()
        return out


class CountingClient:
    """A carla.Client that counts apply_batch / apply_batch_sync calls (and their commands) by the command kinds
    in them; everything else is passed through."""

    def __init__(self, client, counter: CallCounter):
        self._client, self._counter = client, counter

    def _kind(self, name: str, batch) -> str:
        kinds = sorted({type(c).__name__ for c in batch})
        return f"{name}[{'+'.join(kinds)}]"

    def apply_batch(self, batch, *a, **kw):
        self._counter.count(self._kind("apply_batch", batch), len(batch))
        return self._client.apply_batch(batch, *a, **kw)

    def apply_batch_sync(self, batch, *a, **kw):
        self._counter.count(self._kind("apply_batch_sync", batch), len(batch))
        return self._client.apply_batch_sync(batch, *a, **kw)

    def __getattr__(self, name):
        return getattr(self._client, name)
