"""Emulated network link: a token-bucket delay layered over a real (fast) copy.

delivery_time(chunk) = max(real_copy_done, start_tx + bytes/rate + rtt),  start_tx = max(now, link_free_at)
link_free_at = start_tx + bytes/rate          (the wire is busy for bytes/rate; rtt is propagation, overlappable)

Chunks sent back to back therefore achieve goodput == rate regardless of rtt, and a single transfer of B bytes
completes after B/rate + rtt.  Sleeping is done with sleep-then-spin so sub-millisecond deadlines are honoured.

Usage (GPU):
    link = LinkEmulator(rate_gbps=10, rtt_s=0.0005)
    t = link.send(nbytes, copy_fn)            # copy_fn() performs the real copy and returns when it is complete
CPU unit test: python link_emulator.py --test        (fake clock; 10 buffer sizes; goodput within 5%)
"""
from __future__ import annotations
import argparse, json, time
from dataclasses import dataclass, field


@dataclass
class Delivery:
    nbytes: int
    t_start: float
    t_real_done: float
    t_delivered: float

    @property
    def wall(self) -> float:
        return self.t_delivered - self.t_start


@dataclass
class LinkEmulator:
    rate_gbps: float
    rtt_s: float = 0.0
    clock: callable = time.perf_counter
    sleep: callable = time.sleep
    spin_window_s: float | None = None     # spin (busy-wait) for the last part of each wait; None = calibrate
    link_free_at: float = field(default=0.0, init=False)
    log: list = field(default_factory=list, init=False)
    sleep_overshoot_s: float = field(default=0.0, init=False)

    def __post_init__(self):
        if self.spin_window_s is None:
            self.spin_window_s, self.sleep_overshoot_s = calibrate_sleep(self.clock, self.sleep)

    @property
    def bytes_per_s(self) -> float:
        return self.rate_gbps * 1e9 / 8

    def _wait_until(self, t: float):
        remaining = t - self.clock()
        if remaining > self.spin_window_s:
            self.sleep(remaining - self.spin_window_s)
        while self.clock() < t:
            pass

    def submit(self, nbytes: int, copy_fn=None) -> Delivery:
        """Enqueue one chunk on the emulated wire and perform the real copy now. Does NOT wait for emulated delivery.
        The sender may submit the next chunk immediately (pipelining); the wire serialises chunks, rtt is paid once."""
        t0 = self.clock()
        start_tx = max(t0, self.link_free_at)
        self.link_free_at = start_tx + nbytes / self.bytes_per_s
        t_emulated = self.link_free_at + self.rtt_s
        if copy_fn is not None:
            copy_fn()
        t_real = self.clock()
        d = Delivery(nbytes, t0, t_real, max(t_real, t_emulated))
        self.log.append(d)
        return d

    def wait(self, d: Delivery) -> Delivery:
        """Block (receiver side) until the chunk is deliverable."""
        if d.t_delivered > self.clock():
            self._wait_until(d.t_delivered)
        return d

    def send(self, nbytes: int, copy_fn=None) -> Delivery:
        """Blocking convenience: submit + wait."""
        return self.wait(self.submit(nbytes, copy_fn))

    def send_chunks(self, chunk_sizes: list[int], copy_fns: list | None = None) -> list[Delivery]:
        """Pipelined: submit all chunks back to back, then wait for the last one."""
        ds = [self.submit(n, None if copy_fns is None else copy_fns[i]) for i, n in enumerate(chunk_sizes)]
        self.wait(ds[-1])
        return ds

    def reset(self):
        self.link_free_at = 0.0
        self.log.clear()


def calibrate_sleep(clock=time.perf_counter, sleep=time.sleep, n=20) -> tuple[float, float]:
    """Measure how much sleep(1 ms) overshoots on this OS; spin for 2x the worst case (capped at 20 ms)."""
    worst = 0.0
    for _ in range(n):
        t0 = clock(); sleep(0.001); worst = max(worst, clock() - t0 - 0.001)
    return min(0.02, max(0.0002, 2 * worst)), worst


class FakeClock:
    """Deterministic clock for unit tests: time only advances through sleep() or tick()."""
    def __init__(self):
        self.t = 0.0
    def now(self):
        self.t += 1e-9                       # every read costs 1 ns so spin loops terminate
        return self.t
    def sleep(self, s):
        self.t += max(0.0, s)
    def tick(self, s):
        self.t += s


def _goodput_check(rate_gbps=10.0, rtt_s=0.0005, sizes=None, tol=0.05, copy_time_s=0.0) -> dict:
    sizes = sizes or [1 << k for k in range(17, 27)]          # 128 KB .. 64 MB (10 sizes)
    fc = FakeClock()
    link = LinkEmulator(rate_gbps, rtt_s, clock=fc.now, sleep=fc.sleep, spin_window_s=0.0)
    rows = []
    for n in sizes:
        link.reset()
        d = link.send(n, copy_fn=(lambda: fc.tick(copy_time_s)) if copy_time_s else None)
        achieved_gbps = n * 8 / (d.wall - rtt_s) / 1e9          # rate component: exclude the constant rtt
        rows.append({"bytes": n, "wall_s": d.wall, "achieved_gbps_excl_rtt": achieved_gbps,
                     "effective_gbps_incl_rtt": n * 8 / d.wall / 1e9,
                     "within_tol": abs(achieved_gbps - rate_gbps) / rate_gbps <= tol})
    # back-to-back chunks: aggregate goodput must also match the rate, rtt included only once
    link.reset()
    chunks = [4 << 20] * 36
    ds = link.send_chunks(chunks)
    agg = sum(chunks) * 8 / (ds[-1].t_delivered - ds[0].t_start - rtt_s) / 1e9
    return {"rate_gbps": rate_gbps, "rtt_s": rtt_s, "rows": rows, "all_within_tol": all(r["within_tol"] for r in rows),
            "pipelined_36_chunks_gbps": agg, "pipelined_within_tol": abs(agg - rate_gbps) / rate_gbps <= tol}


def test_goodput_fake_clock():
    for rate in (1, 5, 10, 25, 100):
        for rtt in (0.0005, 0.005):
            res = _goodput_check(rate, rtt)
            assert res["all_within_tol"], res
            assert res["pipelined_within_tol"], res


def test_real_copy_slower_than_link_dominates():
    fc = FakeClock()
    link = LinkEmulator(100.0, 0.0, clock=fc.now, sleep=fc.sleep, spin_window_s=0.0)
    d = link.send(1 << 20, copy_fn=lambda: fc.tick(1.0))      # 1 MB at 100 Gbps = 84 us, but the copy takes 1 s
    assert d.wall >= 1.0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="fake-clock unit check")
    ap.add_argument("--smoke", action="store_true", help="real-clock check on CPU with small buffers")
    a = ap.parse_args()
    if a.test or a.smoke:
        print(json.dumps({f"{r}gbps": _goodput_check(r, 0.0005)["all_within_tol"] for r in (1, 5, 10, 25, 100)}))
    if a.smoke:
        # real clock, no real copy: how precise is sleep+spin on this machine?
        out = {}
        for rate in (1, 10, 100):
            link = LinkEmulator(rate, 0.0005)
            n = 10 << 20
            d = link.send(n)
            out[f"{rate}gbps_10MB"] = {"expected_s": n * 8 / (rate * 1e9) + 0.0005, "wall_s": d.wall,
                                       "err_pct": 100 * (d.wall - (n * 8 / (rate * 1e9) + 0.0005)) / (n * 8 / (rate * 1e9)),
                                       "spin_window_ms": link.spin_window_s * 1e3, "sleep_overshoot_ms": link.sleep_overshoot_s * 1e3}
            ds = link.send_chunks([n // 36] * 36)
            tot = (ds[-1].t_delivered - ds[0].t_start)
            out[f"{rate}gbps_10MB_36chunks"] = {"expected_s": n * 8 / (rate * 1e9) + 0.0005, "wall_s": tot,
                                                "err_pct": 100 * (tot - (n * 8 / (rate * 1e9) + 0.0005)) / (n * 8 / (rate * 1e9))}
        print(json.dumps(out, indent=1))
