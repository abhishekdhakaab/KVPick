"""G7.1: real GPU0 -> GPU1 copy bandwidth for 100 MB and 1 GB (10 reps, CUDA events)."""
import json, torch
from common import Result
r = Result("G7.1")
if not (torch.cuda.is_available() and torch.cuda.device_count() >= 2):
    r.note("fewer than 2 GPUs visible"); r.save("NOT_RUN", name="G7_p2p_copy"); print("NOT_RUN"); raise SystemExit
peer = torch.cuda.can_device_access_peer(0, 1)
rows = []
for mb in (100, 1000):
    n = mb << 20; src = torch.empty(n, dtype=torch.uint8, device="cuda:0").fill_(1); dst = torch.empty(n, dtype=torch.uint8, device="cuda:1")
    ts = []
    for i in range(12):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(0); torch.cuda.synchronize(1); s.record(); dst.copy_(src); e.record(); torch.cuda.synchronize(0); torch.cuda.synchronize(1)
        if i >= 2: ts.append(s.elapsed_time(e) / 1e3)
    med = sorted(ts)[len(ts) // 2]
    rows.append({"MB": mb, "median_ms": med * 1e3, "GBps": n / med / 1e9, "gbps": n * 8 / med / 1e9})
    print(rows[-1])
r.d["measurements"] = {"peer_access": peer, "gpu0": torch.cuda.get_device_name(0), "gpu1": torch.cuda.get_device_name(1), "rows": rows}
print("saved", r.save("INFO", name="G7_p2p_copy"))
