from kvpick.link.link_emulator import _goodput_check, FakeClock, LinkEmulator

def test_goodput_fake_clock():
    for rate in (1, 5, 10, 25, 100):
        for rtt in (0.0005, 0.005):
            res = _goodput_check(rate, rtt)
            assert res["all_within_tol"] and res["pipelined_within_tol"], res

def test_real_copy_slower_than_link_dominates():
    fc = FakeClock(); link = LinkEmulator(100.0, 0.0, clock=fc.now, sleep=fc.sleep, spin_window_s=0.0)
    assert link.send(1 << 20, copy_fn=lambda: fc.tick(1.0)).wall >= 1.0
