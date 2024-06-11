from concurrent.futures import ThreadPoolExecutor
from torch._subclasses.fake_tensor import FakeTensorMode

fake_mode = FakeTensorMode()
_fake_mode_worker = None


def fake_call(fn, args, kwargs):
    global _fake_mode_worker

    def work():
        with fake_mode:
            return fn(*args, **kwargs)
    if _fake_mode_worker is None:
        _fake_mode_worker = ThreadPoolExecutor(max_workers=1)
    return _fake_mode_worker.submit(work).result()
