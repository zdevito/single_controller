import torch
from single_controller import DTensor, active_sharding, Manager, LocalWorker, to_local, _debug_wait_pending_callbacks
import unittest

manager = Manager()
local_worker, worker = None, None

class TestLocal(unittest.TestCase):
    def setUp(self):
        global local_worker
        if local_worker is None:
            local_worker = manager.create_worker(local=True)
        self.w = local_worker

    def test_operations(self):
        x = DTensor.to_remote(torch.ones(2, 2), sharding=self.w)
        xx = x.add(x)
        xx = xx.add(x).max(dim=0)
        f = xx[0].to_local()

        self.assertTrue(torch.allclose(torch.ones(2)*3, f.wait()))

        with active_sharding(sharding=self.w):
            o = torch.ones(2) + x

        lo = o.to_local()
        self.assertEqual(f.wait().dim(), 1)

        lo.then(lambda f: print("CB", f))
        self.assertTrue(torch.allclose(lo.wait(), torch.ones(2, 2) * 2))

        saw = False
        def print_and_mark(x):
            print(x)
            nonlocal saw
            saw = True
        to_local({'x': x, 'o': o, 'y': x + x + x}).then(print_and_mark)
        _debug_wait_pending_callbacks()
        self.assertTrue(saw)
    def test_loop(self):
        x = DTensor.to_remote(torch.zeros(2, 2), sharding=self.w)
        for i in range(1000):
            if i % 10 == 0:
                x.to_local().then(print)
            x = x + 1
        self.assertTrue(x.to_local().wait()[0, 0].item() == 1000)
        print(x.to_local().wait())

    def test_dim(self):
        x = DTensor.to_remote(torch.ones(2, 2), sharding=self.w).to_local().wait()
        self.assertEqual(x.dim(), 2)


class TestRemote(TestLocal):
    def setUp(self):
        global worker
        if worker is None:
            worker = manager.create_worker(local=False)
        self.w = worker

if __name__ == '__main__':
    unittest.main()
