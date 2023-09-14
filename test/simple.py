import torch
from single_controller import DTensor, active_sharding, Manager, LocalWorker, to_local, _debug_wait_pending_callbacks, Sharding, WorkerMesh
import unittest
from torch.testing import assert_close
import subprocess
manager = Manager()
local_workers, workers = None, None

class TestLocal(unittest.TestCase):
    def setUp(self):
        global local_workers
        if local_workers is None:
            local_workers = [manager.create_worker(local=True) for _ in range(4)]
        self.w = local_workers[0]
        self.workers = local_workers

    def test_operations(self):
        x = DTensor.to_remote(torch.ones(2, 2), sharding=self.w)
        xx = x.add(x)
        xx = xx.add(x).max(dim=0)
        f = xx[0].to_local()

        assert_close(torch.ones(2)*3, f.wait())

        with active_sharding(sharding=self.w):
            o = torch.ones(2) + x

        lo = o.to_local()
        self.assertEqual(f.wait().dim(), 1)

        lo.then(lambda f: print("CB", f))
        assert_close(lo.wait(), torch.ones(2, 2) * 2)

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

    def test_two_workers(self):
        mesh = WorkerMesh(self.workers[0:2])
        sharding = Sharding(mesh, [0])
        r = DTensor.to_remote(torch.arange(12).reshape(4, 3), sharding)
        r = (r + r).sum(dim=1)
        assert_close(r.to_local().wait(), (torch.arange(12).reshape(4, 3)*2).sum(dim=1))

        sharding = Sharding(mesh, [1])

        r = DTensor.to_remote(torch.arange(12).reshape(3, 4), sharding)
        r = (r + r).sum(dim=0, keepdim=True)
        self.assertTrue(torch.allclose(r.to_local().wait(), (torch.arange(12).reshape(3, 4)*2).sum(dim=0, keepdim=True)))

    def test_four_workers(self):
        mesh = WorkerMesh(self.workers[0:4]).reshape(2, 2)
        sharding = Sharding(mesh, [2, 0])
        inp = torch.arange(4*5*4).reshape(4, 5, 4)
        r = DTensor.to_remote(inp, sharding)
        out = r.to_local().wait()
        self.assertTrue(torch.allclose(inp, out))
        self.assertTrue(torch.allclose(inp + inp, (r + r).to_local().wait()))

    def test_replicated(self):
        mesh = WorkerMesh(self.workers[0:2])
        rep = Sharding(mesh, ['r'])
        split = Sharding(mesh, [0])
        W = torch.rand(3, 4)
        I = torch.rand(6, 3)
        Wr = DTensor.to_remote(W, rep)
        Ir = DTensor.to_remote(I, split)
        O = I @ W
        Or = Ir @ Wr
        assert_close(Or.to_local().wait(), O)
        assert_close(Wr.to_local().wait(), W)

    def test_shardreplicate(self):
        mesh = WorkerMesh(self.workers[0:4]).reshape(2, 2)
        rep = Sharding(mesh, ['r', 'r'])
        split = Sharding(mesh, [0, 'r'])
        W = torch.rand(3, 4)
        I = torch.rand(6, 3)
        Wr = DTensor.to_remote(W, rep)
        Ir = DTensor.to_remote(I, split)
        O = I @ W
        Or = Ir @ Wr
        assert_close(Or.to_local().wait(), O)
        assert_close(Wr.to_local().wait(), W)

    def test_partial(self):
        mesh = WorkerMesh(self.workers[0:2])
        for d in range(2):
            sharded = Sharding(mesh, [d])
            tl = torch.arange(4*6).reshape(4, 6)
            t = sharded.from_local(tl).cuda()
            t = t.sum(dim=d)
            ts = t.to_sharding_('r').cpu().to_local().wait()
            assert_close(tl.sum(dim=d), ts)

class TestRemote(TestLocal):
    def setUp(self):
        global workers
        if workers is None:
            workers = [manager.create_worker(devices=i % 2, local=False) for i in range(4)]
        self.w = workers[0]
        self.workers = workers

    def test_devices(self):
        mesh = WorkerMesh([manager.create_worker(0), manager.create_worker(1)])
        sharding = Sharding(mesh, [0])
        r = DTensor.to_remote(torch.rand(4, 2), sharding)
        r = r.cuda()
        rl = r.cpu().to_local().wait()

if __name__ == '__main__':
    unittest.main()
