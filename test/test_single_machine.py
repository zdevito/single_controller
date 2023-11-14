import torch
from single_machine import DTensor, active_sharding, Manager, to_local, Sharding, WorkerMesh, psum_, wait_pending_callbacks, stats
import unittest
from torch.testing import assert_close
import subprocess
manager = Manager()
local_workers, workers = None, None

class TestSingleMachine(unittest.TestCase):
    def setUp(self):
        global local_workers
        if local_workers is None:
            local_workers = manager.create_workers(4)
        self.w = local_workers[0]
        self.workers = local_workers

    def tearDown(self):
        wait_pending_callbacks()
        stats.report()

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
        wait_pending_callbacks()
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
        mesh = self.workers[0:2]
        sharding = mesh.Sharding(0)
        r = DTensor.to_remote(torch.arange(12).reshape(4, 3), sharding)
        r = (r + r).sum(dim=1)
        assert_close(r.to_local().wait(), (torch.arange(12).reshape(4, 3)*2).sum(dim=1))

        sharding = mesh.Sharding(1)

        r = DTensor.to_remote(torch.arange(12).reshape(3, 4), sharding)
        r = (r + r).sum(dim=0, keepdim=True)
        self.assertTrue(torch.allclose(r.to_local().wait(), (torch.arange(12).reshape(3, 4)*2).sum(dim=0, keepdim=True)))

    def test_four_workers(self):
        mesh = self.workers[0:4].reshape(2, 2)
        sharding = mesh.Sharding(2, 0)
        inp = torch.arange(4*5*4).reshape(4, 5, 4)
        r = DTensor.to_remote(inp, sharding)
        out = r.to_local().wait()
        self.assertTrue(torch.allclose(inp, out))
        self.assertTrue(torch.allclose(inp + inp, (r + r).to_local().wait()))

    def test_replicated(self):
        mesh = self.workers[0:2]
        rep = mesh.Sharding('r')
        split = mesh.Sharding(0)
        W = torch.rand(3, 4)
        I = torch.rand(6, 3)
        Wr = DTensor.to_remote(W, rep)
        Ir = DTensor.to_remote(I, split)
        O = I @ W
        Or = Ir @ Wr
        assert_close(Or.to_local().wait(), O)
        assert_close(Wr.to_local().wait(), W)

    def test_shardreplicate(self):
        mesh = self.workers[0:4].reshape(2, 2)
        rep = mesh.Sharding('r', 'r')
        split = mesh.Sharding(0, 'r')
        W = torch.rand(3, 4)
        I = torch.rand(6, 3)
        Wr = DTensor.to_remote(W, rep)
        Ir = DTensor.to_remote(I, split)
        O = I @ W
        Or = Ir @ Wr
        assert_close(Or.to_local().wait(), O)
        assert_close(Wr.to_local().wait(), W)

    def test_partial(self):
        mesh = self.workers[0:2]
        for d in range(2):
            sharded = mesh.Sharding(d)
            tl = torch.arange(4*6).reshape(4, 6)
            t = sharded.DTensor(tl).cuda()
            t = t.sum(dim=d)
            ts = t.cpu().to_local().wait()
            assert_close(tl.sum(dim=d), ts)

    def test_reduce(self):
        mesh = self.workers[0:2]
        sharded = mesh.Sharding(0)
        replicated = mesh.Sharding('r')

        tl = torch.arange(4*6).reshape(4, 6)
        t = sharded.DTensor(tl).cuda()
        t = t.sum(dim=0)
        t.to_sharding_(replicated)
        ts = t.cpu().to_local().wait()
        assert_close(tl.sum(dim=0), ts)

    def test_implicit_batch(self):
        mesh = self.workers[0:2]
        sharded = mesh.Sharding('b')
        replicated = mesh.Sharding('r')
        tl = torch.arange(4*6).reshape(4, 6)
        rl = torch.arange(2*6).reshape(2, 6)
        t = sharded.DTensor(tl)
        r = replicated.DTensor(rl)
        assert t.size(0) == 2
        assert_close((t + r).to_local().wait(), tl + torch.cat([rl, rl]))
        tc = t.cuda()
        psum_(tc)
        assert_close(tc.cpu().to_local().wait(), tl[0:2] + tl[2:4])

    def test_psum(self):
        mesh = self.workers[0:2]
        sharded = mesh.Sharding('b')
        x = torch.arange(4*6).reshape(4, 6)
        t = sharded.DTensor(x.clone())
        psum_(t)
        print(t.to_local().wait(), x[0:2] + x[2:4])
        assert_close(t.to_local().wait(), x[0:2] + x[2:4])

    def test_in_place(self):
        mesh = self.workers[0:2]
        sharded = mesh.Sharding('b')
        x = torch.arange(4*6).reshape(4, 6)
        t = sharded.DTensor(x.clone())
        with active_sharding(mesh.Sharding('r')):
            t.add_(torch.ones(2, 6, dtype=torch.long))

if __name__ == '__main__':
    unittest.main()
