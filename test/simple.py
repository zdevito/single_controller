import torch
from single_controller import DTensor, active_sharding, Manager, LocalWorker, to_local, _debug_wait_pending_callbacks, Sharding, compile, WorkerMesh, psum_
import unittest
from torch.testing import assert_close
import subprocess
manager = Manager()
local_workers, workers = None, None

class TestLocal(unittest.TestCase):
    def setUp(self):
        global local_workers
        if local_workers is None:
            local_workers = manager.create_workers(4, local=True)
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

    def test_compile(self):
        def foo(a, b):
            return a*a*a + b + torch.arange(4).reshape(2, 2)

        foo = compile(foo)
        x = DTensor.to_remote(torch.ones(2, 2) + 1, sharding=self.w)
        y = DTensor.to_remote(torch.ones(2, 2), sharding=self.w)
        with active_sharding(self.w):
            z = foo(x, y)
        assert_close(z.to_local().wait(), torch.ones(2, 2)*9 + torch.arange(4).reshape(2, 2))

    def test_compile_grad(self):

        def foo(a, b):
            return (a*a*a + b).sum(0).sum(0)

        x_l = torch.full((2, 2), 2.0, requires_grad=True)
        y_l = torch.ones(2, 2, requires_grad=True)
        z_l = foo(x_l, y_l)
        z_l.backward()

        foo = compile(foo)
        sharding = WorkerMesh([self.workers.flat_workers[0]]).Sharding('r')
        x = DTensor.to_remote(torch.full((2, 2), 2.0, requires_grad=True), sharding=sharding)
        y = DTensor.to_remote(torch.ones(2, 2, requires_grad=True), sharding=sharding)
        z = foo(x, y)
        z.backward()

        assert_close(x_l.grad, x.grad.to_local().wait())

    def test_compile_grad_sharded(self):

        def foo(a, b, t):
            return torch.nn.functional.cross_entropy(a @ b + a @ b, t, ignore_index=-1, reduction='sum')
        t_l = torch.zeros(4, dtype=torch.long)
        x_l = torch.full((4, 3), 1.0, requires_grad=True)
        y_l = torch.full((3, 2), 2.0, requires_grad=True)
        z_l = foo(x_l, y_l, t_l)

        z_l.backward()

        w = self.workers[0:2]
        sharding = w.Sharding(0)
        rep = w.Sharding('r')

        t = torch.zeros(4, dtype=torch.long)
        x = torch.full((4, 3), 1.0, requires_grad=True)
        y = torch.full((3, 2), 2.0, requires_grad=True)

        t = DTensor.to_remote(t, sharding=sharding)
        x = DTensor.to_remote(x, sharding=sharding)
        y = DTensor.to_remote(y, sharding=rep)
        foo = compile(foo)
        z = foo(x, y, t)
        z.backward()
        assert_close(x_l.grad, x.grad.to_local().wait())


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



class TestRemote(TestLocal):
    def setUp(self):
        global workers
        if workers is None:
            workers = manager.create_workers(4)
        self.w = workers[0]
        self.workers = workers

    def test_devices(self):
        mesh = manager.create_workers(2)
        sharding = mesh.Sharding(0)
        r = DTensor.to_remote(torch.rand(4, 2), sharding)
        r = r.cuda()
        rl = r.cpu().to_local().wait()

if __name__ == '__main__':
    unittest.main()
