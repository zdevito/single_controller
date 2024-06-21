import traceback
import sys
from supervisor.logging import fix_exception_lines
def custom_excepthook(exc_type, exc_value, exc_traceback):
    tb_lines = fix_exception_lines(traceback.format_exception(exc_type, exc_value, exc_traceback))
    print('\n'.join(tb_lines), file=sys.stderr)

sys.excepthook = custom_excepthook

from collections import defaultdict
from supervisor import Context, HostConnected
from supervisor.host import Host as HostManager
from controller import world_mesh, active_mesh, _Controller, DeviceMesh, remote_function, Future, RemoteException, fetch_shard, Stream, active_stream
from controller.ndslice import NDSlice
from contextlib import contextmanager, ExitStack
from threading import Thread
from functools import cache, partial
from unittest import main, TestCase
from time import sleep
import torch
import signal
import os
from unittest.mock import patch
from weakref import WeakKeyDictionary
from controller._testing import LocalContext
import random
import math

@remote_function('controller.worker.log')
def log(*args):
    pass

@remote_function('builtins.list')
def rlist(elem):
    return elem

@remote_function('controller._testing.do_bogus_tensor_work')
def do_bogus_tensor_work(x, y, fail_rank=None):
    return x + y  # real function actually does x @ y


local = LocalContext()

class TestController(TestCase):

    @classmethod
    def tearDownClass(cls):
        local.close()

    @classmethod
    def local_device_mesh(cls, N, gpu_per_host, activate=True):
        return local.local_device_mesh(N, gpu_per_host, activate)

    def test_hello(self):
        with self.local_device_mesh(2, 2) as device_mesh:
            log(device_mesh)

    def test_simple_tensors(self):
        with self.local_device_mesh(2, 2):
            x = torch.rand(3, 4)
            y = x + x
            log("%s %s", x, y)
            z = torch.std_mean(x)
            log("%s", z)

    def test_errors(self):
        t = torch.rand(3, 4)
        with self.local_device_mesh(2, 2) as device_mesh:
            y = torch.rand(3, 4)
            with self.assertRaisesRegex(TypeError, 'LOCAL_TENSOR'):
                t.add(y)
            with self.assertRaisesRegex(TypeError, 'WRONG_MESH'):
                sm = device_mesh(host=0)
                with active_mesh(sm):
                    x = torch.rand(3, 4)
                    x.add(y)

    def test_mesh_index(self):
        fake_processes = NDSlice(0, [2, 3, 4], [12, 4, 1])
        dm = DeviceMesh(None, fake_processes, ('a', 'b', 'c'))
        self.assertEqual(0, dm(a=0, b=0, c=0).processes[0])
        x = dm(a=0, b=0)
        print(x.processes, x.names)
        self.assertEqual(x.processes[:], fake_processes[0:4])
        self.assertEqual(x.names, ('c',))
        self.assertEqual(x.processes.sizes[0], 4)
        x = dm(c=slice(None, None, 2))
        self.assertEqual(x.processes[:], fake_processes[::2])
        x = dm(b=2, c=3)
        self.assertEqual(x.processes[:], (11, 23))

    def test_sub_mesh(self):
        with self.local_device_mesh(2, 2) as device_mesh:
            h0 = device_mesh(host=0)
            h1 = device_mesh(host=1)
            with active_mesh(h0):
                x = torch.rand(3, 4)
                log('x: %s', x)
            with active_mesh(h1):
                y = torch.rand(3, 4)
                log('y: %s', y)
                with self.assertRaisesRegex(TypeError, 'WRONG_MESH'):
                    log("x: %s", x)


    def test_user_call(self):
        with self.local_device_mesh(2, 2) as device_mesh:
            x = torch.rand(3, 4)
            y = rlist((x + 1, x))
            log("%s", y)

            # resume monday:
            # 1. tensor ctor resource guard (done)
            # 2. __torch_dispatch__ forward of normal ops (done)
            # 3. collectives created for device mesh
            # 4. implement comms APIs
            # 5. transfer tensor back, and simple future to wait for result.

    @patch('torch.distributed.new_group', new=lambda ranks, use_local_synchronization: ranks)
    def test_worker_mesh_init(self):
        from controller.worker import DeviceMesh as WorkerDeviceMesh
        processes = NDSlice(0, [3, 4], [4, 1])
        wdm = WorkerDeviceMesh(('a', 'b'), processes, rank=1)
        a, b = wdm.dims['a'], wdm.dims['b']
        self.assertEqual(b.process_group, [0, 1, 2, 3])
        self.assertEqual(b.rank, 1)

        self.assertEqual(a.process_group, [1, 5, 9])
        self.assertEqual(a.rank, 0)

        wdm = WorkerDeviceMesh(('a', 'b'), processes, rank=6)
        a, b = wdm.dims['a'], wdm.dims['b']
        self.assertEqual(b.process_group, [4, 5, 6, 7])
        self.assertEqual(b.rank, 2)
        self.assertEqual(a.process_group, [2, 6, 10])
        self.assertEqual(a.rank, 1)

        processes = NDSlice(0, [3, 4, 2],  [8, 2, 1])
        wdm = WorkerDeviceMesh(('a', 'b', 'c'), processes, rank=10)
        print(wdm.dims)

    def test_reduce(self):

        with self.local_device_mesh(2, 2) as device_mesh:
            x = 12*2*device_mesh.rank('host') + 12*device_mesh.rank('gpu') + torch.arange(12).reshape(3, 4)
            x = x.cuda()
            y = x.reduce('gpu', 'sum')
            g = x.reduce('gpu', 'stack')
            with self.assertRaisesRegex(TypeError, 'When scattering'):
                x = x.reduce('gpu', 'sum', scatter=True)
            x = x.reshape(2, 6)
            atoa = x.reduce('gpu', 'stack', scatter=True)
            rs = x.reduce('gpu', 'sum', scatter=True)
            log("x: %s\ny:%s\ng:%s\natoa:%s\nrs:%s\n", x, y, g, atoa, rs)

    def test_fetch(self):
        with self.local_device_mesh(2, 2) as device_mesh:
            h = device_mesh.rank('host')
            g = device_mesh.rank('gpu')
            for hi in range(2):
                for gi in range(2):
                    x, y = fetch_shard((h, g), dict(host=hi, gpu=gi)).result()
                    with active_mesh(None):
                        self.assertTrue((hi, gi) == (x.item(), y.item()))

    def test_remote_exception(self):
        with self.local_device_mesh(2, 2) as device_mesh:
            x = torch.rand(3, 4)
            y = torch.rand(3, 4)
            z = do_bogus_tensor_work(x, y)
            a = z + x
            b = x + y            
            with self.assertRaisesRegex(RemoteException, 'do_bogus_tensor_work'):
                r = fetch_shard(a).result(timeout=10)
            # but values not dependent on z are fine
            fetch_shard(b).result(timeout=10)

    def test_distributed_error(self):
        with self.local_device_mesh(2, 2) as device_mesh:
            x = torch.rand(3, 4).cuda()
            y = torch.rand(3, 4).cuda()
            # z is broken on rank 1 but not others
            z = do_bogus_tensor_work(x, y, fail_rank=1)
            # test that rank 1 is still doing work despite z failing
            a = (x + y).reduce('gpu')
            fetch_shard(a).result()
            # but z itself should fail, even if we do not fetch it from rank 1
            # (since fetch shard says we first want to assert the whole tensor is correct)
            with self.assertRaisesRegex(RemoteException, 'do_bogus_tensor_work'):
                fetch_shard(z).result()
            # try to reduce z, which should fail, but ranks that are not 1 do not 
            # know about the failure. Rank 1 should still participate in the reduce
            # to unblock work.
            rz = z.reduce('gpu')
            # but we should see the error message still retrieving it because it is
            # dependent on an error.
            with self.assertRaisesRegex(RemoteException, 'do_bogus_tensor_work'):
                fetch_shard(rz).result()
            # however, we should still be able to compute and get a result back
            # from host 1, signaling that the reduction didn't get cuda compute stuck.
            fetch_shard(2*x, {'gpu': 1, 'host': 0}).result()

    def test_future(self):
        the_time = 0
        the_messages = []
        class MockController:
            def _read_messages(self, timeout):
                nonlocal the_time
                if not the_messages:
                    return
                time, action = the_messages[0]
                if timeout is None or time <= the_time + timeout:
                    the_time = time
                    action()
                    the_messages.pop(0)
                else:
                    the_time += timeout
            def _request_status(self):
                pass

        ctrl = MockController()

        def time():
            return the_time

        with patch('time.time', time):
            f = Future(ctrl)
            the_messages = [(1, lambda: f._set_result(4))]
            self.assertTrue(not f.done())
            with self.assertRaises(TimeoutError):
                f.result(timeout=.5)
            self.assertEqual(4, f.result(timeout=1))
            self.assertIsNone(f.exception())
            self.assertTrue(f.done())
            f = Future(ctrl)
            the_messages = [(1, lambda: None), (2, lambda: f._set_result(3))]
            the_time = 0
            self.assertEqual(3, f.result())
            f = Future(ctrl)
            re = RemoteException(0, Exception(), [], [])

            the_messages = [(1, lambda: None), (2, lambda: f._set_result(re))]
            the_time = 0
            self.assertIsNotNone(f.exception())

            f = Future(ctrl)
            the_messages = [(0, lambda: None), (.2, lambda: f._set_result(7))]
            the_time = 0
            self.assertEqual(7, f.result(timeout=.3))

    def test_mutate(self):

        with self.local_device_mesh(2, 2) as device_mesh:
            x = torch.rand(3, 4).cuda()
            x.abs_()
            s = Stream('other')
            b = s.borrow(x)
            with self.assertRaisesRegex(ValueError, "would be mutated"):
                x.abs_()
            with active_stream(s):
                c = b.add(b)
            b.drop()
            x.abs_()
            b = s.borrow(x, mutable=True)
            with active_stream(s):
                b.abs_()
            b.drop()
            # del b
            x.abs_()

    def test_movement(self):
        with self.local_device_mesh(2, 2) as device_mesh:
            
            sm0 = device_mesh(host=0)
            sm1 = device_mesh(host=1)

            with active_mesh(sm0):
                x = torch.rand(3, 4, device='cuda')
                log("x: %s", x)
                y = x.to_mesh(sm1)

            a = torch.rand(3, 4, device='cuda')

            b = a.slice_mesh(host=0)
            d = b.to_mesh(sm0)
            c = b.to_mesh(sm1)
            with active_mesh(sm0):
                log("d: %s", d)
            with active_mesh(sm1):
                log("y: %s", y)
                log("c: %s", c)

    def test_ndslice(self):
        def check(s):
            elems = list(s)
            # print("checking", s, elems)
            all = set(elems)
            assert len(all) == len(elems), "wrong checks for valid strides"
            if all:
                small = min(all)
                large = max(all)

            for i, e in enumerate(s):
                assert s[i] == e, "index broken"
                # print(i, ":", e, s.index(e))
                assert s.index(e) == i, "inverse broken"

            N = math.prod(s.sizes)
            try:
                s[N]
                raise RuntimeError("index broken, too many elements")
            except IndexError:
                pass

            try:
                s[-1]
                raise RuntimeError("index broken, too many elements")
            except IndexError:
                pass

            if not all:
                return
            for i in range(small, large + 1):
                try:
                    if i not in s:
                        s.index(i)
                        raise RuntimeError("index broken, extra elements")
                except ValueError:
                    pass

        check(NDSlice(0, [3, 4], [4, 1]))
        check(NDSlice(0, [3, 4], [1, 4]))
        check(NDSlice(1, [3], [2]))
        check(NDSlice(1, [3, 3], [12, 2]))
        check(NDSlice(0, [4, 3, 8], [48, 16, 2]))
        check(NDSlice(24, [4, 3, 8], [48*2, 16, 2]))
        check(NDSlice(37, [], []))

        def fuzz(gen):
            ndim = gen.choices([1, 2, 3, 4], [1, 2, 3, 4])[0]
            sizes = [gen.randrange(1, 10) for _ in range(ndim)]
            strides = []
            run = 1
            for j in range(ndim):
                run *= gen.randrange(1 if j == 0 else 2, 4)
                strides.append(run)

            order = list(range(ndim))
            gen.shuffle(order)
            try:
                return NDSlice(gen.randrange(10000), [sizes[o] for o in order], [strides[o] for o in order])
            except ValueError:
                # rejection sample, because some things are not in bound
                return fuzz(gen)

        gen = random.Random(0)
        for _ in range(1000):
            s = fuzz(gen)
            check(s)


    def test_autograd(self):
        from controller.gradient_generator import grad_generator
        with self.local_device_mesh(2, 2) as device_mesh:
            x = torch.rand(3, 4, requires_grad=True)
            y = torch.rand(4, 3, requires_grad=True)
            z = torch.rand(3, requires_grad=True)

            l = (x @ y + z).sum()
            with active_mesh(None):
                # check backward restores forward mesh
                for x in grad_generator(l, [z, y, x]):
                    with active_mesh(device_mesh):
                        fetch_shard(x).result()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # this is normally done by the host process in a singal handler
        # but we are using threads to run the host managers in this
        # test setup, it is necessary to have this process issue the
        # kills of any running worker processes
        local.interrupt()
        raise
