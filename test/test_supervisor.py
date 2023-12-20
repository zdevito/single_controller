from supervisor import Context, as_completed, Future, get_message_queue
import unittest
from unittest.mock import patch, Mock
from contextlib import contextmanager
import supervisor
import time
import subprocess
import zmq
import os
import pickle
import threading
import supervisor.host
import logging
from socket import gethostname
import signal
import tempfile
from pathlib import Path
import sys
import socket

logging.basicConfig(
    format="%(asctime)s %(levelname)s:%(name)s:%(message)s", level=logging.INFO
)
@contextmanager
def context(*args, **kwargs):
    try:
        ctx = Context(*args,  **kwargs)
        yield ctx
    finally:
        ctx.shutdown()

@contextmanager
def mock_process_handling():
    class Interface:
        pass
    lock = threading.Lock()
    all_processes = []
    def killpg(pid, return_code):
        with lock:
            p = all_processes[pid]
            if p.immortal and return_code != signal.SIGKILL:
                return
            if not hasattr(p, 'returncode'):
                p.returncode = return_code
                with os.fdopen(p._done_w, "w") as f:
                    f.write('done')

    class MockPopen:
        def __init__(self, *args, fail=False, immortal=False, **kwargs):
            if fail:
                raise RuntimeError("process fail")
            with lock:
                self.args = args
                self.kwargs = kwargs
                self.immortal = immortal
                self.pid = len(all_processes)
                all_processes.append(self)
                self.signals_sent = []
                self._done_r, self._done_w = os.pipe()

        def send_signal(self, sig):
            killpg(self.pid, sig)

        def wait(self):
            return self.returncode

    def mock_pidfdopen(pid):
        with lock:
            return all_processes[pid]._done_r

    with patch.object(subprocess, 'Popen', MockPopen), patch.object(supervisor.host, 'pidfd_open', mock_pidfdopen), patch.object(os, 'killpg', killpg):
        yield killpg

@contextmanager
def connected_host():
    context: zmq.Context = zmq.Context(1)
    backend = context.socket(zmq.ROUTER)
    backend.setsockopt(zmq.IPV6, True)
    backend.bind("tcp://*:55555")
    exited = None
    host = supervisor.host.Host('tcp://127.0.0.1:55555')
    def run_host():
        nonlocal exited
        try:
            host.run_event_loop_forever()
        except ConnectionAbortedError as e:
            exited = e
        except SystemExit:
            exited = True

    thread = threading.Thread(target=run_host, daemon=True)
    thread.start()
    try:
        yield backend, host
    finally:
        backend.close()
        context.term()
        thread.join(timeout=1)
        if thread.is_alive():
            raise RuntimeError("thread did not terminate")
    host.context.destroy(linger=500)
    if exited != True:
        raise exited

@contextmanager
def host_sockets(N: int):
    context: zmq.Context = zmq.Context(1)
    def create_socket():
        backend = context.socket(zmq.DEALER)
        backend.setsockopt(zmq.IPV6, True)
        backend.connect("tcp://localhost:55555")
        return backend
    sockets = [create_socket() for i in range(N)]
    try:
        yield sockets
    finally:
        context.destroy(linger=500)

def emulate_mast_launch(args, N: int = 4):
    # for testing the mast launcher entrypoint
    def create_host(i):
        env = {**os.environ}
        env['TW_TASK_ID'] = str(i)
        env['MAST_HPC_TASK_GROUP_HOSTNAMES'] = socket.gethostname()
        env['TORCH_ELASTIC_SUPERVISOR'] = str(i == 0)
        env['MAST_HPC_TASK_GROUP_SIZE'] = str(N)
        return subprocess.Popen(args, env=env)
    hosts = [create_host(i) for i in range(N)]
    expiry = time.time() + 10
    try:
        r = [h.wait(timeout=max(0, expiry - time.time())) for h in hosts]
        return r
    finally:
        for h in hosts:
            if h.poll() is None:
                h.kill()

class TestSupervisor(unittest.TestCase):
    def test_future(self):
        with context() as ctx:
            def future(hostname=None):
                return Future(ctx, "test", hostname)

            f = future()
            self.assertFalse(f.done())
            fstr = str(f)
            self.assertIn("incomplete", fstr)
            ctx._schedule(lambda: f.set_result("finished"))
            self.assertEqual(f.result(timeout=1), "finished")
            fstr = str(f)
            self.assertIn("[complete", fstr)
            f2 = future(hostname=f)
            fstr = str(f2)
            self.assertIn("finished", fstr)
            with self.assertRaises(TimeoutError):
                f2.result(timeout=.001)
            with self.assertRaises(TimeoutError):
                f2.exception(timeout=.001)
            ctx._schedule(lambda: f2.set_exception(ValueError('foo')))
            self.assertIsInstance(f2.exception(timeout=1), ValueError)
            with self.assertRaises(ValueError):
                f2.result()

            m = Mock()
            with patch.object(supervisor.logger, 'exception', m):
                f3 = future()
                l = []
                f3.add_done_callback(lambda f: l.append(True))
                def err(v):
                    raise ValueError(v)
                f3.add_done_callback(lambda f: err("done0"))
                self.assertFalse(l)
                ctx._schedule(lambda: f3.set_result("finished"))
                f3.result(timeout=1)
                self.assertEqual(l[0], True)
                f3.add_done_callback(lambda f: l.append(4))
                f3.add_done_callback(lambda f: err("done1"))
                self.assertEqual(l[1], 4)
            self.assertEqual(len(m.call_args_list), 2)
            f4 = future(hostname=future())
            self.assertIn("unconnected", str(f4))
            ctx._schedule(lambda: f4.set_result("finished"))
            time.sleep(0.1)
            self.assertEqual(f4.result(timeout=0), "finished")
            f5 = future()
            ctx._schedule(lambda: f5.set_result("finished"))
            self.assertEqual(f5.result(), "finished")


    def test_as_completed(self):
        with context() as ctx:
            futures = [Future(ctx, f"future_{i}", None) for i in range(10)]
            ctx._schedule(lambda: futures[1].set_result(1))
            for f in as_completed(futures, timeout=1):
                self.assertTrue(f is futures[1] and f.result() == 1)
                break
            ctx._schedule(lambda: futures[2].set_result(2))
            a = as_completed(futures, timeout=1)
            for f in a:
                if f is futures[1]:
                    pass
                elif f is futures[2]:
                    self.assertTrue(f.result() == 2)
                    nf = Future(ctx, "new_future", None)
                    a.add(nf)
                    ctx._schedule(lambda: nf.set_result(3))
                else:
                    self.assertIs(f, nf)
                    break
            with self.assertRaises(TimeoutError):
                for x in as_completed(futures[3:], timeout=.001):
                    self.fail("should have timed out")
            m = Mock()
            with patch.object(Context._process_futures, '__defaults__', (0,)), patch.object(supervisor.logger, 'info', m):
                with self.assertRaises(TimeoutError):
                    for x in as_completed(futures[3:], timeout=.001):
                        pass
            self.assertIn("Waiting for {Future[", m.call_args[0][0])

            for _ in as_completed([]):
                pass

            seen = False
            for x in as_completed([futures[1]]):
                seen = True
            self.assertTrue(seen)

            x = supervisor.wait(futures, return_when=supervisor.FIRST_COMPLETED)
            self.assertTrue(len(x.done))
            self.assertTrue(len(x.not_done))

            x = supervisor.wait(futures[1:3], supervisor.ALL_COMPLETED)
            self.assertEqual(len(x.done), 2)
            self.assertEqual(len(x.not_done), 0)
            ctx._schedule(lambda: futures[4].set_exception(ValueError()))
            x = supervisor.wait(futures[4:6], return_when=supervisor.FIRST_EXCEPTION)
            self.assertTrue(futures[4] in x.done)

    @patch('subprocess.Popen', 'subprocess.')
    def test_host_manager(self):
        with mock_process_handling() as kill, connected_host() as (socket, host), patch.object(supervisor.host, 'ABORT_INTERVAL', 0.01):
            f, msg = socket.recv_multipart()
            _hostname, _, hostname = pickle.loads(msg)
            def launch(proc_id, rank=0, processes_per_rank=1, world_size=1, popen={"env": None}, name='fake', simulate=False, log_file=None):
                msg = ('launch', proc_id, rank, processes_per_rank, world_size, popen, name, simulate, log_file)
                socket.send_multipart([f, pickle.dumps(msg)])
            def send(msg):
                socket.send_multipart([f, pickle.dumps(msg)])
            def recv():
                return pickle.loads(socket.recv_multipart()[1])
            self.assertEqual(_hostname, "_hostname")
            self.assertEqual(hostname, gethostname())

            launch(1)
            self.assertEqual(recv(), ('_started', 1, 0))
            kill(0, 4)
            self.assertEqual(recv(), ('_exited', 1, 4))

            launch(2)
            self.assertEqual(recv(), ('_started', 2, 1))
            send(('send', 2, "a message"))
            msg_queue = get_message_queue(2, host.proc_addr)
            self.assertEqual(pickle.loads(msg_queue.recv()), "a message")
            send(('send', 2, "another message"))
            self.assertEqual(pickle.loads(msg_queue.recv()), "another message")
            msg_queue.send(b'a reply')
            msg_queue.close()
            msg_queue.context.term()
            self.assertEqual(recv(), ('_response', 2, b'a reply'))
            send(('signal', 2, 8, True))
            self.assertEqual(recv(), ('_exited', 2, 8))
            launch(3, popen = {'env': {'foo': '3'}})
            self.assertEqual(recv(), ('_started', 3, 2))
            send(('signal', 3, 9, False))
            self.assertEqual(recv(), ('_exited', 3, 9))
            launch(4, popen={'fail': True, 'env': None})
            _started, _, msg = recv()
            self.assertEqual(_started, '_started')
            self.assertIn('process fail', msg)
            launch(5, simulate=True)
            self.assertEqual(recv(), ('_started', 5, 2))
            self.assertEqual(recv(), ('_exited', 5, 0))
            launch(6) # leave something open
            launch(7, popen={'immortal': True, 'env': None})
            send(('abort', None))
        # test double shutodwn
        host.shutdown()

        with self.assertRaises(ConnectionAbortedError):
            with connected_host() as (socket, _):
                f, msg = socket.recv_multipart()
                socket.send_multipart([f, pickle.dumps(('abort', 'An error'))])

    def test_host_timeout_and_heartbeat(self):
        with self.assertRaises(ConnectionAbortedError):
            with patch.object(supervisor.host, "HEARTBEAT_INTERVAL", .01), connected_host() as (socket, host):
                f, msg = socket.recv_multipart()
                socket.send_multipart([f, b""])
                time.sleep(0.1)
                f, msg = socket.recv_multipart()
                self.assertEqual(msg, b"")


    def test_supervisor_api(self):
        with context() as ctx:
            h0, h1 = ctx.request_hosts(2)
            proc = ctx.create_process_group([h0], args = ['test'])[0]
            proc.send('hello')
            pm = proc.recv(filter=lambda x: x == 'world')
            with host_sockets(1) as (socket,):
                socket.send_pyobj(('_hostname', None, 'host0'))
                self.assertEqual(h0.hostname().result(timeout=1), 'host0')
                self.assertIn('host0', str(h0))
                self.assertEqual(socket.recv(), b'')
                expected = ('launch', 0, 0, 1, 1, {'args': ['test'], 'env': None, 'cwd': None}, 'pg0_rank0', False, None)
                self.assertEqual(socket.recv_pyobj(), expected)
                self.assertEqual(socket.recv_pyobj(), ('send', 0, 'hello'))
                self.assertFalse(pm.done())
                socket.send_pyobj(('_started', 0, 7))
                socket.send_pyobj(('_response', 0, pickle.dumps('nope')))
                socket.send_pyobj(('_response', 0, pickle.dumps('world')))
                self.assertEqual(proc.pid().result(timeout=1), 7)
                self.assertEqual(pm.result(timeout=1), 'world')
                self.assertEqual(proc.recv().result(timeout=1), 'nope')
                pm1 = proc.recv()
                ctx.return_hosts([h0])
                self.assertTrue(proc.returncode().exception(timeout=1) is not None)
                self.assertTrue(proc.recv().exception(timeout=1) is not None)
                proc = None
                self.assertTrue(h0.connection_lost())
                self.assertTrue(pm1.exception(timeout=1) is not None)
                p, = ctx.create_process_group([h0], args = ['test3'])
                self.assertTrue(p.pid().exception(timeout=1) is not None)

            with host_sockets(1) as (socket,):
                socket.send_pyobj(('_hostname', None, 'host1'))
                self.assertEqual(socket.recv(), b'')
                self.assertEqual(h1.hostname().result(timeout=1), 'host1')
                proc = ctx.create_process_group([h1], args = ['test2'])[0]
                self.assertIn('rank=0', str(proc))
                expected = ('launch', 2, 0, 1, 1, {'args': ['test2'], 'env': None, 'cwd': None}, 'pg2_rank0', False, None)
                self.assertEqual(socket.recv_pyobj(), expected)
                proc = None
                # test sending a message after a proc timeout
                socket.send_pyobj(('_response', 2, pickle.dumps('old response')))
                socket.send(b'')
                self.assertEqual(socket.recv(), b'')




        # now try with host connecting before the host object exists
        with host_sockets(1) as (socket,):
            socket.send_pyobj(('_hostname', None, 'host0'))
            with context() as ctx:
                self.assertEqual(socket.recv(), b'')
                h, = ctx.request_hosts(1)
                self.assertEqual(h.hostname().result(timeout=1), 'host0')

    def test_bad_host_managers(self):
        with context() as ctx:
            with host_sockets(5) as (socket0, socket1, socket2, socket3, socket4):
                socket0.send(b'somegarbage')
                self.assertEqual(socket0.recv_pyobj(), ('abort', 'Connection did not start with a hostname'))
                socket1.send_pyobj(('_hostname', None, 7))
                self.assertEqual(socket1.recv_pyobj(), ('abort', 'Connection did not start with a hostname'))
                socket2.send_pyobj(('_hostname', None, 'host0'))
                self.assertEqual(socket2.recv(), b'')
                socket2.send_pyobj(('_started', 0, 7))
                self.assertEqual(socket2.recv_pyobj(), ('abort', 'Host manager sent messages before attached.'))
                h, = ctx.request_hosts(1)
                socket3.send_pyobj(('_hostname', None, 'host3'))
                self.assertEqual(h.hostname().result(timeout=1), 'host3')
                socket4.send(b'')
                self.assertEqual(socket4.recv_pyobj(), ('abort', 'Connection did not start with a hostname'))

    def test_host_manager_no_heartbeat(self):
        with patch.object(supervisor, 'HEARTBEAT_INTERVAL', .01), context() as ctx, host_sockets(1) as (socket,):
            socket.send_pyobj(('_hostname', None, 'host0'))
            self.assertEqual(socket.recv(), b'')
            socket.send(b'')
            self.assertEqual(socket.recv(), b'')
            h, = ctx.request_hosts(1)
            self.assertEqual(socket.recv_pyobj(), ('abort', 'Host did not heartbeat'))
            socket.send(b'')
            self.assertEqual(socket.recv_pyobj(), ('abort', 'Supervisor thought host timed out'))
            h.time_connection_lost().result(timeout=1)

    def test_proc_creation(self):
        with context() as ctx, host_sockets(2) as (socket0, socket1):
            h0, h1 = ctx.request_hosts(2)
            socket0.send_pyobj(('_hostname', None, 'host0'))
            self.assertEqual(socket0.recv(), b'')
            socket1.send_pyobj(('_hostname', None, 'host1'))
            self.assertEqual(socket1.recv(), b'')
            pg = ctx.create_process_group([h0, h1], args = ['test'], processes_per_host=3)
            self.assertEqual(len(pg), 6)
            pg[0].signal(signal.SIGTERM)
            for i in range(3):
                socket0.recv_pyobj() # launches
            self.assertEqual(socket0.recv_pyobj(), ('signal', 0, signal.SIGTERM, True))
            socket0.send_pyobj(('_response', 0, pickle.dumps('hello')))
            socket0.send_pyobj(('_response', 0, pickle.dumps('world')))
            self.assertEqual('world', pg[0].recv(lambda x: x == 'world').result(timeout=1))
            self.assertEqual('hello', pg[0].recv().result(timeout=1))
            socket0.send_pyobj(('_started', 1, 8))
            socket0.send_pyobj(('_exited', 1, 7))
            self.assertEqual(7, pg[1].returncode().result(timeout=1))
            socket0.send_pyobj(('_started', 2, "Failed"))
            self.assertTrue(pg[2].pid().exception(timeout=1) is not None)

    def test_log_redirect(self):
        m = Mock()
        with tempfile.NamedTemporaryFile(delete=True) as f, patch.object(os, 'dup2', m), context(log_format=f.name):
            pass
        m.assert_called()

    def test_host_lost_first(self):
        with context() as ctx, host_sockets(1) as (socket0,):
            h0, = ctx.request_hosts(1)
            h1, = ctx.replace_hosts([h0])
            self.assertTrue(h0.hostname().exception(timeout=1) is not None)
            socket0.send_pyobj(('_hostname', None, 'host0'))
            self.assertEqual(socket0.recv(), b'')
            self.assertEqual('host0', h1.hostname().result(timeout=1))

    def test_integration(self):
        test_name = Path(__file__).parent / 'supervisor_integration.py'
        def launch(health, train, expect, N=4, run_fraction=1, rank_fraction=1, connections=4, ):
            config = {
                'N': N,
                'health': health,
                'train': train,
                'run_fraction': run_fraction,
                'rank_fraction': rank_fraction
            }
            result = emulate_mast_launch([sys.executable, test_name, '--supervise', repr(config)], connections)
            for e, r in zip(expect, result):
                if e == '.':
                    self.assertEqual(r, 0)
                else:
                    self.assertNotEqual(r, 0)
            return result
        launch(health=[[4, 3, 2, 1]],train = ["........"], expect = '....')
        launch(health=[[4, 3, 2, 1], [4, 3, 2, 1]],train = ["....F...", "........"], expect = '....')
        launch(health=[[4, 3, "hang", 1], [4, 3, 2, 1]], train = ["......"], rank_fraction=.75, run_fraction=.75, expect = '....')
        launch(health=[[4, 3, 2, 1], [4, 3, 2, 1], [4, 3, 2, 1]], train = ["...E..", "....F.", "......"], rank_fraction=.75, run_fraction=.75, expect = '....')
        launch(health=[[4, 3, 2, 1], [4, 3, 2, 1], [4, 3, 2, 1]], train = ["...E..", ".E....", "......"], rank_fraction=.75, run_fraction=.75, connections=5, expect = '....')

if __name__ == '__main__':
    unittest.main()
