from concurrent.futures import Future, as_completed, wait
from typing import List, Any
from functools import wraps
from collections import deque
from threading import Thread
from pprint import pprint
import zmq
import time
import signal
import pickle

HEARTBEAT_LIVENESS = 3     # 3..5 is reasonable
HEARTBEAT_INTERVAL = 1.0

def onsupervisor(orig):
    @wraps(orig)
    def wrapper(self, *args, **kwargs):
        fut = Future()
        self._context._schedule(lambda: orig(fut, *args, **kwargs))

    return wrapper

def debug_dict(o):
    if hasattr(o, '_debug_dict'):
        return {k: debug_dict(v) for k,v in o._debug_dict().items()}
    else:
        return repr(o)

class Host:
    def __init__(self, context):
        self._context = context
        self.expiry = None
        self._name = None
        self._connected = False
        self._deferred_sends = []
        self._proc_table = {}

    def _heartbeat(self):
        self.expiry = time.time() + HEARTBEAT_INTERVAL * HEARTBEAT_LIVENESS

    def _connect(self, name):
        self._name = name
        self._connected = True
        for msg in self._deferred_sends:
            self._context._backend.send_multipart([self._name, msg])
        self._deferred_sends.clear()

    def _disconnect(self):
        self._connected = False
        for p in self._proc_table.values():
            p._lost_host()

        # is there value in keeping this around for a reconnect?
        self._proc_table.clear()
        self._deferred_sends.clear()

    def _debug_dict(self):
        return self.__dict__

    def _send(self, msg):
        if not self._connected:
            self._deferred_sends.append(msg)
        else:
            self._context._backend.send_multipart([self._name, msg])

    def _launch(self, p):
        self._proc_table[id(p)] = p
        self._send(pickle.dumps(('launch', id(p), p.rank, p.world_size, p.args)))

class Process:
    def __init__(self, context, host, rank, world_size, args):
        self._context = context
        self.host = host
        self.rank = rank
        self.args = args
        self.world_size = world_size
        self._pid = Future()
        self._returncode = Future()
        self._recvs = []
        self._messages = []
        self._alive = True

    def returncode(self) -> 'Future[int]':
        return self._returncode

    def pid(self) -> 'Future[int]':
        return self._pid

    def _lost_host(self):
        self._alive = False
        e = ConnectionAbortedError("Lost connection to process host")
        for f in self._futures():
            if not f.done():
                f.set_exception(e)

    def _futures(self):
        yield self._pid
        yield self._returncode
        for _, f in self._recvs:
            yield f

    def send(self, msg: Any):
        self._context._schedule(lambda: self._send(msg))

    def _send(self, msg: Any):
        if self._alive:
            self.host._send(pickle.dumps(('send', id(self), msg)))

    def signal(self, signal=signal.SIGTERM):
        self._context._schedule(lambda: self._signal(signal))

    def _signal(self, signal):
        if self._alive:
            self.host._send(self.host._name, pickle.dumps(('signal', id(self), signal)))

    # return first response where filter(msg) is True
    def recv(self, filter: callable) -> 'Future[Any]':
        fut = Future()
        self._context._schedule(lambda: self._recv(fut, filter))
        return fut

    def _recv(self, fut: Future, filter: callable):
        for i, msg in enumerate(self._messages):
            if filter(msg):
                self._messages.pop(i)
                fut.set_result(msg)
                return
        if self._alive:
            self._recvs.append((filter, fut))
        else:
            fut.set_exception(ValueError("process no longer alive"))


    # TODO: annotation that registers this as a valid
    # message that can be sent

    def _response(self, msg):
        msg = pickle.loads(msg)
        for i, (filt, fut) in enumerate(self._recvs):
            if filt(msg):
                self._recvs.pop(i)
                fut.set_result(msg)
                return
        self._messages.append(msg)

    def _exited(self, returncode):
        self._returncode.set_result(returncode)

    def _started(self, pid):
        self._pid.set_result(pid)


class Context:
    def __init__(self):
        self._context = zmq.Context()

        # to talk to python clients in this process
        self._requests = deque()
        self._requests_ready = self._context.socket(zmq.PAIR)
        self._requests_ready.bind('inproc://doorbell')
        self._doorbell = self._context.socket(zmq.PAIR)
        self._doorbell.connect('inproc://doorbell')

        # to talk to other hosts

        self._backend = self._context.socket(zmq.ROUTER)
        self._backend.bind('tcp://*:55555')

        self._poller = zmq.Poller()
        self._poller.register(self._backend, zmq.POLLIN)
        self._poller.register(self._requests_ready, zmq.POLLIN)

        self._unassigned_hosts = deque()
        self._unassigned_connections = deque()
        self._name_to_host = {}
        self._last_heartbeat_check = time.time()

        self._thread = Thread(target=self._event_loop, daemon=True)
        self._thread.start()

    def _event_loop(self):
        while True:
            for sock, _ in self._poller.poll(timeout=HEARTBEAT_INTERVAL*1000*2):
                if sock is self._backend:
                    f, msg = self._backend.recv_multipart()
                    if f not in self._name_to_host:
                        # XXX: we should check that the thing connecting also
                        # thinks it is new, otherwise it might have stale state
                        if self._unassigned_hosts:
                            host = self._unassigned_hosts.popleft()
                        else:
                            host = Host(self)
                            self._unassigned_connections.append(host)
                        host._connect(f)
                        self._name_to_host[f] = host
                        print(f"New Client: {f}")
                    else:
                        host = self._name_to_host[f]
                    host._heartbeat()
                    if len(msg):
                        cmd, proc_id, *args = pickle.loads(msg)
                        # TODO: we shouldn't fail if we get proc_id's that do not exist
                        # they may have gotten delivered after we marked the processes as dead
                        # alternatively we tombstone the proc_ids instead of deleting them
                        proc = host._proc_table[proc_id]
                        getattr(proc, cmd)(*args)

                elif sock is self._requests_ready:
                    while self._requests:
                        self._requests_ready.recv()
                        fn = self._requests.popleft()
                        fn()

            t = time.time()
            if t - self._last_heartbeat_check > HEARTBEAT_INTERVAL*2:
                self._last_heartbeat_check = t
                # priority queue would be log(N)
                for key, host in self._name_to_host.items():
                    if host._connected and host.expiry < t:
                        host._disconnect()
                # pprint(debug_dict(self))

    def _debug_dict(self):
        return self.__dict__

    def _schedule(self, fn):
        self._requests.append(fn)
        self._doorbell.send(b'')

    def request_hosts(self, n: int) -> 'Future[List[Host]]':
        """
        Request from the scheduler n hosts to run processes on.
        The future is fulfilled when the reservation is made, but
        potenially before all the hosts check in with this API.

        Note: implementations that use existing slurm-like schedulers,
        will immediately full the future because the reservation was
        already made.
        """
        f = Future()
        hosts = tuple(Host(self) for i in range(n))
        f.set_result(hosts)
        self._schedule(lambda: self._request_hosts(hosts))
        return f


    def _next_connection(self):
        while self._unassigned_connections:
            host = self._unassigned_connections.popleft()
            # its possible this connection timed out in the
            # meantime
            if host.name is not None:
                return host
        return None

    def _request_hosts(self, hosts):
        for h in hosts:
            u = self._next_connection()
            if u is None:
                self._unassigned_hosts.append(h)
            else:
                h._connect(u.name)

    def return_hosts(self, hosts: List[Host]):
        """
        Processes on the returned hosts will be killed,
        and future processes launches with the host will fail.
        """
        self._schedule(lambda: self._return_hosts(hosts))

    def _return_hosts(self, hosts: List[Host]):
        for h in hosts:
            if h._connected:
                self._backend.send_multipart([h._name, pickle.dumps(('abort', False))])
                h._disconnect()

    def replace_hosts(self, hosts: List[Host]):
        """
        Request that these hosts be replaced with new hosts.
        Processes on the host will be killed, and future processes
        launches will be launched on the new hosts.
        """
        # if the host is disconnected, return it to the pool of unused hosts
        # and we hope that scheduler has replaced the job
        # if the host is still connected, then send the host a message
        # then cancel is processes and abort with an error to get the
        # the scheduler to reassign the host
        self._context._schedule(lambda: self._replace_hosts(hosts))

    def _replace_hosts(self, hosts):
        for h in hosts:
            if h._connected:
                self._backend.send_multipart([h._name, pickle.dumps(('abort', True))])
                h._disconnect()
            # detach this host object from current name
            self._name_to_host[h._name] = Host(self)
            h._name = None
            # let it get assigned to the next host to checkin
            self._unassigned_hosts.append(h)



    # TODO: other arguments like environment, etc.
    def create_process_group(self, hosts: List[Host], args, npp=1) -> List[Process]:
        world_size = npp*len(hosts)
        procs = tuple(Process(self, h, i*npp + j, world_size, args) for i, h in enumerate(hosts) for j in range(npp))
        self._schedule(lambda: self._launch_processes(procs))
        return procs

    def _launch_processes(self, procs):
        for p in procs:
            p.host._launch(p)
