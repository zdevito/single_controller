from typing import List, Any
from functools import wraps
from collections import deque
from threading import Thread
from pprint import pprint
import types
import zmq
import time
import signal
import pickle
import logging
import os
import weakref

logger = logging.getLogger("concurrent.futures")


HEARTBEAT_LIVENESS = 3     # 3..5 is reasonable
HEARTBEAT_INTERVAL = 1.0


class Future:
    """Represents the result of an asynchronous computation."""
    def __init__(self, context):
        self._context = context
        """Initializes the future. Should not be called by clients."""
        self._complete = False
        self._value = None
        self._was_exception = False
        self._done_callbacks = []

    def done(self):
        # wait 0 just polls to see if the done message
        # is already in the message queue
        return self._wait(0)

    def add_done_callback(self, fn):
        """Attaches a callable that will be called when the future finishes.

        Args:
            fn: A callable that will be called with this future as its only
                argument when the future completes or is cancelled. The callable
                will always be called by a thread in the same process in which
                it was added. If the future has already completed or been
                cancelled then the callable will be called immediately. These
                callables are called in the order that they were added.
        """
        if not self.done():
            self._done_callbacks.append(fn)
        else:
            try:
                fn(self)
            except Exception:
                logger.exception('exception calling callback for %r', self)

    def __get_result(self):
        if self._was_exception:
            try:
                raise self._value
            finally:
                # Break a reference cycle with the exception in self._exception
                self = None
        else:
            return self._value

    def _wait(self, timeout):
        if self._complete:
            return True
        for futs in self._context._generate_futures(timeout):
            for f, value, was_exception in futs:
                f._set_value(value, was_exception)
            if self._complete:
                return True
        return False

    def _invoke_callbacks(self):
        for callback in self._done_callbacks:
            try:
                callback(self)
            except Exception:
                logger.exception('exception calling callback for %r', self)

    def result(self, timeout=None):
        """Return the result of the call that the future represents.

        Args:
            timeout: The number of seconds to wait for the result if the future
                isn't done. If None, then there is no limit on the wait time.

        Returns:
            The result of the call that the future represents.

        Raises:
            CancelledError: If the future was cancelled.
            TimeoutError: If the future didn't finish executing before the given
                timeout.
            Exception: If the call raised then that exception will be raised.
        """
        try:
            if self._wait(timeout):
                return self.__get_result()
            raise TimeoutError()
        finally:
            # Break a reference cycle with the exception in self._exception
            self = None

    def exception(self, timeout=None):
        """Return the exception raised by the call that the future represents.

        Args:
            timeout: The number of seconds to wait for the exception if the
                future isn't done. If None, then there is no limit on the wait
                time.

        Returns:
            The exception raised by the call that the future represents or None
            if the call completed without raising.

        Raises:
            CancelledError: If the future was cancelled.
            TimeoutError: If the future didn't finish executing before the given
                timeout.
        """
        if self._wait(timeout):
            return self._value if self._was_exception else None
        raise TimeoutError()

    # called on user thread to modify future state
    def _set_value(self, value, was_exception):
        """Sets the return value of work associated with the future.

        Should only be used by Executor implementations and unit tests.
        """
        if self._complete:
            raise ValueError('Future already completed')
        self._complete = True
        self._value = value
        self._was_exception = was_exception
        self._invoke_callbacks()

    # called from context event loop

    def set_exception(self, exception):
        self._context._finished_futures[-1].append((self, exception, True))

    def set_result(self, result):
        self._context._finished_futures[-1].append((self, result, False))

    __class_getitem__ = classmethod(types.GenericAlias)

_enumerate = enumerate

def _create_dict(futures, enumerate):
    context = None
    completed = []
    completed_exception = False
    d = {}
    for i, f in _enumerate(futures):
        v = (i, f) if enumerate else f
        context = f._context
        if f.done():
            completed.append(v)
            completed_exception |= f._was_exception
        else:
            d[f] = v
    return context, completed, completed_exception, d

def as_completed(futures, timeout=None, enumerate=False):
    ctx, completed, _, d = _create_dict(futures, enumerate)
    if not ctx:
        return
    yield from completed
    for futs in ctx._generate_futures(timeout):
        to_yield = []
        for fut, value, was_exception in futs:
            fut._set_value(value, was_exception)
            v = d.pop(fut, None)
            if v is not None:
                to_yield.append(v)
        yield from to_yield
        if not d:
            return
    raise TimeoutError()

FIRST_COMPLETED = lambda completed, completed_exception, d: completed
FIRST_EXCEPTION = lambda completed, completed_exception, d: completed_exception
ALL_COMPLETED = lambda completed, completed_exception, d: not d

def wait(futures, timeout=None, return_when=ALL_COMPLETED, enumerate=False):
    ctx, completed, completed_exception, d = _create_dict(futures, enumerate)
    if not ctx:
        return [], []
    if return_when(completed, completed_exception, d):
        return completed, list(d.values())
    for futs in ctx._generate_futures(timeout):
        for fut, value, was_exception in futs:
            fut._set_value(value, was_exception)
            v = d.pop(fut, None)
            if v is not None:
                completed.append(v)
                completed_exception |= was_exception
        if return_when(completed, completed_exception, d):
            return completed, list(d.values())
    raise TimeoutError()

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
        self._state = 'unconnected'
        self._deferred_sends = []
        self._proc_table = weakref.WeakValueDictionary()
        self._on_connection_lost = Future(context)

    def _heartbeat(self):
        self.expiry = time.time() + HEARTBEAT_INTERVAL * HEARTBEAT_LIVENESS

    def _connect(self, name):
        self._context._name_to_host[name] = self
        self._state = 'connected'
        self._name = name
        for msg in self._deferred_sends:
            self._context._backend.send_multipart([self._name, msg])
        self._deferred_sends.clear()

    def _disconnect(self):
        self._state = 'lost'
        for p in self._proc_table.values():
            p._lost_host()

        # is there value in keeping this around for a reconnect?
        self._proc_table.clear()
        self._on_connection_lost.set_result(None)

    def _debug_dict(self):
        return self.__dict__

    def _send(self, msg):
        if self._state == 'unconnected':
            self._deferred_sends.append(msg)
        else:
            self._context._backend.send_multipart([self._name, msg])

    def _launch(self, p):
        if self._state == "lost":
            # launch after we lost connection to this host.
            p._lost_host()
            return
        self._proc_table[p._id] = p
        self._send(pickle.dumps(('launch', p._id, p.rank, p.world_size, p.args, p.simulate)))
        self._context._launches += 1

    def __repr__(self):
        return f"Host({self._name})"

    def on_connection_lost(self):
        return self._on_connection_lost

    def connection_lost(self):
        return self._on_connection_lost.done()

class Process:
    def __init__(self, context, host, rank, world_size, args, simulate):
        self._id = context._next_id
        context._next_id += 1
        self._context = context
        self.host = host
        self.rank = rank
        self.args = args
        self.simulate = simulate
        self.world_size = world_size
        self._pid = Future(context)
        self._returncode = Future(context)
        self._recvs = []
        self._messages = []
        self._state = 'launched'

    def returncode(self) -> 'Future[int]':
        return self._returncode

    def pid(self) -> 'Future[int]':
        return self._pid

    def __repr__(self):
        pid = self._pid.result() if self._pid.done() and not self._pid.exception() else None
        return f"Process(rank={self.rank}, host={self.host}, pid={pid})"

    def _lost_host(self):
        e = ConnectionAbortedError("Lost connection to process host")
        if self._state == 'launched':
            self._pid.set_exception(e)
        if self._state in ['launched', 'running']:
            self._returncode.set_exception(e)
        for _, f in self._recvs:
            f.set_exception(e)
        self._recvs.clear()
        self._state = 'aborted'

    def send(self, msg: Any):
        self._context._schedule(lambda: self._send(msg))

    def _send(self, msg: Any):
        if self._state != 'aborted':
            self.host._send(pickle.dumps(('send', self._id, msg)))

    def signal(self, signal=signal.SIGTERM, group=True):
        self._context._schedule(lambda: self._signal(signal, group))

    def _signal(self, signal, group):
        if self._state != 'aborted':
            self.host._send(pickle.dumps(('signal', self._id, signal, group)))

    # return first response where filter(msg) is True
    def recv(self, filter: callable=lambda x: True) -> 'Future[Any]':
        fut = Future(self._context)
        self._context._schedule(lambda: self._recv(fut, filter))
        return fut

    def _recv(self, fut: Future, filter: callable):
        for i, msg in enumerate(self._messages):
            if filter(msg):
                self._messages.pop(i)
                fut.set_result(msg)
                return
        if self._state == 'aborted':
            fut.set_exception(ConnectionAbortedError("Lost connection to process host"))
        else:
            self._recvs.append((filter, fut))

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
        self._state = 'exited'
        self._returncode.set_result(returncode)
        self._context._exits += 1

    def _started(self, pid):
        self._state = 'running'
        self._pid.set_result(pid)

class Context:
    def __init__(self):
        self._context = zmq.Context(1)

        # to talk to python clients in this process
        self._requests = deque()
        self._finished_futures = deque([[]])
        self._requests_ready = self._context.socket(zmq.PAIR)
        self._requests_ready.bind('inproc://doorbell')
        self._doorbell = self._context.socket(zmq.PAIR)
        self._doorbell.connect('inproc://doorbell')
        self._doorbell_poller = zmq.Poller()
        self._doorbell_poller.register(self._doorbell, zmq.POLLIN)

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
        self._next_id = 0
        self._exits = 0
        self._launches = 0

    def _event_loop(self):
        _time_poll = 0
        _time_process = 0
        while True:
            time_begin = time.time()
            poll_result = self._poller.poll(timeout=HEARTBEAT_INTERVAL*1000*2)
            time_poll = time.time()
            if self._finished_futures[-1]:
                self._finished_futures.append([])
                self._requests_ready.send(b'')
            for sock, _ in poll_result:
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
                    else:
                        host = self._name_to_host[f]
                    host._heartbeat()
                    if host._state == 'lost':
                        # got a message from a host that expired, but
                        # eventually came back to life
                        # At this point we've marked its processes as dead
                        # so we are going to tell it to abort so that it gets
                        # restarted and can become a new connection.
                        logger.info("Host %s that was lost reconnected, sending abort", host._name)
                        self._send_abort(host, True)
                    elif len(msg):
                        cmd, proc_id, *args = pickle.loads(msg)
                        proc = host._proc_table.get(proc_id)
                        if proc is None:
                            # messages from a process might arrive after the user
                            # no longer has a handle to the Process object
                            # in which case they are ok to just drop
                            assert proc_id >= 0 and proc_id < self._next_id, "unexpected proc_id"
                            logger.info("Received message %s from process %s after local handle deleted", cmd, proc_id)
                        else:
                            getattr(proc, cmd)(*args)

                elif sock is self._requests_ready:
                    while self._requests:
                        self._requests_ready.recv()
                        fn = self._requests.popleft()
                        fn()
            time_end = time.time()
            _time_poll += time_poll - time_begin
            _time_process += time_end - time_poll
            t = time.time()
            if t - self._last_heartbeat_check > HEARTBEAT_INTERVAL*2:
                elapsed = t - self._last_heartbeat_check
                self._last_heartbeat_check = t
                # priority queue would be log(N)
                for key, host in self._name_to_host.items():
                    if host._state == 'connected' and host.expiry < t:
                        # host timeout
                        logger.warning("Host %s has not heartbeated in %s seconds, disconnecting it", host._name, elapsed)
                        host._disconnect()
                self.logstatus(_time_poll / elapsed, _time_process / elapsed)
                _time_poll = 0
                _time_process = 0

    def logstatus(self, poll_fraction, active_fraction):
        host_histogram = {}
        for h in self._name_to_host.values():
            host_histogram[h._state] = host_histogram.setdefault(h._state, 0) + 1
        logger.info("supervisor status: %s process launches, %s exits, %s host handles without hosts, %s connected hosts with handles, %s time spent polling, %s time spent active, hosts %s",
         self._launches, self._exits, len(self._unassigned_hosts), len(self._unassigned_connections), poll_fraction*100, active_fraction*100, host_histogram)


    def _debug_dict(self):
        return self.__dict__

    def _schedule(self, fn):
        self._requests.append(fn)
        self._doorbell.send(b'')

    def _send_abort(self, host, with_error):
        self._backend.send_multipart([host._name, pickle.dumps(('abort', True))])

    def request_hosts(self, n: int) -> 'Future[List[Host]]':
        """
        Request from the scheduler n hosts to run processes on.
        The future is fulfilled when the reservation is made, but
        potenially before all the hosts check in with this API.

        Note: implementations that use existing slurm-like schedulers,
        will immediately full the future because the reservation was
        already made.
        """
        f = Future(self)
        hosts = tuple(Host(self) for i in range(n))
        f.set_result(hosts)
        self._schedule(lambda: self._request_hosts(hosts))
        return f


    def _next_connection(self):
        while self._unassigned_connections:
            host = self._unassigned_connections.popleft()
            # its possible this connection timed out in the
            # meantime
            if host._state == 'connected':
                return host
        return None

    def _request_host(self, h):
        u = self._next_connection()
        if u is None:
            self._unassigned_hosts.append(h)
        else:
            h._connect(u._name)

    def _request_hosts(self, hosts):
        for h in hosts:
            self._request_host(h)

    def return_hosts(self, hosts: List[Host]):
        """
        Processes on the returned hosts will be killed,
        and future processes launches with the host will fail.
        """
        self._schedule(lambda: self._return_hosts(hosts))

    def _return_hosts(self, hosts: List[Host]):
        for h in hosts:
            # XXX: this fails to return a host if it was
            # not connected when it was returned.
            if h._state == 'connected':
                self._send_abort(h, False)
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
        self._schedule(lambda: self._replace_hosts(hosts))

    def _replace_hosts(self, hosts):
        for h in hosts:
            if h._state == 'connected':
                self._send_abort(h, True)
                h._disconnect()
            # detach this host object from current name
            old = self._name_to_host[h._name] = Host(self)
            old._disconnect()
            h.__init__(self)
            # let it get assigned to the next host to checkin
            self._request_host(h)



    # TODO: other arguments like environment, etc.
    def create_process_group(self, hosts: List[Host], args, npp=1, simulate=False) -> List[Process]:
        world_size = npp*len(hosts)
        procs = tuple(Process(self, h, i*npp + j, world_size, args, simulate) for i, h in enumerate(hosts) for j in range(npp))
        self._schedule(lambda: self._launch_processes(procs))
        return procs

    def _launch_processes(self, procs):
        for p in procs:
            p.host._launch(p)

    def _generate_futures(self, timeout=None) -> 'Generator[List[Future]]':
        if timeout is None:
            while True:
                self._doorbell.recv()
                yield self._finished_futures.popleft()
        elif timeout == 0:
            while self._doorbell_poller.poll(0):
                self._doorbell.recv()
                yield self._finished_futures.popleft()
        else:
            t = time.time()
            expiry = t + timeout
            while t < expiry:
                if self._doorbell_poller.poll(expiry - t):
                    self._doorbell.recv()
                    yield self._finished_futures.popleft()
                t = time.time()


def get_message_queue():
    """
    Processes launched on the hosts can use this function to connect
    to the messaging queue of the supervisor.

    Messages send from here can be received by the supervisor using
    `proc.recv()` and messages from proc.send() will appear in this queue.
    """
    ctx = zmq.Context(1)
    sock = ctx.socket(zmq.DEALER)
    proc_id = int(os.environ['SUPERVISOR_IDENT']).to_bytes(8, byteorder='little')
    sock.setsockopt(zmq.IDENTITY, proc_id)
    sock.connect(os.environ['SUPERVISOR_PIPE'])
    sock.send(b'')
    return sock
