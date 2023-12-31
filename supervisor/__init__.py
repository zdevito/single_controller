# pyre-strict
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
import io
import traceback
from typing import (
    NamedTuple,
    Generic,
    TypeVar,
    Tuple,
    Callable,
    Sequence,
    Iterable,
    Optional,
    Union,
    Set,
    Iterator,
    Dict,
    Mapping,
)
from enum import Enum
from pathlib import Path
import sys
from logging import Logger

logger: Logger = logging.getLogger(__name__)

T = TypeVar("T")

HEARTBEAT_LIVENESS = 3  # 3..5 is reasonable
HEARTBEAT_INTERVAL = 1.0


def _get_hostname(hostname_future: "Future[str]") -> str:
    if hostname_future.done() and hostname_future.exception() is None:
        return repr(hostname_future.result())
    else:
        return "unconnected"


Callback = Callable[
    [
        T,
    ],
    None,
]


class Future(Generic[T]):
    """Represents the result of an asynchronous computation."""

    def __init__(
        self, context: "Context", name: str, hostname_future: "Optional[Future[str]]"
    ) -> None:
        """Initializes the future. Should not be called by clients."""
        self._context = context
        self._complete = False
        self._value: Union[T, None, BaseException] = None
        self._was_exception = False
        self._done_callbacks: "List[Callback]" = []
        self._name = name
        self._hostname_future = hostname_future

    def __repr__(self) -> str:
        status = (
            "exception"
            if self._was_exception
            else "complete"
            if self._complete
            else "incomplete"
        )
        hostname = self._hostname_future
        if hostname is None:
            return f"Future[{status}, {self._name}]"

        hostname = _get_hostname(hostname)
        return f"Future[{status}, hosts[{hostname}].{self._name}]"

    def done(self) -> bool:
        # wait 0 just polls to see if the done message
        # is already in the message queue
        return self._wait(0)

    def add_done_callback(self, fn: Callback) -> None:
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
                logger.exception("exception calling callback for %r", self)

    def __get_result(self) -> T:
        if self._was_exception:
            try:
                # pyre-ignore
                raise self._value
            finally:
                # Break a reference cycle with the exception in self._exception
                self = None
        else:
            # pyre-ignore
            return self._value

    def _wait(self, timeout: Optional[float]) -> bool:
        if self._complete:
            return True
        for _ in self._context._process_futures(timeout, lambda: str((self,))):
            if self._complete:
                return True
        return False

    def _invoke_callbacks(self) -> None:
        for callback in self._done_callbacks:
            try:
                callback(self)
            except Exception:
                logger.exception("exception calling callback for %r", self)
        self._done_callbacks.clear()

    def result(self, timeout: Optional[float] = None) -> T:
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

    def exception(self, timeout: Optional[float] = None) -> Optional[BaseException]:
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
            # pyre-ignore
            return self._value if self._was_exception else None
        raise TimeoutError()

    # called on user thread to modify future state
    def _set_value(
        self, value: Union[T, BaseException], was_exception: bool
    ) -> Callable[[], None]:
        """Sets the return value of work associated with the future.

        Should only be used by Executor implementations and unit tests.
        """
        assert not self._complete, f"Future {self} already completed"
        self._complete = True
        self._value = value
        self._was_exception = was_exception
        return self._invoke_callbacks

    # called from context event loop

    def set_exception(self, exception: BaseException) -> None:
        # pyre-ignore
        self._context._finished_futures[-1].append((self, exception, True))

    def set_result(self, result: object) -> None:
        # pyre-ignore
        self._context._finished_futures[-1].append((self, result, False))


class as_completed:
    def __init__(
        self, futures: Sequence[Future[object]] = (), timeout: Optional[float] = None
    ) -> None:
        self._timeout = timeout
        self._not_done: Set[Future[object]] = set()
        self._worklist: deque[Future[object]] = deque()
        self._ctx: "Optional[Context]" = None
        self.update(futures)

    def add(self, fut: Future[object]) -> None:
        self._not_done.add(fut)
        self._ctx = fut._context
        append = self._worklist.append
        if fut._complete:
            append(fut)
        else:
            fut._done_callbacks.append(append)

    def update(self, futures: Sequence[Future[object]]) -> None:
        for f in futures:
            self.add(f)

    def __len__(self) -> int:
        return len(self._not_done)

    def __iter__(self) -> Iterator[Future[object]]:
        ctx = self._ctx
        if not ctx:
            return
        not_done = self._not_done
        worklist = self._worklist
        for _ in ctx._process_futures(self._timeout, lambda: str(not_done)):
            while worklist:
                f = worklist.popleft()
                not_done.remove(f)
                yield f
            if not not_done:
                return
        raise TimeoutError()


ReturnWhenType = Callable[[Future[object]], bool]
FIRST_COMPLETED: ReturnWhenType = lambda fut: True
FIRST_EXCEPTION: ReturnWhenType = lambda fut: fut._was_exception
ALL_COMPLETED: ReturnWhenType = lambda fut: False


class _WaitResult(NamedTuple):
    done: set[Future[object]]
    not_done: set[Future[object]]


_State = Enum("_State", ["UNATTACHED", "ATTACHED", "LOST"])
_UNATTACHED: _State = _State.UNATTACHED
_ATTACHED: _State = _State.ATTACHED
_LOST: _State = _State.LOST


def wait(
    futures: Sequence[Future[object]],
    timeout: Optional[float] = None,
    return_when: ReturnWhenType = ALL_COMPLETED,
) -> _WaitResult:
    gen = as_completed(futures, timeout)
    done = set()
    for fut in gen:
        done.add(fut)
        if return_when(fut):
            break
    return _WaitResult(done, gen._not_done)


class Connection:
    def __init__(self, ctx: "Context", name: bytes, hostname: Optional[str]) -> None:
        self.state: _State = _UNATTACHED
        self.name = name
        self.hostname = hostname
        self.host: "Optional[Host]" = None
        self.heartbeat()
        if hostname is None:
            self.lost(ctx, "Connection did not start with a hostname")
        else:
            # let the connection know we exist
            ctx._backend.send_multipart([name, b""])

    def heartbeat(self) -> None:
        self.expiry = time.time() + HEARTBEAT_INTERVAL * HEARTBEAT_LIVENESS

    def check_alive_at(self, ctx: "Context", t: float) -> None:
        if self.state is not _LOST and self.expiry < t:
            # host timeout
            elapsed = t - self.expiry + HEARTBEAT_INTERVAL * HEARTBEAT_LIVENESS
            logger.warning(
                "Host %s (%s) has not heartbeated in %s seconds, disconnecting it",
                self.hostname, self.name,
                elapsed,
            )
            self.lost(ctx, "Host did not heartbeat")

    def handle_message(self, ctx: "Context", msg: bytes) -> None:
        self.heartbeat()
        if self.state is _LOST:
            # got a message from a host that expired, but
            # eventually came back to life
            # At this point we've marked its processes as dead
            # so we are going to tell it to abort so that it gets
            # restarted and can become a new connection.
            logger.info("Host %s that was lost reconnected, sending abort", self.name)
            self.send_abort(ctx, "Supervisor thought host timed out")
            return

        if not len(msg):
            # heartbeat, respond with our own
            ctx._backend.send_multipart([self.name, b""])
            return

        if self.state is _UNATTACHED:
            logger.warning(
                f"Got message from host %s manager before it was attached.", self.name
            )
            self.lost(ctx, "Host manager sent messages before attached.")
            return

        cmd, proc_id, *args = pickle.loads(msg)
        assert self.host is not None
        receiver = self.host if proc_id is None else self.host._proc_table.get(proc_id)
        if receiver is None:
            # messages from a process might arrive after the user
            # no longer has a handle to the Process object
            # in which case they are ok to just drop
            assert proc_id >= 0 and proc_id < ctx._next_id, "unexpected proc_id"
            logger.debug(
                "Received message %s from process %s after local handle deleted",
                cmd,
                proc_id,
            )
        else:
            getattr(receiver, cmd)(*args)
            receiver = None

    def lost(self, ctx: "Context", with_error: Optional[str]) -> None:
        orig_state = self.state
        if orig_state is _LOST:
            return
        self.state = _LOST
        if orig_state is _ATTACHED:
            assert self.host is not None
            self.host._lost(with_error)
            self.host = None
        self.send_abort(ctx, with_error)

    def send_abort(self, ctx: "Context", with_error: Optional[str]) -> None:
        ctx._backend.send_multipart([self.name, pickle.dumps(("abort", with_error))])


class Host:
    def __init__(self, context: "Context") -> None:
        self._context = context
        self._state: _State = _UNATTACHED
        self._name: Optional[bytes] = None
        self._deferred_sends: List[bytes] = []
        self._proc_table: weakref.WeakValueDictionary[
            int, Process
        ] = weakref.WeakValueDictionary()
        self._hostname_future = Future[str](context, "hostname", None)
        self._on_connection_lost = Future[float](
            context, "host_connection_lost", self._hostname_future
        )

    def __repr__(self) -> str:
        return f"Host[{_get_hostname(self._hostname_future)}]"

    def hostname(self) -> Future[str]:
        return self._hostname_future

    def _lost(self, msg: Optional[str]) -> None:
        orig_state = self._state
        if orig_state is _LOST:
            return
        self._state = _LOST
        if orig_state is _ATTACHED:
            self._context._name_to_connection[self._name].lost(self._context, msg)
        else:
            self._hostname_future.set_exception(
                ConnectionAbortedError("Lost connection to process host")
            )
        self._name = None
        self._deferred_sends.clear()
        for p in self._proc_table.values():
            p._lost_host()
        # is there value in keeping this around for a reconnect?
        self._proc_table.clear()
        self._on_connection_lost.set_result(time.time())

    def _send(self, msg: bytes) -> None:
        if self._state is _ATTACHED:
            self._context._backend.send_multipart([self._name, msg])
        elif self._state is _UNATTACHED:
            self._deferred_sends.append(msg)

    def _launch(self, p: "Process") -> None:
        if self._state is _LOST:
            # launch after we lost connection to this host.
            p._lost_host()
            return
        self._proc_table[p._id] = p
        self._send(
            pickle.dumps(
                (
                    "launch",
                    p._id,
                    p.rank,
                    p.processes_per_host,
                    p.world_size,
                    p.popen,
                    p.name,
                    p.simulate,
                    p.logfile,
                )
            )
        )
        self._context._launches += 1

    def time_connection_lost(self) -> Future[float]:
        return self._on_connection_lost

    def connection_lost(self) -> bool:
        return self._on_connection_lost.done()


class ProcessFailedToStart(Exception):
    pass


class Process:
    def __init__(
        self,
        context: "Context",
        host: "Host",
        rank: int,
        processes_per_host: int,
        world_size: int,
        popen: Mapping[str, object],
        name: str,
        simulate: bool,
    ) -> None:
        self._id: int = context._next_id
        context._next_id += 1
        self._context = context
        self.host = host
        self.rank = rank
        self.processes_per_host = processes_per_host
        self.world_size = world_size
        self.popen = popen
        self.simulate = simulate
        self.name: str = name.format(rank=str(rank).zfill(len(str(world_size))))
        self.logfile: Optional[str] = (
            None
            if context._log_format is None
            else context._log_format.format(name=self.name)
        )
        hostname = self.host.hostname()
        self._pid = Future[int](context, f"proc[{self.name!r}].pid()", hostname)
        self._returncode = Future[int](context, f"proc[{self.name!r}].returncode()", hostname)

        self._recvs: List[Tuple[Callable[[object], bool], Future[object]]] = []
        self._messages: List[object] = []
        self._state = "launched"

    def returncode(self) -> "Future[int]":
        return self._returncode

    def pid(self) -> "Future[int]":
        return self._pid

    def __repr__(self) -> str:
        pid = (
            self._pid.result()
            if self._pid.done() and not self._pid.exception()
            else None
        )
        return f"Process(rank={self.rank}, host={self.host}, pid={pid})"

    def _lost_host(self) -> None:
        self._abort(ConnectionAbortedError("Lost connection to process host"))

    def _abort(self, e: BaseException) -> None:
        if self._state == "launched":
            self._pid.set_exception(e)
        if self._state in ["launched", "running"]:
            self._returncode.set_exception(e)
        for _, f in self._recvs:
            f.set_exception(e)
        self._recvs.clear()
        self._state = "aborted"

    def send(self, msg: object) -> None:
        self._context._schedule(lambda: self._send(msg))

    def _send(self, msg: object) -> None:
        if self._state != "aborted":
            self._context._sends += 1
            self.host._send(pickle.dumps(("send", self._id, msg)))

    def signal(self, signal: int = signal.SIGTERM, group: bool = True) -> None:
        self._context._schedule(lambda: self._signal(signal, group))

    def _signal(self, signal: int, group: bool) -> None:
        if self._state != "aborted":
            self.host._send(pickle.dumps(("signal", self._id, signal, group)))

    # return first response where filter(msg) is True
    def recv(
        self, filter: Callable[[object], bool] = lambda x: True
    ) -> "Future[object]":
        hostname = self.host.hostname()
        fut = Future(self._context, f"proc[{self.name!r}].recv()", hostname)
        self._context._schedule(lambda: self._recv(fut, filter))
        return fut

    def _recv(self, fut: Future[object], filter: Callable[[object], bool]) -> None:
        for i, msg in enumerate(self._messages):
            if filter(msg):
                self._messages.pop(i)
                fut.set_result(msg)
                return
        if self._state == "aborted":
            fut.set_exception(ConnectionAbortedError("Lost connection to process host"))
        else:
            self._recvs.append((filter, fut))

    # TODO: annotation that registers this as a valid
    # message that can be sent

    def _response(self, msg: bytes) -> None:
        self._context._responses += 1
        msg = pickle.loads(msg)
        for i, (filt, fut) in enumerate(self._recvs):
            if filt(msg):
                self._recvs.pop(i)
                fut.set_result(msg)
                return
        self._messages.append(msg)

    def _exited(self, returncode: int) -> None:
        self._state = "exited"
        self._returncode.set_result(returncode)
        self._context._exits += 1

    def _started(self, pid: Union[str, int]) -> None:
        if isinstance(pid, int):
            self._state = "running"
            self._pid.set_result(pid)
        else:
            self._abort(ProcessFailedToStart(pid))

    def __del__(self) -> None:
        self._context._proc_deletes += 1


def _check_for_hostname(msg: bytes) -> Optional[str]:
    if not len(msg):
        return None
    try:
        cmd, _, hostname = pickle.loads(msg)
        if cmd != "_hostname" or not isinstance(hostname, str):
            return None
        return hostname
    except:
        return None


class Context:
    def __init__(self, port: int = 55555, log_format: Optional[str] = None, log_interval: float=10) -> None:
        if log_format is not None:
            path = log_format.format(name="supervisor")
            logger.info(f"Redirect logging to {path}")
            Path(path).parent.mkdir(exist_ok=True, parents=True)
            with open(path, "w") as f:
                os.dup2(f.fileno(), sys.stdout.fileno())
                os.dup2(f.fileno(), sys.stderr.fileno())
        self._log_interval: float = log_interval
        self._context: zmq.Context = zmq.Context(1)

        # to talk to python clients in this process
        self._requests: deque[Callable[[], None]] = deque()
        self._finished_futures = deque[
            List[Tuple[Future[object], Union[object, BaseException], bool]]
        ]([[]])
        self._requests_ready: zmq.Socket = self._context.socket(zmq.PAIR)
        self._requests_ready.bind("inproc://doorbell")
        self._doorbell: zmq.Socket = self._context.socket(zmq.PAIR)
        self._doorbell.connect("inproc://doorbell")
        self._doorbell_poller = zmq.Poller()
        self._doorbell_poller.register(self._doorbell, zmq.POLLIN)

        # to talk to other hosts

        self._backend: zmq.Socket = self._context.socket(zmq.ROUTER)
        self._backend.setsockopt(zmq.IPV6, True)
        self._backend.bind(f"tcp://*:{port}")

        self._poller = zmq.Poller()
        self._poller.register(self._backend, zmq.POLLIN)
        self._poller.register(self._requests_ready, zmq.POLLIN)

        self._unassigned_hosts: deque[Host] = deque()
        self._unassigned_connections: deque[Connection] = deque()
        self._name_to_connection: Dict[bytes, Connection] = {}
        self._last_heartbeat_check: float = time.time()
        self._last_logstatus: float = self._last_heartbeat_check
        self._next_id = 0
        self._exits = 0
        self._sends = 0
        self._responses = 0
        self._launches = 0
        self._proc_deletes = 0
        self._exit_event_loop = False
        self._pg_name = 0
        self._log_format = log_format

        self._thread = Thread(target=self._event_loop, daemon=True)
        self._thread.start()

    def _attach(self) -> None:
        while self._unassigned_connections and self._unassigned_hosts:
            c = self._unassigned_connections[0]
            h = self._unassigned_hosts[0]
            if c.state is _LOST:
                self._unassigned_connections.popleft()
            elif h._state is _LOST:
                self._unassigned_hosts.popleft()
            else:
                self._unassigned_connections.popleft()
                self._unassigned_hosts.popleft()
                c.host = h
                h._name = c.name
                h._hostname_future.set_result(c.hostname)
                h._state = c.state = _ATTACHED
                for msg in h._deferred_sends:
                    self._backend.send_multipart([h._name, msg])
                h._deferred_sends.clear()

    def _event_loop(self) -> None:
        _time_poll = 0
        _time_process = 0
        while True:
            time_begin = time.time()
            poll_result = self._poller.poll(timeout=int(HEARTBEAT_INTERVAL * 1000))
            time_poll = time.time()
            for sock, _ in poll_result:
                if sock is self._backend:
                    f, msg = self._backend.recv_multipart()
                    if f not in self._name_to_connection:
                        hostname = _check_for_hostname(msg)
                        connection = self._name_to_connection[f] = Connection(
                            self, f, hostname
                        )
                        self._unassigned_connections.append(connection)
                        self._attach()
                    else:
                        self._name_to_connection[f].handle_message(self, msg)
                elif sock is self._requests_ready:
                    while self._requests:
                        self._requests_ready.recv()
                        fn = self._requests.popleft()
                        fn()
                        del fn  # otherwise we old a handle until
                        # the next time we run a command
            if self._exit_event_loop:
                return
            t = time.time()
            elapsed = t - self._last_heartbeat_check
            should_check_heartbeat = elapsed > HEARTBEAT_INTERVAL * HEARTBEAT_LIVENESS
            if should_check_heartbeat:
                self._last_heartbeat_check = t
                # priority queue would be log(N)
                for connection in self._name_to_connection.values():
                    connection.check_alive_at(self, t)

            # Marking futures ready should always happen at the end of processing events above
            # to unblock anything processing the futures, before we start waiting for more events.
            if self._finished_futures[-1]:
                self._finished_futures.append([])
                self._requests_ready.send(b"")

            time_end = time.time()
            _time_poll += time_poll - time_begin
            _time_process += time_end - time_poll

            elapsed = t - self._last_logstatus
            if elapsed > self._log_interval:
                self._last_logstatus = t
                self._logstatus(_time_poll / elapsed, _time_process / elapsed)
                _time_poll = 0
                _time_process = 0

    def _logstatus(self, poll_fraction: float, active_fraction: float) -> None:
        connection_histogram = {}
        for connection in self._name_to_connection.values():
            state = connection.state.name
            connection_histogram[state] = connection_histogram.setdefault(state, 0) + 1
        logger.info(
            "supervisor status: %s process launches, %s exits, %s message sends, %s message responses, %s process __del__, %s hosts waiting for connections, %s connections waiting for handles, time is %.2f%% polling and %.2f%% active, connections %s",
            self._launches,
            self._exits,
            self._sends,
            self._responses,
            self._proc_deletes,
            len(self._unassigned_hosts),
            len(self._unassigned_connections),
            poll_fraction * 100,
            active_fraction * 100,
            connection_histogram,
        )

    def _schedule(self, fn: Callable[[], None]) -> None:
        self._requests.append(fn)
        self._doorbell.send(b"")

    def request_hosts(self, n: int) -> "Tuple[Host, ...]":
        """
        Request from the scheduler n hosts to run processes on.
        The future is fulfilled when the reservation is made, but
        potenially before all the hosts check in with this API.

        Note: implementations that use existing slurm-like schedulers,
        will immediately full the future because the reservation was
        already made.
        """
        hosts = tuple(Host(self) for i in range(n))
        self._schedule(lambda: self._request_hosts(hosts))
        return hosts

    def _request_host(self, h: Host) -> None:
        self._unassigned_hosts.append(h)
        self._attach()

    def _request_hosts(self, hosts: Sequence[Host]) -> None:
        for h in hosts:
            self._request_host(h)

    def return_hosts(self, hosts: Sequence[Host], error: Optional[str] = None) -> None:
        """
        Processes on the returned hosts will be killed,
        and future processes launches with the host will fail.
        """
        self._schedule(lambda: self._return_hosts(hosts, error))

    def _return_hosts(self, hosts: Sequence[Host], error: Optional[str]) -> None:
        for h in hosts:
            h._lost(error)

    def replace_hosts(self, hosts: Sequence[Host]) -> "Tuple[Host, ...]":
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
        hosts = list(hosts)
        self.return_hosts(hosts)
        return self.request_hosts(len(hosts))

    def _shutdown(self) -> None:
        self._exit_event_loop = True
        for connection in self._name_to_connection.values():
            connection.lost(self, None)

    def shutdown(self) -> None:
        self._schedule(self._shutdown)
        self._thread.join()
        self._backend.close()
        self._requests_ready.close()
        self._doorbell.close()
        self._context.term()

    # TODO: other arguments like environment, etc.
    def create_process_group(
        self,
        hosts: Sequence[Host],
        args: Sequence[str],
        processes_per_host: int = 1,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
        name: Optional[str] = None,
        simulate: bool = False,
    ) -> Tuple[Process, ...]:
        world_size = processes_per_host * len(hosts)
        if name is None:
            name = f"pg{self._pg_name}"
            self._pg_name += 1
        logger.info("Starting process group %r with %d processes (%s hosts * %s processes per host)", name, world_size, len(hosts), processes_per_host)
        popen = {"args": args, "env": env, "cwd": cwd}
        procs = tuple(
            Process(
                self,
                h,
                i * processes_per_host + j,
                processes_per_host,
                world_size,
                popen,
                name,
                simulate,
            )
            for i, h in enumerate(hosts)
            for j in range(processes_per_host)
        )
        self._schedule(lambda: self._launch_processes(procs))
        return procs

    def _launch_processes(self, procs: Sequence[Process]) -> None:
        for p in procs:
            p.host._launch(p)

    def _process_futures(
        self,
        timeout: Optional[float],
        remaining_futures_cb: Callable[[], str],
        ttl_report_interval: float = 5,
    ) -> Iterator[int]:
        """
        Return a generator that completes futures. Yields the number of futures it has
        processed in each step, and will stop iterating when timeout is reached.
        """

        def read_futures() -> int:
            self._doorbell.recv()
            futs = self._finished_futures.popleft()
            # All `futs` need to be marked complete before
            # we run any callbacks, because a callback may recursively wait
            # on a future in futs, and re-entering _process_futures won't unblock it.
            callbacks = [
                f._set_value(value, was_exception) for f, value, was_exception in futs
            ]
            for c in callbacks:
                c()
            return len(futs)

        t = time.time()
        # by always yielding right after starting the timer
        # we allow the already-done futures to be returned
        # from the caller while still tracking total time
        # allowed to be waiting here
        yield 0
        if timeout is None:
            while True:
                yield read_futures()
        else:
            expiry = t + timeout
            while t < expiry:
                if self._doorbell_poller.poll(
                    timeout=int(1000 * min(ttl_report_interval, expiry - t))
                ):
                    yield read_futures()
                elif ttl_report_interval < expiry - t:
                    s = io.StringIO()
                    traceback.print_stack(file=s)
                    logger.info(
                        f"Waiting for {remaining_futures_cb()} futures, {expiry - t} seconds before timeout:\n{s.getvalue()}"
                    )
                t = time.time()
            while self._doorbell_poller.poll(0):
                yield read_futures()


def get_message_queue(
    supervisor_ident: Optional[int] = None, supervisor_pipe: Optional[str] = None
) -> zmq.Socket:
    """
    Processes launched on the hosts can use this function to connect
    to the messaging queue of the supervisor.

    Messages send from here can be received by the supervisor using
    `proc.recv()` and messages from proc.send() will appear in this queue.
    """
    if supervisor_ident is None:
        supervisor_ident = int(os.environ["SUPERVISOR_IDENT"])
    if supervisor_pipe is None:
        supervisor_pipe = os.environ["SUPERVISOR_PIPE"]
    ctx = zmq.Context(1)
    sock = ctx.socket(zmq.DEALER)
    proc_id = supervisor_ident.to_bytes(8, byteorder="little")
    sock.setsockopt(zmq.IDENTITY, proc_id)
    sock.connect(supervisor_pipe)
    sock.send(b"")
    return sock
