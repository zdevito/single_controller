# pyre-strict
import zmq
import sys
import time
import os
import io
import traceback
import pickle
import subprocess
from typing import Mapping, Optional, Tuple, List, Dict
import ctypes
from supervisor import HEARTBEAT_INTERVAL, HEARTBEAT_LIVENESS, ProcessFailedToStart
import signal
import logging
import socket
from contextlib import nullcontext


logger: logging.Logger = logging.getLogger()
ABORT_INTERVAL = 5
__NR_pidfd_open = 434
libc = ctypes.CDLL(None)


# older libc do not have this syscall
def pidfd_open(pid: int) -> int:
    return libc.syscall(__NR_pidfd_open, pid, 0)


# objects in this file represent Host/Process
# on the host machine itself.

# main package has Host/Process used by
# the supervisor.


class Process:
    def __init__(
        self,
        name: str,
        logfilename: Optional[str],
        proc_comm: zmq.Socket,
        proc_id: int,
        rank: int,
        processes_per_host: int,
        world_size: int,
        popen: Mapping[str, object],
        proc_addr: str,
    ) -> None:
        self.proc_id = proc_id
        self.proc_comm = proc_comm
        environ = dict(os.environ)
        if popen["env"] is not None:
            # pyre-ignore
            environ.update(popen["env"])
        environ["RANK"] = str(rank)
        environ["WORLD_SIZE"] = str(world_size)
        environ["LOCAL_RANK"] = str(rank % processes_per_host)
        environ["LOCAL_WORLD_SIZE"] = str(processes_per_host)
        environ["SUPERVISOR_PIPE"] = proc_addr
        environ["SUPERVISOR_IDENT"] = str(proc_id)
        popen = {**popen, "env": environ}
        try:
            logcontext = (
                nullcontext() if logfilename is None else open(logfilename, "a")
            )
            with logcontext as logfile:
                self.subprocess: subprocess.Popen[str] = subprocess.Popen(
                    # pyre-ignore
                    **popen,
                    start_new_session=True,
                    stdout=logfile,
                    stderr=logfile,
                )
        except Exception:
            s = io.StringIO()
            traceback.print_exc(file=s)
            logger.warning(f"Process failed to start: %s\n", s.getvalue())
            raise ProcessFailedToStart(s.getvalue())
        self.fd: int = pidfd_open(self.subprocess.pid)
        self.proc_id_bytes: bytes = proc_id.to_bytes(8, byteorder="little")
        self.deferred_sends: Optional[List[bytes]] = []

    def _send(self, _msg: object) -> None:
        msg = pickle.dumps(_msg)
        if self.deferred_sends is not None:
            self.deferred_sends.append(msg)
        else:
            self.proc_comm.send_multipart([self.proc_id_bytes, msg])

    def _notify_connected(self) -> None:
        deferred_sends = self.deferred_sends
        if deferred_sends is not None:
            for msg in deferred_sends:
                self.proc_comm.send_multipart([self.proc_id_bytes, msg])
            self.deferred_sends = None


class Host:
    def __init__(self, supervisor_port: str) -> None:
        self.context: zmq.Context = zmq.Context(1)
        self.backend: zmq.Socket = self.context.socket(zmq.DEALER)
        self.backend.setsockopt(zmq.IPV6, True)
        logger.info("Host Manager Connecting to %s", supervisor_port)
        self.backend.connect(supervisor_port)

        # tell the supervisor we exist, and provide
        # hostname for debugging.
        self.backend.send(pickle.dumps(("_hostname", None, socket.gethostname())))

        self.poller = zmq.Poller()
        self.poller.register(self.backend, zmq.POLLIN)

        self.proc_comm: zmq.Socket = self.context.socket(zmq.ROUTER)
        self.proc_addr = f"ipc:///tmp/proc_{os.getpid()}"
        self.proc_comm.bind(self.proc_addr)
        self.poller.register(self.proc_comm, zmq.POLLIN)

        self.process_table: Dict[bytes, Process] = {}
        self.fd_to_pid: Dict[int, bytes] = {}
        self._launches = 0
        self.has_shutdown = False

    def heartbeat(self) -> None:
        self.backend.send(b"")

    # TODO: validate these are valid messages to send

    def launch(
        self,
        proc_id: int,
        rank: int,
        processes_per_rank: int,
        world_size: int,
        popen: Mapping[str, object],
        name: str,
        simulate: bool,
        log_file: Optional[str],
    ) -> None:
        self._launches += 1
        if simulate:
            self.backend.send(pickle.dumps(("_started", proc_id, 2)))
            self.backend.send(pickle.dumps(("_exited", proc_id, 0)))
            return
        try:
            process = Process(
                name,
                log_file,
                self.proc_comm,
                proc_id,
                rank,
                processes_per_rank,
                world_size,
                popen,
                self.proc_addr,
            )
            self.process_table[process.proc_id_bytes] = process
            self.fd_to_pid[process.fd] = process.proc_id_bytes
            self.poller.register(process.fd, zmq.POLLIN)
            reply = process.subprocess.pid
        except ProcessFailedToStart as e:
            reply = str(e)
        self.backend.send(pickle.dumps(("_started", proc_id, reply)))

    def send(self, _proc_id: int, msg: bytes) -> None:
        proc_id = _proc_id.to_bytes(8, byteorder="little")
        if proc_id in self.process_table:
            process = self.process_table[proc_id]
            process._send(msg)

    def signal(self, _proc_id: int, sig: int, group: bool) -> None:
        proc_id = _proc_id.to_bytes(8, byteorder="little")
        if proc_id in self.process_table:
            process = self.process_table[proc_id]
            if group:
                os.killpg(process.subprocess.pid, sig)
            else:
                process.subprocess.send_signal(sig)

    def _fd_exit(self, fd: int) -> Tuple[Process, int]:
        pid_bytes = self.fd_to_pid.pop(fd)
        process = self.process_table.pop(pid_bytes)
        # we do not allow descendents to outlive the parent
        # if any remain this kill will clean them up
        os.killpg(process.subprocess.pid, signal.SIGKILL)
        returncode = process.subprocess.wait()
        self.poller.unregister(fd)
        os.close(fd)
        return process, returncode

    def shutdown(self) -> None:
        if self.has_shutdown:
            return
        self.has_shutdown = True
        for proc in self.process_table.values():
            os.killpg(proc.subprocess.pid, signal.SIGTERM)
        expiry = time.time() + ABORT_INTERVAL
        ttl = ABORT_INTERVAL
        while ttl > 0 and self.process_table:
            for s, _ in self.poller.poll(timeout=int(1000 * ttl)):
                if isinstance(s, int):
                    self._fd_exit(s)
            ttl = time.time() - expiry
        if self.process_table:
            for proc in self.process_table.values():
                os.killpg(proc.subprocess.pid, signal.SIGKILL)


    def abort(self, with_error: Optional[str] = None) -> None:
        self.shutdown()
        if with_error:
            raise ConnectionAbortedError(with_error)
        else:
            sys.exit(0)

    def run_event_loop_forever(self) -> None:
        heartbeat_at = time.time() + HEARTBEAT_INTERVAL
        expiry = None
        while True:
            for s, _ in self.poller.poll(timeout=int(1000 * HEARTBEAT_INTERVAL)):
                if isinstance(s, int):
                    process, returncode = self._fd_exit(s)
                    self.backend.send(
                        pickle.dumps(("_exited", process.proc_id, returncode))
                    )
                elif s is self.backend:
                    if expiry is None:
                        logging.info(f"Connected to supervisor")
                    expiry = time.time() + HEARTBEAT_INTERVAL * HEARTBEAT_LIVENESS
                    msg = self.backend.recv()
                    if msg:
                        cmd, *args = pickle.loads(msg)
                        getattr(self, cmd)(*args)
                elif s is self.proc_comm:
                    proc_id_bytes, msg = self.proc_comm.recv_multipart()
                    process = self.process_table.get(proc_id_bytes)
                    # it is possible for the process to have already exited before
                    # we get its messages, so process_table will be empty
                    if process is not None:
                        process._notify_connected()
                    if len(msg):
                        proc_id = int.from_bytes(proc_id_bytes, byteorder="little")
                        self.backend.send(pickle.dumps(("_response", proc_id, msg)))
            if expiry is not None:
                t = time.time()
                if t > heartbeat_at:
                    heartbeat_at = time.time() + HEARTBEAT_INTERVAL
                    self.heartbeat()
                if t > expiry:
                    self.abort(
                        f"No messages from supervisor for {HEARTBEAT_INTERVAL*HEARTBEAT_LIVENESS} seconds, aborting."
                    )


def main(addr: str) -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s", level=logging.INFO
    )
    manager: Host = Host(addr)

    def handler(signal: int, frame: object) -> None:
        manager.shutdown()
        sys.exit(1)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
    try:
        manager.run_event_loop_forever()
    finally:
        manager.shutdown()


if __name__ == "__main__":
    main(sys.argv[1])
