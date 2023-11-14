import zmq
import sys
import time
import os
import pickle
import subprocess
from typing import NamedTuple, Any
import ctypes
__NR_pidfd_open = 434
libc = ctypes.CDLL(None)
syscall = libc.syscall

def pidfd_open(pid):
    return syscall(__NR_pidfd_open, pid, 0)

class Proc(NamedTuple):
    ident: int
    proc: subprocess.Popen
    send: Any
    response: Any


fd_to_pid = {}
process_table = {}

def popen(ident, *args):
    proc = subprocess.Popen([sys.executable, '-m', 'proc_rpc', proc_addr])
    fd = pidfd_open(proc.pid)
    pid = proc.pid.to_bytes(4, byteorder='little')
    process_table[pid] = Proc(ident, proc, args, [])
    fd_to_pid[fd] = pid
    poller.register(fd, zmq.POLLIN)
    return ('started', ident)

HEARTBEAT_INTERVAL = 1

def hello(arg, t):
    print(arg)
    time.sleep(t)
    return f"HI I WAS HELLO {arg}"

if __name__ == '__main__':
    context = zmq.Context(1)
    supervisor = sys.argv[1]
    backend = context.socket(zmq.DEALER)
    backend.connect(supervisor)

    heartbeat_at = time.time() + HEARTBEAT_INTERVAL
    backend.send(str(os.getpid()).encode())

    poller = zmq.Poller()
    poller.register(backend, zmq.POLLIN)

    proc_comm = context.socket(zmq.ROUTER)
    proc_addr = f'ipc:///tmp/proc_{os.getpid()}'
    proc_comm.bind(proc_addr)

    poller.register(proc_comm, zmq.POLLIN)

    while True:
        for s, _ in poller.poll(timeout=HEARTBEAT_INTERVAL):
            if isinstance(s, int):
                fd = s
                pid = fd_to_pid.pop(fd)
                p = process_table.pop(pid)
                e = p.proc.wait()
                poller.unregister(fd)
                os.close(fd)
                backend.send(pickle.dumps(('finished', p.ident, e, p.response.pop() if p.response else None)))
            elif s is backend:
                fn, *args = pickle.loads(backend.recv())
                fn = globals()[fn]
                print(fn, args)
                result = fn(*args)
                backend.send(pickle.dumps(result))
            elif s is proc_comm:
                pid, msg = proc_comm.recv_multipart()
                p = process_table[pid]
                if len(msg) == 0:
                    proc_comm.send_multipart([pid, pickle.dumps(p.send)])
                else:
                    p.response.append(pickle.loads(msg))

        if time.time() > heartbeat_at:
            heartbeat_at = time.time() + HEARTBEAT_INTERVAL
            backend.send(b"")
