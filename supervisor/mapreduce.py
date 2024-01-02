# only in itertools after 3.12
from supervisor import as_completed, Context, Host, Process, Future, get_message_queue
from typing import Iterator, List, TypeVar, Sequence, Dict
import sys
import zmq
from itertools import repeat
import pickle
import io

T = TypeVar("T")
def batched(xs: Sequence[T], n: int) -> Iterator[List[T]]:
    b = []
    for x in xs:
        b.append(x)
        if len(b) == n:
            yield b
            b = []
    if b:
        yield b

def mapreduce(ctx: Context, hosts: Sequence[Host], map=None, reduce=None, inputs=None, finish=None, branch=4):
    """
    Run finish(reduce([map(input) for input in inputs)]))
    Reduce must be associative and commutative.
    Finish will be run once on the final value.

    map, reduce, finish, inputs, and the outputs of each must be picklable.

    Not really map-reduce in the Hadoop sense with sort/shuffle/keys, but easy to implement for
    simple debug info gathering.
    """
    processes = ctx.create_process_group(hosts, [sys.executable, '-m', 'supervisor.mapreduce'])
    if inputs is None:
        binputs = repeat(None)
    else:
        binputs = batched(inputs, (len(inputs) - 1) // len(hosts) + 1)
    mesg_to_proc: Dict[Future[object], Process] = {}
    for proc, inputs in zip(processes, binputs):
        proc.send((map, reduce, inputs))
        mesg_to_proc[proc.recv()] = proc

    ready = as_completed(mesg_to_proc.keys())
    for resp in batched(ready, branch):
        for r in resp:
            r.result()
        h, *tail = resp
        proc = mesg_to_proc[h]
        to = proc.host.hostname().result()
        dst = f'tcp://{to}:{h.result()}'
        for t in tail:
            mesg_to_proc[t].send(('send', dst))
        proc.send(('reduce', len(tail)))
        f = proc.recv()
        mesg_to_proc[f] = proc
        ready.add(f)
    last = next(iter(ready))
    last.result()
    proc = mesg_to_proc[last]
    proc.send(('finish', finish))

    # this seems more expensive than just concating the bytes and
    # unpickling. However, unpickling bytes never releases the GIL,
    # so it will hang Python, and the supervisor event loop will
    # fail to heartbeat. This is slower, but everytime readinto
    # is called, the event_loop can run.
    class Custom(io.RawIOBase):
        def __init__(self):
            self.rest = None
        def readinto(self, b):
            if self.rest is None:
                self.rest = memoryview(proc.recv().result())
            lr, lb = len(self.rest), len(b)
            if lr <= lb:
                b[:lr] = self.rest
                self.rest = None
                return lr
            else:
                b[:] = self.rest[:lb]
                self.rest = self.rest[lb:]
                return lb
        def readable(self):
            return True

    return pickle.load(io.BufferedReader(Custom()))

def _socket(f):
    s = f.context.socket(zmq.DEALER)
    s.setsockopt(zmq.IPV6, True)
    return s

def main():
    CHUNK_SIZE = 1024*1024*8 # 8MB chunks
    sock = get_message_queue()
    map, reduce, inputs = sock.recv_pyobj()
    if reduce is None:
        reduce = sum
    if map is None:
        map = lambda x: x
    name = f"tcp://*:*"
    router = _socket(sock)
    router.bind(name)
    port = router.getsockopt(zmq.LAST_ENDPOINT).decode().split(":")[-1]
    r = map(inputs)
    # print(f"{port}: r = reduce(map({inputs})")
    sock.send_pyobj(port)
    while True:
        action, value = sock.recv_pyobj()
        if action != 'reduce':
            break
        inputs = [r, *(router.recv_pyobj() for i in range(value))]
        # print(f"{port}: r = reduce({inputs})")
        r = reduce(inputs)
        sock.send_pyobj(port)

    if action == 'send':
        # print(f"{port}: send {r} -> {value}")
        s = _socket(sock)
        s.connect(value)
        s.send_pyobj(r)
    else:
        assert action == 'finish'
        # print(f"{port}: finish {r}")
        if value is not None:
            r = value(r)
        data = pickle.dumps(r)
        nchunks = (len(data) - 1) // CHUNK_SIZE + 1
        print("SIZE", len(data) / (1024*1024))
        for i in range(nchunks):
            chunk = data[i * CHUNK_SIZE : (i+1) * CHUNK_SIZE]
            sock.send_pyobj(chunk)

if __name__ == "__main__":
    main()
