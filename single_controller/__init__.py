import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten, tree_unflatten, tree_map
from typing import NamedTuple, List
from contextlib import contextmanager
from .base_tensor import BaseTensor
from socket import socket, AF_INET, SOCK_STREAM
from uuid import uuid4
from subprocess import Popen
import sys
from pickle import Unpickler, Pickler
import asyncio

from torch._subclasses.fake_tensor import FakeTensorMode
fake_mode = FakeTensorMode()


_next_handle = 0
class DTensorRef:
    def __init__(self):
        global _next_handle
        _next_handle += 1
        self.id = _next_handle


def dtensor_dispatch(func, args=(), kwargs=None, worker=None):
    def unwrap_fake(t):
        nonlocal worker
        if isinstance(t, DTensor):
            if worker is None:
                worker = t._worker
            elif worker is not t._worker:
                raise NotImplementedError("mixed workers")
            return t._fake
        elif isinstance(t, torch.Tensor):
            raise NotImplementedError("mixed DTensor/local tensor")
        else:
            return t
    def unwrap_ref(t):
        if isinstance(t, DTensor):
            return t._ref
        else:
            return t

    fake_args = tree_map(unwrap_fake, args)
    fake_kwargs = tree_map(unwrap_fake, kwargs)

    ref_args = tree_map(unwrap_ref, args)
    ref_kwargs = tree_map(unwrap_ref, kwargs)

    with fake_mode:
        result = func(*fake_args, **fake_kwargs)
    flat, spec = tree_flatten(result)
    flat_fake_tensors = [e for e in flat if isinstance(e, torch.Tensor)]
    flat_tensor_refs = [DTensorRef() for _ in flat_fake_tensors]

    flat_results = iter(DTensor(fake, ref, worker) for fake, ref in zip(flat_fake_tensors, flat_tensor_refs))
    worker.send_command(func, ref_args, ref_kwargs, flat_tensor_refs)
    to_unflatten = [next(flat_results) if isinstance(e, torch.Tensor) else e for e in flat]
    results = tree_unflatten(to_unflatten, spec)
    return results

class ActiveSharding(TorchDispatchMode):
    def __init__(self, worker):
        self.worker = worker
    def __torch_dispatch__(self, func, types, args, kwargs):
        for x in tree_flatten((args, kwargs))[0]:
            if isinstance(x, torch.Tensor):
                return func(*args, **kwargs)
        return dtensor_dispatch(func, args, kwargs, worker=self.worker)

class PicklableFunc:
    def __init__(self, callable):
        if isinstance(callable, str):
            first, *parts = callable.split('.')
            callable = globals()[first]
            for p in parts:
                callable = getattr(callable, p)
        self.callable = callable
    def __call__(self, *args, **kwargs):
        return self.callable(*args, **kwargs)
    def __reduce__(self):
        if isinstance(self.callable, torch._ops.OpOverload):
            return PicklableFunc, ("torch.ops." + str(self.callable),)
        else:
            return PicklableFunc, (self.callable,)

class DTensor(BaseTensor):
    @staticmethod
    def __new__(
        cls,
        fake: torch.Tensor,
        ref: DTensorRef,
        worker: 'Worker',
    ):
        r = torch.Tensor._make_wrapper_subclass(
            cls,
            fake.size(),
            strides=fake.stride(),
            storage_offset=fake.storage_offset(),
            device=fake.device,  # This is the device of of either input tensor or first tensor of a list
            dtype=fake.dtype,
            layout=fake.layout,
            requires_grad=fake.requires_grad,
        )
        r._ref = ref
        r._fake = fake
        r._worker = worker
        return r

    def __init__(self, fake, ref, worker):
        pass

    @classmethod
    def to_remote(cls, t: torch.Tensor, worker: 'Worker'):
        f = fake_mode.from_tensor(t)
        r = DTensorRef()
        worker.send_value(r, t)
        return DTensor(f, r, worker)

    def __repr__(self):
        if self.grad_fn:
            return f"DTensor(_handle={self._ref.id}, grad_fn={self.grad_fn})"
        return f"DTensor(_ref={self._ref.id})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        return dtensor_dispatch(func, args, kwargs)

    def to_local(self):
        return self._worker.request_value(self._ref)

    def __del__(self):
        self._worker.del_value(self._ref)

class Manager:
    def __init__(self):
        self.host = "127.0.0.1"
        self.port = 12345
        self.socket = socket(AF_INET, SOCK_STREAM)
        self.socket.bind((self.host, self.port))
        self.socket.listen(1)

    def create_worker(self):
        secret = str(uuid4())
        # if we were only ever going to use local processes then it would easier to just directly
        # create pipes, but this uses sockets so that we can add remote processes without changing the
        # setup.
        proc = Popen([sys.executable, '-m', 'single_controller.worker_process', self.host, str(self.port), secret])
        client_socket, client_address = self.socket.accept()
        return Worker(proc, client_socket, secret)

class Future(asyncio.Future):
    def wait(self):
        worker = self.get_loop() # actually our worker
        while not self.done():
            worker.wait_one()
        return self.result()

class Worker:
    def __init__(self, proc, sock, secret):
        self.proc = proc
        self.ofile = sock.makefile("wb")
        self.ifile = sock.makefile("rb")
        self.output = Pickler(self.ofile)
        self.input = Unpickler(self.ifile)
        assert secret == self.input.load()
        self.pending_futures = []

    def send_command(self, func, args, kwargs, results):
        self.output.dump(('send_command', PicklableFunc(func), args, kwargs, results))

    def request_value(self, ref: DTensorRef):
        self.output.dump(('request_value', ref))
        self.pending_futures
        f = Future(loop=self)
        self.pending_futures.append(f)
        return f

    def wait_one(self):
        assert self.pending_futures
        self.ofile.flush()
        # TODO error pathways are not implemented
        # here or on worker if something goes wrong with a command
        self.pending_futures.pop().set_result(self.input.load())

    def wait_all(self):
        if self.pending_futures:
            self.ofile.flush()
            for f in self.pending_futures:
                f.set_result(self.input.load())
            self.pending_futures.clear()

    def send_value(self, ref: DTensorRef, value: torch.Tensor):
        self.output.dump(('send_value', ref, value))

    def del_value(self, ref: DTensorRef):
        self.output.dump(('del_value', ref))

    def __del__(self):
        self.output.dump(('exit',))
        self.ofile.flush()
        self.proc.wait()

    # HACKS: event loop functions to make Future object happy
    def call_soon(self, callback, *args, context):
        context.run(callback, *args)

    def get_debug(self):
        return False
