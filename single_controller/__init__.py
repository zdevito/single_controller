import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten, tree_unflatten, tree_map
from typing import NamedTuple, List, Union, Literal
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


def dtensor_dispatch(func, args=(), kwargs=None, sharding=None):
    worker = sharding.mesh.workers[0] if sharding else None
    def stringify(t):
        if isinstance(t, DTensor):
            return 'DTensor'
        elif isinstance(t, torch.Tensor):
            return 'Tensor'
        else:
            return t

    def unwrap_fake(t):
        nonlocal worker, sharding
        if isinstance(t, DTensor):
            if worker is None:
                worker = t._worker
                sharding = t._sharding
            elif worker is not t._worker:
                raise NotImplementedError("mixed workers")
            return t._fake
        elif isinstance(t, torch.Tensor):
            raise NotImplementedError(f"mixed DTensor/local tensor {func}(args={tree_map(stringify, args)} kwargs={tree_map(stringify, kwargs)})")
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

    flat_results = iter(DTensor(fake, ref, sharding) for fake, ref in zip(flat_fake_tensors, flat_tensor_refs))
    worker.send_command(func, ref_args, ref_kwargs, flat_tensor_refs)
    to_unflatten = [next(flat_results) if isinstance(e, torch.Tensor) else e for e in flat]
    results = tree_unflatten(to_unflatten, spec)
    return results

class ActiveSharding(TorchDispatchMode):
    ignore = ['profiler._record_function_exit._RecordFunction']
    def __init__(self, sharding):
        self.sharding = Sharding.lift(sharding)
    def __torch_dispatch__(self, func, types, args, kwargs):
        if str(func) in self.ignore:
            return func(*args, **kwargs)
        for x in tree_flatten((args, kwargs))[0]:
            if isinstance(x, torch.Tensor):
                return func(*args, **kwargs)
        return dtensor_dispatch(func, args, kwargs, sharding=self.sharding)

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
        sharding: 'Sharding',
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
        assert isinstance(sharding, Sharding)
        r._sharding = sharding
        return r

    def __init__(self, fake, ref, worker):
        pass

    @classmethod
    def to_remote(cls, t: torch.Tensor, sharding: 'Sharding'):
        sharding = Sharding.lift(sharding)
        f = fake_mode.from_tensor(t)
        r = DTensorRef()
        result = DTensor(f, r, sharding)
        result._worker.send_value(r, t)
        return result

    def __repr__(self):
        if self.grad_fn:
            return f"DTensor(_handle={self._ref.id}, grad_fn={self.grad_fn})"
        return f"DTensor(_ref={self._ref.id})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        return dtensor_dispatch(func, args, kwargs)

    @property
    def _worker(self):
        # all the sharding is fake for now...
        return self._sharding.mesh.workers[0]

    def to_local(self):
        return self._worker.request_value(self._ref)

    def __del__(self):
        self._worker.del_value(self._ref)

class Manager:
    def __init__(self):
        self.host = "127.0.0.1"
        self.port = 12345
        self.socket = socket(AF_INET, SOCK_STREAM)
        success = False
        for i in range(10):
            try:
                self.socket.bind((self.host, self.port))
                break
            except OSError:
                self.port += 1
                print("Trying next port...")

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

class LocalWorker:
    """
    Run in the local process rather than remotely
    """
    def __init__(self):
        self.ref_to_tensor = {}
    def send_command(self, func, args, kwargs, results):
        def get_tensor(t):
            if isinstance(t, DTensorRef):
                return self.ref_to_tensor[t.id]
            else:
                return t
        args = tree_map(get_tensor, args)
        kwargs = tree_map(get_tensor, kwargs)
        result = func(*args, **kwargs)
        flat_results, _ = tree_flatten(result)
        real_results = [e for e in flat_results if isinstance(e, torch.Tensor)]
        for real, ref in zip(real_results, results):
            self.ref_to_tensor[ref.id] = real

    def request_value(self, ref: DTensorRef):
        f = Future(loop=self)
        f.set_result(self.ref_to_tensor[ref.id])
        return f

    def wait_one(self):
        pass

    def wait_all(self):
        pass

    def send_value(self, ref: DTensorRef, value: torch.Tensor):
        self.ref_to_tensor[ref.id] = value

    def del_value(self, ref: DTensorRef):
        del self.ref_to_tensor[ref.id]

    # HACKS: event loop functions to make Future object happy
    def call_soon(self, callback, *args, context):
        context.run(callback, *args)

    def get_debug(self):
        return False

class WorkerMesh:
    """
    A multi-dimensional array of devices used to specify shardings.
    Similar to the collective DTensor we have today
    """
    def __init__(self, workers: List[Worker], shape=None):
        self.workers = workers
        self.shape = torch.arange(len(self.workers)) if shape is None else shape

    def reshape(self, dims: List[int]):
        return WorkerMesh(self.workers, self.shape.reshape(dims))

    def __getitem__(self, elem):
        return WorkerMesh(self.workers, self.shape[elem])


class Sharding(NamedTuple):
    """
    A description of how a single tensor is sharded across devices.
    This is equivalent to our collective dtensor
    """
    mesh: WorkerMesh
    # one entry per dimension of device mesh,
    # which specifies how the tensor is represented
    # in that dimension
    sharding: List[Union[int, # shareded - e.g. the 0th dimension of
                              #   of the tensor is split across this dimesion
                              #   of the device mesh
                 Literal["r", # replicated - a copy of the data is stored
                              #   across this dimension of the device mesh
                         "+"  # partial sum - the value of the tensor is
                              #   the sum of the tensors stored along
                              #   this dimension of the device mesh
                         ]]]
    # note: it is also equivlent to specify a sharding
    # by saying if each dimension of the tensor is sharded and which
    # device mesh dimension it correspond to, but it still
    # requires saying whether the remaining device mesh dimensions are
    # replicated or sharded.
    @staticmethod
    def lift(obj):
        if isinstance(obj, (LocalWorker,Worker)):
            mesh = WorkerMesh([obj]).reshape(())
            return Sharding(mesh=mesh, sharding = [])
        elif isinstance(obj, Sharding):
            return obj
        else:
            raise ValueError("expected Sharding")


    def change(self, mesh=None, sharding=None):
        return Sharding(mesh=mesh if mesh else self.mesh, sharding=sharding if sharding else self.sharding)
