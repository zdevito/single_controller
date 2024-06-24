from supervisor import ProcessList
from typing import TYPE_CHECKING, Dict, NamedTuple, Optional, Tuple,  Callable, Any, Union, List
from torch.utils._python_dispatch import TorchDispatchMode

import torch
from contextlib import contextmanager, ExitStack
import math
from . import messages
from .reference import Referenceable
from . import stream
from .tensor import dtensor_dispatch
from .ndslice import NDSlice
import itertools

from .fake import fake_call

if TYPE_CHECKING:
    from .controller import Controller


class RemoteFunction:
    def __init__(self, real, propagate: Callable):
        if isinstance(real, torch._ops.OpOverload):
            path = "torch.ops." + str(real)
        else:
            path = real
        self.path: Union[str, Callable] = path
        self.propagate = propagate
    
    def __repr__(self):
        return f"RemoteFunction(path={self.path})"

    def fake_call(self, args, kwargs):
        return fake_call(self.propagate, args, kwargs)

    def __reduce_ex__(self, protocol):
        return messages.RemoteFunction, (self.path,)

    def __call__(self, *args: Any, **kwargs):
        return dtensor_dispatch(self, args, kwargs, _active, stream._active)

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return lambda *args, **kwargs: self(instance, *args, **kwargs)


def remote_function(path):
    return lambda propagate: RemoteFunction(path, propagate)

class DeviceMesh(Referenceable):
    def __init__(self, ctrl: 'Controller', processes: 'NDSlice', names: Tuple[str, ...]):
        assert isinstance(processes, NDSlice)
        self.ctrl = ctrl
        assert processes.ndim == len(names)
        self.names = names
        self.processes = processes
        self.ref = None

    def __repr__(self):
        return f'<DeviceMesh({self.names}, {self.processes.sizes}) at {hex(id(self))}>'

    def define_ref(self):
        # Optimize: we do not have to send device meshes to all workers if we can
        # Create process groups as subsets
        self.ref = self.ctrl.ref()
        msg = messages.CreateDeviceMesh(self, self.names, self.processes)
        self._send(msg, flush_deletes=False)

    def delete_ref(self, ref: int):
        if not self.ctrl._shutdown:
            self._send(messages.DeleteRefs([ref]))

    def _send(self, cmd: NamedTuple, flush_deletes=True):
        if flush_deletes:
            to_delete = self.ctrl._flush_deletes(self)
            if to_delete:
                self.ctrl.send(self.processes, messages.CommandGroup([to_delete, cmd]))
                return
        self.ctrl.send(self.processes, cmd)

    def stack(self, **kwargs):
        raise NotImplementedError()

    def __call__(self, **kwargs) -> 'DeviceMesh':
        """
        m.index(batch=3) or m.index(batch=slice(3, None))
        """
        offset = self.processes.offset
        names = []
        sizes = []
        strides = []
        for name, size, stride in zip(self.names, self.processes.sizes, self.processes.strides):
            if name in kwargs:
                e = kwargs.pop(name)
                if isinstance(e, slice):
                    start, stop, slice_stride = e.indices(size)
                    offset += start * stride
                    names.append(name)
                    sizes.append((stop - start) // slice_stride)
                    strides.append(slice_stride * stride)
                else:
                    if e >= size or e < 0:
                        raise IndexError('index out of range')
                    offset += e * stride
            else:
                names.append(name)
                sizes.append(size)
                strides.append(stride)

        if kwargs:
            raise TypeError(f'{self} does not have dimension(s) named {tuple(kwargs.keys())}')

        processes = NDSlice(offset, sizes, strides)
        return DeviceMesh(self.ctrl, processes, tuple(names))

    def split(self, **kwargs: Dict[str, Tuple[str, ...]]):
        raise NotImplementedError()

    def rotate(self, **kwargs: Dict[str, int]):
        raise NotImplementedError()

    @remote_function('controller.worker._rank')
    def rank(self, dim: str):
        if dim not in self.names:
            raise KeyError(f'{self} does not have dimension {repr(dim)}')
        return torch.full((), 0, dtype=torch.long)

    def _process(self, coordinates: Optional[Dict[str, int]]) -> NDSlice:
        if coordinates is None:
            return NDSlice(self.processes.offset, [1], [1])
        if len(coordinates) > len(self.names):
            extra = set(coordinates.keys()) - set(self.names)
            raise KeyError(f'{list(extra)}')
        flat = [coordinates[name] for name in self.names]
        return NDSlice(self.processes.nditem(flat), [1], [1])

    def _use(self, tensor):
        borrows = tensor._borrows
        if tensor.stream is not borrows.origin_stream:
            borrow = borrows.active[tensor.stream]
            if not borrow.used:
                self._send(messages.BorrowFirstUse(borrow.id))
                borrows.active[tensor.stream] = borrow._replace(used=True)

    def activate(self):
        self._activated = active_mesh(self)
        self._activated.__enter__()
        assert _active is not None
        assert _dispatch_enabled
    
    def deactivate(self):
        self._activated.__exit__(None, None, None)

_active: Optional[DeviceMesh] = None
_dispatch_enabled = False


class _ActiveMesh(TorchDispatchMode):
    ignore = ['profiler._record_function_exit._RecordFunction']

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if _active is None:
            return func(*args, **kwargs)
        return dtensor_dispatch(RemoteFunction(func, func), args, kwargs, _active, stream._active)

@contextmanager
def _dispatch():
    global _dispatch_enabled
    if _dispatch_enabled:
        yield
    else:
        _dispatch_enabled = True
        try:
            with _ActiveMesh():
                yield
        finally:
            _dispatch_enabled = False

_on_change: List[Callable] = []

@contextmanager
def active_mesh(mesh: Optional[DeviceMesh]):
    global _active
    for on_change in _on_change:
        on_change(_active, mesh)
    _active, old = mesh, _active
    try:
        with _dispatch():
            yield
    finally:
        for on_change in _on_change:
            on_change(_active, old)
        _active = old
