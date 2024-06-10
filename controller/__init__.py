from supervisor import Context, Host
from typing import Dict, Sequence, TypedDict, Union, Optional, List

from typing_extensions import Unpack
import torch
from . import messages
from .history import RemoteException
from .stream import active_stream, Stream
from .future import Future
from .tensor import Tensor, dtensor_check
from .device_mesh import DeviceMesh, remote_function, active_mesh
from .ndslice import NDSlice
from . import stream, device_mesh
from .controller import Controller as _Controller, ProcessBackend as _ProcessBackend

def fetch_shard(obj, coordinates: Optional[Dict[str, int]] = None, preprocess: Optional[str] = None):
    """
    Retrieve the shard at `coordinates` of the current device mesh of each tensor in obj.
        obj - a pytree containing the tensors the fetch
        coordinates - a dictionary from mesh dimension name to coordinate of the shard
                      If None, this will fetch from coordinate 0 for all dimensions (useful after all_reduce/all_gather)
        preprocess - an optional specifier for a remote function that is applyied to obj before returning the value.
                     This can be used to turn the tensor into some non-tensor values before copying it back
    """
    if device_mesh._active is None:
        raise RuntimeError("No device mesh active")
    mesh = device_mesh._active
    ctrl = mesh.ctrl
    tensors, _ = dtensor_check('fetch_shard', (obj,), {}, mesh, stream._active)
    fut = Future(ctrl)
    ident = ctrl.history.ident((), tensors, fut)
    process = mesh._process(coordinates)
    ctrl.send(process, messages.FetchValue(ident, preprocess, obj, stream._active))
    return fut


class TensorOptions(TypedDict, total=False):
    dtype: 'torch.dtype'
    layer: 'torch.layout'
    device: Union[str, 'torch.device']
    requires_grad: bool
    pin_memory: bool
    memory_format: 'torch.memory_format'


class Pipe:
    def push(self, tensor: 'Tensor'):
        raise NotImplementedError()

    def pop(self, sizes: Sequence[int], **kwargs: Unpack[TensorOptions]) -> 'Tensor':
        raise NotImplementedError()

def world_mesh(ctx: Context, hosts: List[Host], gpu_per_host: int, _processes=None):
    backend = _ProcessBackend(ctx, hosts, gpu_per_host, _processes=_processes)
    ctrl = _Controller(backend)
    return DeviceMesh(ctrl, NDSlice(0, [len(hosts), gpu_per_host], [gpu_per_host, 1]), ('host', 'gpu'))

def get_active_stream():
    return stream._active


__all__ = [
    'RemoteException',
    'remote_function',
    'Future',
    'active_stream',
    'active_mesh',
    'DeviceMesh',
    'Tensor',
    'Stream',
    'get_active_stream',
]
