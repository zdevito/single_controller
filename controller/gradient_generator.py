from ._gradient_generator import GradientGenerator as _GradientGenerator
import torch
import torch.autograd.graph
from torch._C._autograd import _get_sequence_nr
from typing import Union, Optional, Sequence, TYPE_CHECKING, NamedTuple, Dict
TensorOrEdge = Union[torch.Tensor, torch.autograd.graph.GradientEdge]

from . import device_mesh
from . import stream
from .tree import flatten

from .device_mesh import active_mesh
from .stream import active_stream
from .tensor import Tensor
from functools import partial
from types import FunctionType, CellType

class Context(NamedTuple):
    device_mesh: 'Optional[device_mesh.DeviceMesh]'
    stream: 'stream.Stream'

    def enable(self):
        with active_mesh(self.device_mesh), active_stream(self.stream), torch.no_grad():
            yield


_sequence_nr_to_context: Dict[int, Context] = {}
_sequence_nr_end = 0

def restore_context(t: Optional[Tensor], sn: Optional[int], last: bool):
    if sn is not None:
        _update_context_map(Context(device_mesh._active, stream._active))
        ctx = _sequence_nr_to_context.pop(sn) if last else _sequence_nr_to_context[sn]
        return ctx.enable()
    if t is not None:
        return Context(t.mesh, t.stream).enable()
    return Context(device_mesh._active, stream._active).enable()

def _update_context_map(ctx: Context):
    global _sequence_nr_end
    next_sequence_nr = _get_sequence_nr()
    for i in range(_sequence_nr_end, next_sequence_nr):
        _sequence_nr_to_context[i] = ctx
    _sequence_nr_end = _get_sequence_nr()

device_mesh._on_change.append(lambda old, mesh: _update_context_map(Context(old, stream._active)))
stream._on_change.append(lambda old, stream: _update_context_map(Context(device_mesh._active, old)))


def grad_generator(roots: Union[torch.Tensor, Sequence[TensorOrEdge]] = [], with_respect_to: Sequence[TensorOrEdge] = [], grad_roots: Sequence[Optional[torch.Tensor]] = []):
    if isinstance(roots, torch.Tensor):
        roots = [roots]
    return _GradientGenerator(list(roots), list(with_respect_to), list(grad_roots), restore_context)        

class GradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fn, *args, **kwargs):
        result, backward_continuation = fn(*args, **kwargs)
        ctx.backward_continuation = backward_continuation
        values = []
        for cell in backward_continuation.__closure__:
            values.append(cell.cell_contents)
            cell.cell_contents = None
        tensors, ctx.unflatten = flatten(values, lambda x: isinstance(x, torch.Tensor))
        ctx.save_for_backward(*tensors)
        return result

    @staticmethod
    def backward(ctx, *args, **kwargs):
        closure = tuple(CellType(v) for v in ctx.unflatten(ctx.saved_tensors)) 
        orig = ctx.backward_continuation
        fn = FunctionType(orig.__code__, orig.__globals__, orig.__name__, orig.__defaults__, closure)
        return None, *fn(*args, **kwargs)

def grad_function(fn):
    return partial(GradFunction.apply, fn)
