from typing import TYPE_CHECKING, Dict, Any, Union, Literal, Optional, List, Tuple

from controller.ndslice import NDSlice
from .tree import flatten, tree_map
from .base_tensor import BaseTensor

import torch
import torch._ops
from . import messages
from .reference import Referenceable
from .history import Invocation
from .borrows import Borrows
from .stream import Stream
from . import stream
from supervisor import ProcessList
from .fake import fake_call

import traceback
import warnings

if TYPE_CHECKING:
    from .device_mesh import DeviceMesh, RemoteFunction
    from .future import Future

_valid_reduce = Literal['stack', 'sum', 'avg', 'product', 'min', 'max', 'band', 'bor', 'bxor']

class Tensor(Referenceable, BaseTensor):
    stream: Stream
    mesh: 'DeviceMesh'
    ref: Optional[int]
    _borrowed: bool
    _invocation: Optional[Invocation]
    _fake: torch.Tensor

    def __new__(
        cls,
        fake: torch.Tensor, mesh: 'DeviceMesh', stream: 'Stream', borrowed: bool
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
        r._fake = fake
        ctrl = mesh.ctrl
        r.ref = ctrl.ref()
        r.mesh = mesh
        r.stream = stream
        r._borrowed = borrowed
        r._borrows.aliases.add(r)
        r._invocation = None
        return r

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        return dtensor_dispatch(RemoteFunction(func, func), args, kwargs, None, stream._active)

    def __init__(self, fake: torch.Tensor, mesh: 'DeviceMesh', stream: Stream, borrowed: bool):
        pass

    def __repr__(self, *, tensor_contents=None):
        return f"DTensor(mesh={self.mesh}, stream={self.stream}, fake={self._factory()})"

    def drop(self):
        if self.ref is None:
            return
        borrows = self._borrows
        for alias in borrows.aliases:
            if alias.stream is not self.stream:
                continue
            alias._drop_ref()
        if borrows.origin_stream is not self.stream:
            borrow = borrows.active.pop(self.stream)
            self.mesh._send(messages.BorrowDrop(borrow.id))
            if not borrows.active:
                borrows.writing_stream = borrows.origin_stream

        # we should be in the tensors list as well
        assert self.ref is None

    @property
    def _borrows(self) -> Borrows:
        storage = self._fake.untyped_storage()
        ctrl = self.mesh.ctrl
        if storage not in ctrl.allocation_borrows:
            ctrl.allocation_borrows[storage] = Borrows(self.stream)
        return ctrl.allocation_borrows[storage]

    @property
    def dropped(self):
        return self.ref is None

    def _drop_ref(self):
        assert self.ref is not None
        # silence borrowing warning
        self._borrowed = False
        self.delete_ref(self.ref)
        self.ref = None

    def to_mesh(self, mesh: 'DeviceMesh'):
        """
        Move data between one device mesh and another. Sizes of named dimensions must match.
        If mesh has dimensions that self.mesh does not, it will broadcast to those dimensions.


        broadcast:
            t.slice_mesh(batch=0).to_mesh(t.mesh)

        """
        return MeshSliceTensor(self, self.mesh).to_mesh(mesh)

    def reduce_(self, dim: str, reduction: _valid_reduce = "sum", scatter=False, mesh=None):
        # TODO: checks that this can actually happen in place e.g. if scatter is True, operation must be gather.
        inplace_valid = (reduction == 'gather' and scatter) or not scatter
        if not inplace_valid:
            raise ValueError(f'reduction {reduction} is not valid for in-place operation because the output size will not match the input size')
        return self.reduce(dim, reduction, scatter, mesh, _inplace=True)

    def reduce(self, dim: str, reduction: _valid_reduce = "sum", scatter=False, mesh=None, _inplace=False):
        """
        Perform a reduction operation along dim, and move the data to mesh. If mesh=None, then mesh=self.mesh
        'gather' will concat the values along dim, and produce a local result tensor with an addition outer dimension of len(dim).
        If scatter=True, the local result tensor will be evenly split across dim.

        allreduce:
            t.reduce('batch', 'sum')

            First reduces dim 'batch' creating a local tensor with the 'batch' dimension, then because output_mesh=input_mesh, and it still has dim 'batch',
            we broadcast the result reduced tensor to all members of batch.

        reducescatter:
            t.reduce('batch', 'sum', scatter=True)

            Same as above except that scatter=True introduces a new 'batch' dimension that is the result of splitting the local tensor across 'batch'

        allgather:
            t.reduce('batch', 'gather')

            First reduces dim 'batch' creating a bigger local tensor, then because output_mesh=input_mesh, and it still has dim 'batch',
            broadcasts the result concatenated tensor to all members of batch.

        alltoall:
            t.reduce('batch', 'gather', scatter=True)


            First reduces dim 'batch' creating a bigger local tensor, then introduces a new 'batch' dimension that is the result of splitting this
            (bigger) tensor across 'batch'. The result is the same dimension as the original tensor, but with each rank sending to all other ranks.


        gather (to dim 0):
            t.reduce('batch', 'gather', mesh=t.mesh.index(batch=0))

            First gathers dim 'batch' and then places it on the first rank. t.mesh.batch[0] doesn't have a 'batch' dimension, but this is
            ok because we eliminated the 'batch' dim via reduction.

        reduce:
            t.reduce('batch', 'sum', mesh=t.mesh.index(batch=0))

            First reduces dim 'batch' and then places it on the first rank. t.mesh.batch[0] doesn't have a 'batch' dimension, but this is
            ok because we eliminated the 'batch' dim via reduction.
        """
        if mesh is not None:
            raise NotImplementedError()
        if dim not in self.mesh.names:
            raise KeyError(f'dim {dim} not found in {self.mesh}')
        if reduction not in _valid_reduce.__args__:
            raise ValueError(f'reduction {reduction} not supported, reductions are {_valid_reduce.__args__}')
        if mesh is None:
            mesh = self.mesh

        if _inplace:
            fake_output = self._fake
        else:
            fake_output = fake_call(_fake_reduce, (self._fake, self.mesh, dim, reduction, scatter), {})
        r = Tensor(fake_output, self.mesh, self.stream, borrowed=False)
        assert r.ref is not None
        self.mesh._send(messages.Reduce(r, self, self._factory(), self.mesh, self.stream, dim, reduction, scatter, _inplace))
        self.mesh.ctrl.history.invocation((r,), (self,))
        return r

    def slice_mesh(self, **kwargs: Dict[str, Union[int, slice]]) -> 'MeshSliceTensor':
        # technically a slice of a device mesh and a device mesh are not same thing
        # because a device mesh also has caches for doing collectives.
        # but this is an easy way to create a MeshSliceTensor until we optimize
        # how we represent mesh slices.
        slicing = self.mesh(**kwargs)
        return MeshSliceTensor(self, slicing)

    def delete_ref(self, ref: int):
        if self._borrowed:
            current = ''.join(traceback.format_stack())
            borrowtb = ''.join(traceback.format_list(self._borrows.active[self.stream].frames))
            warnings.warn('t.drop() must be called before a borrowed tensor is freed to specify when the borrowed tensor should return to its origin stream, but this tensor is being deleted before drop.'
                          't.drop() is being called automatically here to ensure correctness, but this will force a synchronization back to the original stream at this point which might not be intended.'
                          f'\nTraceback of __del__(most recent call last):\n{current}\nTraceback of original borrow (most recent call last):{borrowtb}',
                          stacklevel=2)
            self.drop()
            return
        mesh = self.mesh
        mesh.ctrl.pending_del[mesh].append(ref)
    
    def _factory(self):
        return messages.TensorFactory.from_tensor(self._fake)


class MeshSliceTensor:
    def __init__(self, tensor: 'Tensor', slicing: 'DeviceMesh'):
        self.tensor = tensor
        self.slicing = slicing

    def to_mesh(self, mesh: 'DeviceMesh') -> 'Tensor':
        sizes = []
        strides = []
        for name, size in zip(mesh.names, mesh.processes.sizes):
            if name not in self.slicing.names:
                # TODO: allow broadcasting, and remove this check
                raise ValueError(f'dimension {name} is missing from destination device_mesh')
            index = self.slicing.names.index(name)
            if self.slicing.processes.sizes[index] != size:
                raise ValueError(f'dimension {name} of destination device_mesh has a different length than the source tensor')
            sizes.append(size)
            strides.append(self.slicing.processes.strides[index])

        if len(sizes) != len(self.slicing.names):
            missing = set(self.slicing.names) - set(mesh.names)
            raise ValueError(f'destination mesh does not have dimensions {missing}')

        # Optimized algorithm where:
        # 1. We can represent submeshes as NDSlice(offet, sizes, strides) on rank.
        # 2. A message can be efficiently broadcast to List[NDSlice] ranks by a smart tree based algorithm that can
        #    figure out which subtrees need the message.
        # 3. The message itself will uses List[NDSlice] objects to express the send/recv set and so it is very small

        # so basically both the way the messsage is broadcast and its size will be compressed but the
        # send pattern and the meaning of the message will be the same as this ineffiecient form

        from_ranks = NDSlice(self.slicing.processes.offset, sizes, strides)
        to_ranks = mesh.processes
        r = Tensor(self.tensor._fake, mesh, stream._active, False)
        assert r.ref is not None
        self.tensor.mesh.ctrl.send([from_ranks, to_ranks], messages.SendTensor(r, from_ranks, to_ranks, self.tensor, self.tensor._factory(), self.tensor.stream))
        self.tensor.mesh.ctrl.history.invocation((r,), (self.tensor,))
        return r

def _fake_reduce(tensor, source_mesh: 'DeviceMesh', dim: 'str', reduction, scatter: bool):
    if scatter:
        N = source_mesh.processes.sizes[source_mesh.names.index(dim)]
        if tensor.ndim == 0 or tensor.size(0) != N:
            raise TypeError(f'When scattering results the outer most dimension of tensor ({list(tensor.size())} must match the size ({N}) of the dimension "{dim}" being reduced')
        if reduction == 'stack':
            # scatter removes a dimension of mesh size
            # but gather adds the dimension back
            return tensor
        return tensor.sum(dim=0)
    else:
        if reduction == 'stack':
            N = source_mesh.processes.sizes[source_mesh.names.index(dim)]
            return torch.empty([N, *tensor.shape],
                               dtype=tensor.dtype, device=tensor.device, layout=tensor.layout)
        return tensor.add(tensor)


_explain = """\
LOCAL_TENSOR
A local (non-distributed) tensor is being passed while a device_mesh is active.
If you want to do local tensor compute use `with active_mesh(None):`

WRONG_MESH
A tensor being passed is on a device mesh that is not the current device_mesh.
Use `with active_mesh(m):` to switch the active mesh, or move the tensor to the correct device mesh with `to_mesh`/`on_mesh`.

WRONG_STREAM
A tensor being passed is on a stream that is not the current active stream. Use with `active_stream(s)` to switch streams, or
move the tensor to the correct stream with `.borrow`.

DROPPED
This tensor, or a view of it, was explicitly deleted with the t.drop() function and is no longer usable.
"""

explain = {}
for entry in _explain.split('\n\n'):
    k, v = entry.split('\n', 1)
    explain[k] = v

def _dtensor_input_check(func: 'RemoteFunction', args, kwargs, device_mesh, stream) -> Tuple[List['Tensor'], Any]:
    def stringify(t):
        if isinstance(t, Tensor):
            if t.mesh is not device_mesh:
                return 'WRONG_MESH'
            elif t.stream is not stream:
                return 'WRONG_STREAM'
            elif t.dropped:
                return 'DROPPED'
            else:
                return '.'
        elif isinstance(t, torch.Tensor):
            return 'LOCAL_TENSOR'
        else:
            return t

    def tensor_check(x):
        if isinstance(x, torch.Tensor):
            if isinstance(x, Tensor) and x.mesh is device_mesh and x.stream is stream and not x.dropped:
                device_mesh._use(x)
                return True
            fargs, fkwargs = tree_map(stringify, (args, kwargs))
            actuals = ', '.join(str(a) for a in fargs)
            if fkwargs:
                actuals = f'{actuals}, ' + ', '.join(f'{k}={v}' for k, v in kwargs.items())

            call = f"{func.path}({actuals})"
            dmesh = f'active_mesh = {device_mesh}\nactive_stream = {stream}'
            help = '\n'.join(f'{k}: {v}' for k, v in explain.items() if k in call)
            raise TypeError(f'Mismatched arguments to distributed tensor operation:\n\n  {call}\n\n{dmesh}\n{help}')

    dtensors, unflatten = flatten((args, kwargs), tensor_check)
    return dtensors, unflatten


def dtensor_check(func: 'RemoteFunction', args, kwargs, device_mesh: Optional['DeviceMesh'], stream: Stream):
    dtensors, unflatten = _dtensor_input_check(func, args, kwargs, device_mesh, stream)
    if device_mesh is None:
        raise ValueError("Remote functions require an active device mesh (use `with active_mesh(mesh):`")
    ctrl = device_mesh.ctrl
    stream._use_controller(ctrl)

    mutates = []
    assert callable(func.propagate)
    fake_input_tensors = [d._fake for d in dtensors]
    before_versions = [f._version for f in fake_input_tensors]
    fake_args, fake_kwargs = unflatten(fake_input_tensors)
    result = func.fake_call(fake_args, fake_kwargs)
    for i in range(len(dtensors)):
        if before_versions[i] < fake_input_tensors[i]._version:
            borrows = dtensors[i]._borrows
            writing_stream = borrows.writing_stream
            if writing_stream is not stream:
                reason = "it is read only because it is being borrowed" if writing_stream is None \
                            else f"it can only be mutated by f{writing_stream}"
                tbs = ''.join(f"Traceback of borrow to {k} (most recent frame last):\n{''.join(traceback.format_list(b.frames))}" for k,b in borrows.active.items())
                raise ValueError(f"\n{tbs}\nTensor input {i} would be mutated by this operator but {reason}")
            # we might still have alias to other streams if they have been dropped but not yet __del__'d
            mutates.extend(alias for alias in dtensors[i]._borrows.aliases if alias.stream is stream)
    return result, dtensors, tuple(mutates)

def dtensor_dispatch(func: 'RemoteFunction', args, kwargs, device_mesh: Optional['DeviceMesh'], stream: Stream):
    fake_result, dtensors, mutates = dtensor_check(func, args, kwargs, device_mesh, stream)
    assert device_mesh is not None

    fake_map = {id(d._fake): i for i, d in enumerate(dtensors)}
    fake_result_dtensors, unflatten_result = flatten(fake_result, lambda x: isinstance(x, torch.Tensor))
    # sometimes operators return references to inputs, in which case the result should be the same DTensor object
    # otherwise we create a new DTensor with a new RemoteRef for the result
    result_dtensors = tuple(dtensors[fake_map[id(fake)]] if id(fake) in fake_map else Tensor(fake, device_mesh, stream, False) for fake in fake_result_dtensors)
    ident = device_mesh.ctrl.history.ident(result_dtensors + mutates, dtensors)
    assert all(t.ref is not None for t in result_dtensors)
    assert all(t.ref is not None for t in mutates)
    device_mesh._send(messages.CallFunction(ident, result_dtensors, tuple(mutates), func, args, kwargs, stream))    
    results = unflatten_result(result_dtensors)
    # XXX - realistically this would be done on a non-python thread, keeping our messages up to date
    # but we can approximate it by checking for all ready meassages whenever we schedule new work
    device_mesh.ctrl._read_messages(0)
    return results
