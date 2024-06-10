from typing import TYPE_CHECKING, Dict, NamedTuple, Any, Union, Optional, List, Tuple, Callable
from traceback import FrameSummary
from .tensor_factory import TensorFactory
from .ndslice import NDSlice

if TYPE_CHECKING:
    from .stream import Stream
    from .tensor import Tensor
    from .device_mesh import DeviceMesh

class CreateDeviceMesh(NamedTuple):
    result: 'DeviceMesh'
    names: Tuple[str, ...]
    ranks: 'NDSlice'

class CreateStream(NamedTuple):
    result: 'Stream'
    default: bool

class CallFunction(NamedTuple):
    ident: int
    results: Tuple['Tensor', ...]
    mutates: Tuple['Tensor', ...]
    function: Union[str, Callable]
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    stream: 'Stream'

class Exit(NamedTuple):
    pass

class CommandGroup(NamedTuple):
    commands: List[NamedTuple]

class DeleteRefs(NamedTuple):
    refs: List[int]

class Restarted(NamedTuple):
    result: int

class FetchValue(NamedTuple):
    ident: int
    function: Optional[str]
    object: Any
    stream: 'Stream'

class FetchResult(NamedTuple):
    ident: int
    value: Any

class RemoteFunctionFailed(NamedTuple):
    failing_ident: int
    exception: Exception
    worker_frames: List[FrameSummary]

class Status(NamedTuple):
    first_uncompleted_ident: int

# When the controller is waiting on a status update,
# it will request one even if it is before the
# periodic one.
class RequestStatus(NamedTuple):
    ident: int

class BorrowCreate(NamedTuple):
    result: 'Tensor'
    tensor: 'Tensor'
    from_stream: 'Stream'
    to_stream: 'Stream'
    alias: bool  # is this an alias of an already existing borrow

class BorrowDrop(NamedTuple):
    borrow: int

class BorrowFirstUse(NamedTuple):
    borrow: int

class SendTensor(NamedTuple):
    result: 'Tensor'
    from_ranks: NDSlice
    to_ranks: NDSlice
    tensor: 'Tensor'
    factory: 'TensorFactory'
    stream: 'Stream'

class Reduce(NamedTuple):
    result: 'Tensor'
    local_tensor_ref: 'Tensor'
    factory: 'TensorFactory'
    source_mesh_ref: 'DeviceMesh'
    stream_ref: 'Stream'
    dim: str
    reduction: str
    scatter: bool
    inplace: bool
