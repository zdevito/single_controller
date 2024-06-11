from contextlib import contextmanager
from controller.ndslice import NDSlice
from supervisor import LocalMessageQueue, get_message_queue, TTL
from supervisor.logging import initialize_logging
from typing import Dict, NamedTuple, Any, Union, Optional, List, Tuple, Callable
import os
import importlib
import logging
from .tree import flatten, tree_map
import torch
import torch.distributed
from traceback import extract_tb
from . import messages
from .messages import RemoteFunction
from .tensor_factory import TensorFactory
from .reference import Ref

logger = logging.getLogger(__name__)

def log(*args):
    logger.info(*args)

class Dim(NamedTuple):
    rank: int
    size: int
    process_group: Any


class DeviceMesh:
    def __init__(self, names: Tuple[str, ...], ranks: NDSlice, rank: int):
        self.dims = {}
        coordinates = ranks.coordinates(rank)
        for coordinate, name, size, stride in zip(coordinates, names, ranks.sizes, ranks.strides):
            start = rank - stride*coordinate
            members = [*range(start, start + stride*size, stride)]
            assert members[coordinate] == rank
            process_group = torch.distributed.new_group(members, use_local_synchronization=True)
            self.dims[name] = Dim(coordinate, size, process_group)

def _rank(mesh: 'DeviceMesh', dim: str):
    return torch.full((), mesh.dims[dim].rank, dtype=torch.long)

class DependentOnError(Exception):
    def __init__(self, ident: int):
        self.ident = ident

class Stream:

    def __init__(self, default: bool):
        if default:
            self._cuda_stream = None
        else:
            self._cuda_stream = torch.cuda.Stream()

    @property
    def cuda_stream(self):
        if self._cuda_stream is None:
            return torch.cuda.current_stream()
        else:
            return self._cuda_stream

    @contextmanager
    def enable(self):
        if self._cuda_stream is None:
            yield
            return
        with torch.cuda.stream(self._cuda_stream):
            yield

    def event(self):
        e = torch.cuda.Event()
        self.cuda_stream.record_event(e)
        return e

    def wait_event(self, event):
        self.cuda_stream.wait_event(event)

    def wait_stream(self, stream):
        self.cuda_stream.wait_stream(stream.cuda_stream)

class Borrow:
    def __init__(self, from_stream: Stream, to_stream: Stream):
        self.from_stream = from_stream
        self.to_stream = to_stream
        self.event = from_stream.event()

    def use(self):
        self.to_stream.wait_event(self.event)
        self.event = None

    def drop(self):
        if self.event is not None:
            return
        self.from_stream.wait_stream(self.to_stream)


class Worker:
    def __init__(self, q: LocalMessageQueue, store, rank: int, world: int, local_rank: int):
        # remote ref id to local value
        self.env: Dict[int, Any] = {}
        self.q = q
        self.store = store
        self.rank = rank
        self.world = world
        self.local_rank = local_rank
        self.first_uncompleted_ident = 0
        self.last_send_status = 0
        self.borrows: Dict[int, Optional[Borrow]] = {}

    def handle_message(self, event: NamedTuple):
        cmd = event.__class__.__name__
        fn = getattr(self, cmd, None)
        if fn is not None:
            return fn(*event)
        raise RuntimeError(f"unhandled event: {event}")

    def CreateDeviceMesh(self, result: 'Ref', names: Tuple[str, ...], ranks: NDSlice):
        self.define(result, DeviceMesh(names, ranks, self.rank))

    def lookup(self, a: Any):
        if isinstance(a, Ref):
            r = self.env[a.id]
            if isinstance(r, DependentOnError):
                raise r
            return r
        return a

    def CallFunction(self, ident: int, results: Tuple[Ref], mutates: Tuple[Ref], rfunction: RemoteFunction, args: Tuple[Any, ...], kwargs: Dict[str, Any], streamref: Ref):
        with self.try_define(ident, results + mutates):
            stream: Stream = self.lookup(streamref)
            args, kwargs = tree_map(self.lookup, (args, kwargs))
            function = rfunction.resolve()
            with stream.enable():
                result = function(*args, **kwargs)
            tensors, _ = flatten(result, lambda x: isinstance(x, torch.Tensor))
            assert len(results) == len(tensors)
            for r, t in zip(results, tensors):
                self.define(r, t)

    def CreateStream(self, result: 'Ref', default: bool):
        self.define(result, Stream(default))

    @contextmanager
    def try_define(self, ident: int, results: Tuple[Ref, ...]):
        try:
            yield
        except DependentOnError as e:
            for r in results:
                self.define(r, e)
            # note: there is no need to to send RemoteFunctionFailed
            # because the controller would have already gotten and propagated the
            # original created of DependentOnError.
        except Exception as e:
            exc = DependentOnError(ident)
            for r in results:
                self.define(r, exc)
            logger.exception(f"Error generating {ident}")
            self.q.send(messages.RemoteFunctionFailed(ident, e, extract_tb(e.__traceback__)))
        self.first_uncompleted_ident = ident + 1      

    def FetchValue(self, ident: int, mutates: Tuple[Ref], rfunction: Optional[RemoteFunction], obj: Any, streamref: Ref):
        with self.try_define(ident, mutates):
            stream: Stream = self.lookup(streamref)
            obj = tree_map(self.lookup, obj)
            with stream.enable():
                if rfunction is not None:
                    function = rfunction.resolve()
                    obj = function(obj)
                self.q.send(messages.FetchResult(ident, obj))

    def RequestStatus(self, ident: int):
        self.first_uncompleted_ident = ident + 1
        self._send_status()

    def Exit(self):
        raise StopIteration()

    def CommandGroup(self, commands: List[NamedTuple]):
        for cmd in commands:
            self.handle_message(cmd)

    def DeleteRefs(self, refs: List[int]):
        for id in refs:
            del self.env[id]

    def BorrowCreate(self, result: Ref, tensorref: Ref, from_streamref, to_streamref, already_borrowed: bool):
        try:
            from_stream = self.lookup(from_streamref)
            to_stream = self.lookup(to_streamref)
            tensor = self.lookup(tensorref)
            self.define(result, tensor)
            if not already_borrowed:
                self.borrows[result.id] = Borrow(from_stream, to_stream)
        except DependentOnError as e:
            self.define(result, e)
            if not already_borrowed:
                self.borrows[result.id] = None

    def BorrowFirstUse(self, borrow: int):
        b = self.borrows[borrow]
        # can be none if the originator of the borrow errored.
        if b is not None:
            b.use()

    def BorrowDrop(self, borrow: int):
        b = self.borrows.pop(borrow)
        if b is not None:
            b.drop()

    def _reduce(self, local_tensor: torch.Tensor, source_mesh: DeviceMesh, dim: str, reduction: str, scatter: bool, inplace: bool):
        group = source_mesh.dims[dim].process_group

        if reduction == 'stack':
            if scatter:
                output = local_tensor
                if not inplace:
                    output = local_tensor.clone()
                torch.distributed.all_to_all_single(output, local_tensor, group=group)
                return output

            assert not inplace
            output = torch.empty([source_mesh.dims[dim].size, *local_tensor.shape],
                                 dtype=local_tensor.dtype, device=local_tensor.device, layout=local_tensor.layout)
            torch.distributed.all_gather_into_tensor(output, local_tensor, group=group)
            return output

        op = getattr(torch.distributed.ReduceOp, reduction.upper())

        if scatter:
            assert not inplace
            output = torch.empty(local_tensor.shape[1:], dtype=local_tensor.dtype,
                                 device=local_tensor.device, layout=local_tensor.layout)
            torch.distributed.reduce_scatter_tensor(output, local_tensor, op=op, group=group)
            return output

        output = local_tensor
        if not inplace:
            output = local_tensor.clone()
        torch.distributed.all_reduce(output, op=op, group=group)
        return output

    def Reduce(self, result: Ref, local_tensor_ref: Ref, factory: TensorFactory, source_mesh_ref: Ref, stream_ref: Ref, dim: str, reduction: str, scatter: bool, inplace: bool):
        source_mesh = self.lookup(source_mesh_ref)
        stream = self.lookup(stream_ref)
        with stream.enable():
            try:
                local_tensor = self.lookup(local_tensor_ref)
            except DependentOnError:
                # even if we were broken before, we have to participate in the collective
                # because we cannot signal to other ranks that we were broken
                # the controller will see the error message we sent before and know
                # the downstream values are broken.
                local_tensor = factory.zeros()
            # XXX - we should be careful about starting the collective with a tensor that doesn't match the expected
            # factory size. It can error. however, before we can do something about it we need to assign a failure
            # identity to this reduce object.
            output = self._reduce(local_tensor, source_mesh, dim, reduction, scatter, inplace)
            self.define(result, output)

    def SendTensor(self, result: Ref, from_ranks: NDSlice, to_ranks: NDSlice, tensorref: Ref, factory: TensorFactory, streamref):
        try:
            stream = self.lookup(streamref)
        except DependentOnError as e:
            self.define(result, e)
            return
        with stream.enable():
            ops = []
            P2POp = torch.distributed.P2POp
            isend, irecv = torch.distributed.isend, torch.distributed.irecv
            try:
                index = from_ranks.index(self.rank)
                try:
                    tensor = self.lookup(tensorref)
                except DependentOnError:
                    # XXX - how do we propagate this error on the host correctly?
                    # the host will see on status, but it will not immediately know
                    # what dependended on this downstream that also has to be invalid now.
                    tensor = factory.zeros()
                to_rank = to_ranks[index]
                ops.append(P2POp(isend, tensor, to_rank))
            except ValueError:
                to_rank = None
                tensor = None  # silence warnings

            try:
                index = to_ranks.index(self.rank)
                from_rank = from_ranks[index]
                if from_rank == to_rank:
                    assert tensor is not None
                    self.define(result,  tensor)
                recv = factory.empty()
                ops.append(P2POp(irecv, recv, from_rank))
                self.define(result, recv)
            except ValueError:
                pass
            # invoke batched p2p ops
            for op in torch.distributed.batch_isend_irecv(ops):
                op.wait()

    def define(self, r: Ref, value: Any):
        assert isinstance(r, Ref)
        self.env[r.id] = value

    def _send_status(self):
        if self.first_uncompleted_ident > self.last_send_status:
            self.q.send(messages.Status(self.first_uncompleted_ident))
            self.last_send_status = self.first_uncompleted_ident

    def event_loop(self):
        STATUS_INTERVAL = 1.0
        status_ttl = TTL(STATUS_INTERVAL)
        while True:
            try:
                _, msg = self.q.recv(timeout=status_ttl())
                logger.info(f"event: {msg}")
                self.handle_message(msg)
            except TimeoutError:
                pass
            except StopIteration:
                self.q.recvready(0)
                self.q.recvready(.01)
                return
            if status_ttl() == 0:
                status_ttl = TTL(STATUS_INTERVAL)
                self._send_status()


def worker_main(_restartable):
    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    initialize_logging(process_name=f'worker_{rank}')
    logger.info("starting, restartable=%s", _restartable)
    store = torch.distributed.TCPStore(os.environ['STORE_HOSTNAME'], int(os.environ['STORE_PORT']))
    torch.distributed.init_process_group(backend='nccl', world_size=world, rank=rank, store=store)
    b = torch.zeros(1, device='cuda')
    torch.distributed.all_reduce(b)
    q = get_message_queue()
    # CUDA_VISIBLE_DEVICES should be set on launch to LOCAL_RANK
    while True:
        worker = Worker(q, store, rank, world, local_rank)
        worker.event_loop()
        if not _restartable:
            break
        q.send(messages.Restarted(0))
        logger.info("restarting")
