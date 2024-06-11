from supervisor import Context, Host, FunctionCall, TTL, Process
from typing import Optional, List, Sequence, NamedTuple, Tuple, Union
import torch
from torch.distributed import TCPStore
from .ndslice import NDSlice
from collections import deque
from . import messages
from .tree import flatten

import socket
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


Ranks = Union[NDSlice, List[NDSlice]]

class Backend(ABC):
    @abstractmethod
    def send(self, ranks: Ranks, msg) -> None:
        raise NotImplementedError()
    
    @abstractmethod
    def recvready(self, timeout: Optional[float]) -> Sequence[Tuple[int, NamedTuple]]:
        raise NotImplementedError()

    @abstractmethod
    def recv(self, timeout: Optional[float]) -> Tuple[int, NamedTuple]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def world_size(self):
        raise NotImplementedError()


def _ranks(ranks: Ranks):
    if isinstance(ranks, NDSlice):
        yield from ranks
    else:
        seen = set()
        for slice in ranks:
            for rank in slice:
                if rank not in seen:
                    yield rank
                    seen.add(rank)

class ProcessBackend(Backend):
    def __init__(self, ctx: Context, hosts: List[Host], gpu_per_host: int, _processes=None, _store=None):
        self.ctx = ctx
        self.hosts = hosts
        self.store = self._create_store() if _store is None else _store
        self.all_processes = self._create_pg(ctx, hosts, gpu_per_host, self.store) if _processes is None else _processes

    @property
    def world_size(self):
        return len(self.all_processes)

    def send(self, ranks: Ranks, msg) -> None:
        # the intent is for this to be optimized as tree broadcast
        # base on if members of tree nodes overlap with a slice.
        for rank in _ranks(ranks):
            self.all_processes[rank].send(msg)

    def recvready(self, timeout: Optional[float]) -> Sequence[Tuple[int, NamedTuple]]:
        result = []
        for sender, msg in self.ctx.recvready(timeout):
            if isinstance(sender, Process):
                result.append((sender.rank, msg))
            else:
                logger.warning("TODO: ignoring non-process message: %s %s", sender, msg)
        return result

    def recv(self, timeout: Optional[float]) -> Tuple[int, NamedTuple]:
        while True:
            sender, msg = self.ctx.recv(timeout)
            if isinstance(sender, Process):
                return (sender.rank, msg)
            else:
                logger.warning("TODO: ignoring non-process message: %s %s", sender, msg)

    @staticmethod
    def _create_store():
        hostname = socket.gethostname()
        with socket.socket() as sock:
            sock.bind(('', 0))
            port = sock.getsockname()[1]
        return TCPStore(hostname, port, is_master=True)

    @staticmethod
    def _create_pg(ctx: Context, hosts: List[Host], gpu_per_host: int, store,
                   _restartable=False):
        return ctx.create_process_group(hosts,
                                        FunctionCall('controller.worker.worker_main',
                                                     _restartable=_restartable),
                                        processes_per_host=gpu_per_host,
                                        env={'CUDA_VISIBLE_DEVICES':  '$LOCAL_RANK',
                                             # supervisor_pipe is a unique ID per Host object,
                                             # so it lets us put multiple processes on the same GPU.
                                             'NCCL_HOSTID': '$SUPERVISOR_PIPE',
                                             'STORE_HOSTNAME': store.host,
                                             'STORE_PORT': str(store.port), })

class MockBackend(Backend):
    def __init__(self, world_size: int):
        self._world_size = world_size
        self.responses = deque[Tuple[int, NamedTuple]]()

    @property
    def world_size(self):
        return self._world_size

    def RequestStatus(self, ranks: Ranks, ident: int):
        for rank in _ranks(ranks):
            self.responses.append((rank, messages.Status(ident + 1)))

    def FetchValue(self, ranks, ident, mutates, function, object, stream):
        dtensors, unflatten = flatten(object, lambda x: isinstance(x, torch.Tensor))
        fake_obj = unflatten(d._fake for d in dtensors) 
        if function is not None:
            fake_obj = function.fake_call((fake_obj,), {})
        for rank in _ranks(ranks):
            self.responses.append((rank, messages.FetchResult(ident, fake_obj)))

    def send(self, ranks: Union[NDSlice, List[NDSlice]], msg) -> None:
        attr = getattr(self, type(msg).__name__, None)
        if attr is not None:
            attr(ranks, *msg)
        print(ranks, msg)

    def recvready(self, timeout: Optional[float]) -> Sequence[Tuple[int, NamedTuple]]:
        r = tuple(self.responses)
        self.responses.clear()
        return r

    def recv(self, timeout: Optional[float]) -> Tuple[int, NamedTuple]:
        if self.responses:
            return self.responses.popleft()
        raise TimeoutError()
