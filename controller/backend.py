from supervisor import Context, Host, FunctionCall, TTL, Process
from typing import Optional, List, Sequence, NamedTuple, Tuple, Union

from torch.distributed import TCPStore
from .ndslice import NDSlice

import socket
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class Backend(ABC):
    @abstractmethod
    def send(self, ranks: Union[NDSlice, List[NDSlice]], msg) -> None:
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


class ProcessBackend(Backend):
    def __init__(self, ctx: Context, hosts: List[Host], gpu_per_host: int, _processes=None, _store=None):
        self.ctx = ctx
        self.hosts = hosts
        self.store = self._create_store() if _store is None else _store
        self.all_processes = self._create_pg(ctx, hosts, gpu_per_host, self.store) if _processes is None else _processes

    @property
    def world_size(self):
        return len(self.all_processes)

    def send(self, ranks: Union[NDSlice, List[NDSlice]], msg) -> None:
        # the intent is for this to be optimized as tree broadcast
        # base on if members of tree nodes overlap with a slice.
        if isinstance(ranks, NDSlice):
            for rank in ranks:
                self.all_processes[rank].send(msg)
        else:
            sent = set()
            for slice in ranks:
                for rank in slice:
                    if rank not in sent:
                        self.all_processes[rank].send(msg)
                        sent.add(rank)

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
