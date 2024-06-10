from supervisor import TTL
from typing import Dict, Optional, List, NamedTuple, Union
from concurrent.futures import ThreadPoolExecutor
from torch._subclasses.fake_tensor import FakeTensorMode

import torch
import torch.distributed
from . import messages, stream
from .history import History, RemoteException
from .borrows import Borrows
from .stream import Stream
from .device_mesh import DeviceMesh
from .ndslice import NDSlice
from .backend import Backend

from collections import defaultdict
import itertools
import traceback
from weakref import WeakKeyDictionary
import logging

logger = logging.getLogger(__name__)

check_correctness_per_operator = False
if check_correctness_per_operator:
    class RealMode:
        def from_tensor(self, t):
            return t

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass

    fake_mode = RealMode()
else:
    fake_mode = FakeTensorMode()

_CONTROLLER_STATUS_INTERVAL = 2

class Controller:
    def __init__(self, backend: Backend):
        self.backend = backend
        self.next_ref = itertools.count()
        self.exited = {}
        self.failures: Dict[int, Dict[int, RemoteException]] = defaultdict(dict)
        self.pending_del: Dict[DeviceMesh, List[int]] = defaultdict(list)
        self._shutdown = False
        self.incomplete_futures = {}
        self.controller_status_ttl = TTL(_CONTROLLER_STATUS_INTERVAL)
        self.allocation_borrows: WeakKeyDictionary[torch.UntypedStorage, Borrows] = WeakKeyDictionary()
        self.fake_mode_worker = ThreadPoolExecutor(max_workers=1)
        self.history = History(self.backend.world_size)
        stream._active = Stream("main", _default=True)

    def _run_fake(self, func, args, kwargs):
        def work():
            with fake_mode:
                return func(*args, **kwargs)
        return self.fake_mode_worker.submit(work).result()

    def send(self, ranks: Union[NDSlice, List[NDSlice]], msg: NamedTuple):
        self.backend.send(ranks, msg)

    @property
    def all_ranks(self):
        return NDSlice(0, [self.backend.world_size], [1])

    def shutdown(self):
        self._shutdown = True
        self.send(self.all_ranks, messages.Exit())
        while len(self.exited) < self.backend.world_size:
            rank, msg = self.backend.recv(timeout=None)
            self.handle_message(rank, msg)

    def ref(self) -> int:
        return next(self.next_ref)

    def _flush_deletes(self, device_mesh: 'DeviceMesh') -> Optional[messages.DeleteRefs]:
        to_delete = None
        if device_mesh in self.pending_del:
            to_delete = messages.DeleteRefs(self.pending_del.pop(device_mesh))
        # we also have to make sure if we have deletes to other device meshes,
        # they get processed before we do an op that will try to allocate memory
        for k, v in self.pending_del.items():
            k._send(messages.DeleteRefs(v), flush_deletes=False)
        self.pending_del.clear()
        return to_delete

    def _request_status(self):
        self.send(self.all_ranks, messages.RequestStatus(self.history.last_assigned_ident))

    def _read_messages(self, timeout: Optional[float]):
        # XXX - how can we avoid always requesting status when waiting on futures?
        #       we need to figure out what submesh we need to hear from before a future
        #       is considered 'good'. This means not just waiting for the future value
        #       but also for signal that any failures that could invalidate the future have
        #       not happened. We could do better if tensors/collectives had an invalid bit
        #       that we propagate. In real uses fetches might lag behind anyway so we would not
        #       have to send out so many requests for current status.
        for rank, value in self.backend.recvready(timeout):
            self.handle_message(rank, value)

    def handle_message(self, sender, value):
        getattr(self, value.__class__.__name__)(sender, *value)

    def ProcessExited(self, proc, result):
        self.exited[proc] = result

    def ProcessStarted(self, proc, pid):
        pass

    def Restarted(self, proc, result):
        self.exited[proc] = result

    def FetchResult(self, proc, ident, value):
        self.history.future_completed(ident, value)

    def RemoteFunctionFailed(self, proc, failing_ident, exception: Exception, worker_frames: List[traceback.FrameSummary]):
        self.history.propagate_failure(failing_ident, exception, worker_frames)
        self.history.rank_completed(proc, failing_ident)

    def Status(self, proc, first_uncompleted_ident):
        self.history.rank_completed(proc, first_uncompleted_ident)
