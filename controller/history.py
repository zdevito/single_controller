from typing import TYPE_CHECKING, Any, Sequence, Optional, List
from collections import deque
import itertools
import traceback
if TYPE_CHECKING:
    from .tensor import Tensor
    from .future import Future

class RemoteException(Exception):
    def __init__(self, ident: int, exception: Exception, controller_frames: List[traceback.FrameSummary], worker_frames: List[traceback.FrameSummary]):
        self.ident = ident
        self.exception = exception
        self.controller_frames = controller_frames
        self.worker_frames = worker_frames

    def __str__(self):
        try:
            exe = str(self.exception)
            worker_tb = ''.join(traceback.format_list(self.worker_frames))
            controller_tb = ''.join(traceback.format_list(self.controller_frames)) if self.controller_frames is not None else '<unset>'
            return f'A remote function has failed asynchronously.\n' \
                f'Traceback of where the remote function was issued on controller (most recent call last):\n{controller_tb}' \
                f'Traceback of where the remote function failed on worker (most recent call last):\n{worker_tb}{type(self.exception).__name__}: {exe}'
        except Exception as e:
            print(e)
            return "oops"

class Invocation:
    def __init__(self, traceback: List[traceback.FrameSummary]):
        self.traceback = traceback
        self.users = set['Invocation']()
        self.failure: Optional[RemoteException] = None
        self.fut: Optional['Future'] = None
        self.fut_value: Any = None

    def __repr__(self):
        return f'<Invocation {self.fut}>'

    def fail(self, remote_exception: RemoteException):
        if self.failure is None or self.failure.ident > remote_exception.ident:
            self.failure = remote_exception
            return True
        return False

class History:
    def __init__(self, N):
        self.next_ident = itertools.count()
        self.first_uncompleted_ident = [0 for _ in range(N)]
        self.min_first_uncompleted_ident = 0
        self.invocations = deque[Invocation]()
        self.last_assigned_ident = -1

    def invocation(self, defs: Sequence['Tensor'], uses: Sequence['Tensor']):
        r = Invocation(traceback.extract_stack()[:-2])
        for t in uses:
            u = t._invocation
            assert u is not None
            u.users.add(r)
            if u.failure is not None:
                r.fail(u.failure)
        for t in defs:
            t._invocation = r
        return r

    def ident(self, defs: Sequence['Tensor'], uses: Sequence['Tensor'], future: Optional['Future'] = None) -> int:
        r = self.last_assigned_ident = next(self.next_ident)
        invocation = self.invocation(defs, uses)
        invocation.fut = future
        self.invocations.append(invocation)
        return r

    def propagate_failure(self, ident, exception, worker_frames):
        invocation = self.invocations[ident - self.min_first_uncompleted_ident]
        remote_exception = RemoteException(ident, exception, invocation.traceback, worker_frames)
        worklist = deque((invocation,))
        while worklist:
            invocation = worklist.popleft()
            if invocation.fail(remote_exception):
                worklist.extend(invocation.users)

    def rank_completed(self, rank, first_uncompleted_ident):
        # advance what our last completed action was, and
        # trim the list of tracebacks if we no longer need them.
        prev = self.first_uncompleted_ident[rank]
        self.first_uncompleted_ident[rank] = first_uncompleted_ident
        if prev == self.min_first_uncompleted_ident:
            self.min_first_uncompleted_ident = min(self.first_uncompleted_ident)
            for i in range(self.min_first_uncompleted_ident - prev):
                invocation = self.invocations.popleft()
                if invocation.fut is not None:
                    invocation.fut._set_result(invocation.fut_value if invocation.failure is None else invocation.failure)

    def future_completed(self, ident, value):
        invocation = self.invocations[ident - self.min_first_uncompleted_ident]
        invocation.fut_value = value
