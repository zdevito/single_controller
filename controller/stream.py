from contextlib import contextmanager
from typing import TYPE_CHECKING, Optional, List, Callable
from . import messages
from .reference import Referenceable
from .borrows import Borrow
import traceback

if TYPE_CHECKING:
    from .controller import Controller
    from .tensor import Tensor

class Stream(Referenceable):
    def __init__(self, name: str, _default=False):
        self.name = name
        self.default: bool = _default
        self.ctrl: Optional['Controller'] = None
        self.ref: Optional[int] = None

    def __repr__(self):
        return f'<Stream({repr(self.name)}) at {hex(id(self))}>'

    def __str__(self):
        return f'stream {repr(self.name)}'

    def _use_controller(self, ctrl: 'Controller'):
        if self.ctrl is None:
            self.ctrl = ctrl
        elif self.ctrl is not ctrl:
            raise TypeError('DeviceMesh and stream controller are different.')

    def delete_ref(self, ref):
        # streams are not frequently created/destroyed so
        # no need to buffer the delets
        assert self.ctrl is not None
        if not self.ctrl._shutdown:
            self.ctrl.send(self.ctrl.all_ranks, messages.DeleteRefs([ref]))

    def define_ref(self):
        assert self.ctrl is not None
        self.ref = self.ctrl.ref()
        self.ctrl.send(self.ctrl.all_ranks, messages.CreateStream(self, self.default))

    @contextmanager
    def coalesce(self):
        """
        Delay issuing operators to this stream, grouping them into one big operation that will run once this context manager exits.
        For data movement, this allows us to group the operators together. However coalescing too many ops together will expose
        more scheduling overhead that is normally pipelined with work. So avoid globally coalescing huge parts of a network.
        """
        raise NotImplementedError()

    def borrow(self, t: 'Tensor', mutable: bool = False) -> 'Tensor':
        """
        Borrows tensor 't' for use on this stream.
        The memory of t will stay alive until t.drop() is called, which will free t and
        and any of its alises on stream `self` and will cause t.stream to wait on self at that point so
        that the memory of t can be reused.

        If `mutable` then self can write to the storage of `t`, but t.stream cannot read or write `t` until,
        the borrow is returned (becomes free and a wait_for has been issued).

        If not `mutable` both `self` and `t.stream` can read from t's storage but neither can write to it.
        """
        self._use_controller(t.mesh.ctrl)
        assert self.ctrl is not None
        borrows = t._borrows
        if mutable and borrows.active:
            raise RuntimeError("Cannot borrow this tensor mutably because it (or a view) is already being borrowed non-mutably.")

        already_borrowed = self in borrows.active
        r = type(t)(t._fake, t.mesh, self, True)
        self.ctrl.history.invocation((r,), (t,))
        assert r.ref is not None
        t.mesh._send(messages.BorrowCreate(r, t, t.stream, self, already_borrowed))
        if not already_borrowed:
            borrows.active[self] = Borrow(r.ref, False, traceback.extract_stack())
            borrows.writing_stream = self if mutable else None
        return r


_active = Stream('main')
_on_change: List[Callable] = []

@contextmanager
def active_stream(stream: Stream):
    global _active
    for on_change in _on_change:
        on_change(_active, stream)

    _active, old = stream, _active
    try:
        yield
    finally:
        _active = old
