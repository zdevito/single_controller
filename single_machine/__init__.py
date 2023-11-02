import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten, tree_unflatten, tree_map
from typing import NamedTuple, List, Union, Literal
from contextlib import contextmanager, nullcontext
from .base_tensor import BaseTensor
import sys
import pickle
import threading
import traceback
import atexit
from functools import cache
import os
from torch._subclasses.fake_tensor import FakeTensorMode
import tempfile
import warnings
from queue import Queue

from single_controller.config import check_correctness_per_operator, simulate_function_calls, do_fake_mode_caching

class Stats:
    def report(self):
        print(self.__dict__)

stats = Stats()
stats.mode_change = 0
stats.fake_tensor = 0
stats.dispatched = 0

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

_next_handle = 0
class RemoteRef:
    def __init__(self):
        global _next_handle
        _next_handle += 1
        self.id = _next_handle

    def __repr__(self):
        return f"r{self.id}"


def is_tensor(x):
    return isinstance(x, torch.Tensor)

@cache
def sharding_to_find(func, *args):
    print("Using FAKE sharding rule, things may break: ", str(func), func._schema)
    pass

prules = {}

def prule(name):
    def wrap(x):
        if isinstance(name, str):
            prules[name] = x
        else:
            for n in name:
                prules[n] = x
    return wrap

@prule("aten.sum.dim_IntList")
def rule(args, kwargs, dtensor_input, dtensor_results, sharding_context):
    t, dims = args[0:2]
    annotation_out = t.sharding.sharding.copy()
    for d in dims:
        try:
            idx = annotation_out.index(d)
            annotation_out[idx] = '+'
        except ValueError:
            pass
    dtensor_results[0]._sharding = Sharding(t.sharding.mesh, annotation_out)
    return args, kwargs

@prule('aten._scaled_dot_product_efficient_attention.default')
def attension(args, kwargs, dtensor_input, dtensor_results, sharding_context):
    a = dtensor_input[0]
    dtensor_results[0]._sharding = a.sharding
    dtensor_results[1]._sharding = a.sharding
    dtensor_results[2]._sharding = Sharding(a.sharding.mesh, 'r')
    dtensor_results[3]._sharding = Sharding(a.sharding.mesh, 'r')
    return args, kwargs


@prule(['aten._to_copy.default', 'aten.detach.default', 'aten.split.Tensor', 'aten.gelu.default', 'aten.native_dropout.default', 'aten.ones_like.default', 'aten.native_dropout_backward.default', 'aten.gelu_backward', 'aten.gelu_backward.default'])
def same_as_input(args, kwargs, dtensor_input, dtensor_results, sharding_context):
    a = dtensor_input[0]
    for r in dtensor_results:
        r._sharding = a._sharding
    return args, kwargs


@prule(['aten.ones_like.default', 'aten.zeros_like.default'])
def same_as_input(args, kwargs, dtensor_input, dtensor_results, sharding_context):
    a = dtensor_input[0]
    if any(isinstance(x, int) for x in a.sharding.sharding):
        dtensor_results[0]._sharding = a.sharding
    else:
        dtensor_results[0]._sharding = Sharding(a.sharding.mesh, 'r')
    return args, kwargs

@prule(['aten.zeros.default', 'aten.arange.start','aten.full.default'])
def ctor(args, kwargs, dtensor_input, dtensor_results, sharding_context):
    assert sharding_context, "No context for constructor?"
    for r in dtensor_results:
        r._sharding = sharding_context
    return args, kwargs


def _check_all(t, v):
    assert all(s == v for s in t.sharding.sharding), f"all shardings must be {v}"
def _check_not(t, v):
    assert all(s != v for s in t.sharding.sharding), f"all shardings must not be {v}"

@prule('aten.embedding.default')
def embedding(args, kwargs, dtensor_input, dtensor_results, sharding_context):
    weight, indices = args
    _check_all(weight, 'r')
    _check_not(indices, '+')
    dtensor_results[0]._sharding = indices._sharding
    return args, kwargs

@prule('aten.native_layer_norm.default')
def layer_norm(args, kwargs, dtensor_input, dtensor_results, sharding_context):
    input = args[0]
    _check_not(input, input.dim() - 1)
    for r in dtensor_results:
        r._sharding = input._sharding
    return args, kwargs

@prule('aten.native_layer_norm_backward.default')
def layer_norm_backward(args, kwargs, dtensor_input, dtensor_results, sharding_context):
    input = args[1]
    dtensor_results[0]._sharding = input.sharding
    dtensor_results[1]._sharding = Sharding(input.sharding.mesh, ['+'])
    if len(dtensor_results) == 3:
        dtensor_results[2]._sharding = Sharding(input.sharding.mesh, ['r'])
    return args, kwargs

@prule(['aten._log_softmax.default', 'aten.select.int', 'aten.cat.default'])
def _log_softmax(args, kwargs, dtensor_input, dtensor_results, sharding_context):
    input, dim = args[0:2]
    _check_not(input, dim)
    dtensor_results[0]._sharding = input._sharding
    return args, kwargs

@prule(['aten.cat.default'])
def cat(args, kwargs, dtensor_input, dtensor_results, sharding_context):
    inputs, dim = args[0:2]
    for input in inputs:
        _check_not(input, dim)
    dtensor_results[0]._sharding = input._sharding
    return args, kwargs


@prule('aten.t.default')
def t(args, kwargs, dtensor_input, dtensor_results, sharding_context):
    s = args[0]
    remap = {0: 1, 1: 0}
    new_sharding_ann = [remap.get(x, x) for x in s.sharding.sharding]
    dtensor_results[0]._sharding = Sharding(s.sharding.mesh, new_sharding_ann)
    return args, kwargs

@prule('aten.mean.default')
def t(args, kwargs, dtensor_input, dtensor_results, sharding_context):
    t = args[0]
    _check_all(t, 'r')
    dtensor_results[0]._sharding = t.sharding
    return args, kwargs


@prule('aten.transpose.int')
def t(args, kwargs, dtensor_input, dtensor_results, sharding_context):
    s, dim0, dim1 = args
    remap = {dim0: dim1, dim1: dim0}
    new_sharding_ann = [remap.get(x, x) for x in s.sharding.sharding]
    dtensor_results[0]._sharding = Sharding(s.sharding.mesh, new_sharding_ann)
    return args, kwargs

@prule(['aten.view.default', 'aten._unsafe_view.default'])
def view(args, kwargs, dtensor_input, dtensor_results, sharding_context):
    t, size = args
    # this is wrong in general, but works for our specific case
    mesh_sizes = t.sharding.mesh.shape.size()
    new_size = tuple(s // mesh_sizes[t.sharding.sharding.index(i)] if i in t.sharding.sharding else s for i, s in enumerate(size))
    dtensor_results[0]._sharding = t.sharding
    return (t, new_size), kwargs

@prule('aten.mm.default')
def mm(args, kwargs, dtensor_input, dtensor_results, sharding_context):
    a, b = args
    # special case the partial reduce until we have better generic sharding propagation rules
    if a.sharding.sharding == [1] and b.sharding.sharding == [0]:
        dtensor_results[0]._sharding = Sharding(a.sharding.mesh, ['+'])
        return args, kwargs
    _check_all(b, 'r')
    _check_not(a, '+')
    _check_not(a, 1)
    dtensor_results[0]._sharding = a._sharding
    return args, kwargs

@prule('aten.copy_.default')
def cp(args, kwargs, dtensor_input, dtensor_results, sharding_context):
    dst, src = args[0:2]
    assert dst.sharding.sharding == src.sharding.sharding, "same sharding"
    return args, kwargs

@prule('aten.nll_loss_forward.default')
def nll_loss(args, kwargs, dtensor_input, dtensor_results, sharding_context):
    # print("LOSS ----------------------------------------------------------------------")
    self, target = args[0:2]
    assert self.sharding.sharding == target.sharding.sharding, "same sharding"
    _check_not(self, '+')
    for r in dtensor_results:
        r._sharding = Sharding(self.sharding.mesh, ['+' if isinstance(x, int) else 'r' for x in self.sharding.sharding])
    return args, kwargs


@prule(['aten.nll_loss_backward.default',  'aten._log_softmax_backward_data.default'])
def nll_loss(args, kwargs, dtensor_input, dtensor_results, sharding_context):
    self = args[1]
    dtensor_results[0]._sharding = self.sharding
    return args, kwargs


@prule(['aten.embedding_dense_backward.default'])
def embedding_backward(args, kwargs, dtensor_input, dtensor_results, sharding_context):
    gradOutput = args[0]
    if gradOutput.sharding.sharding[0] == 'r':
        dtensor_results[0]._sharding = gradOutput.sharding
    else:
        dtensor_results[0]._sharding = Sharding(gradOutput.sharding.mesh, '+')
    return args, kwargs

def all_implicit_or_replicated(sharding):
    for ann in sharding.sharding:
        if ann == '+' or isinstance(ann, int):
            return False
    return True

def try_fast_propagate_sharding(dtensor_input, sharding_context, mesh):
    can_fast_propagate = True if sharding_context is None else all_implicit_or_replicated(sharding_context)
    for d in dtensor_input:
        if d._sharding.mesh is not mesh:
            raise NotImplementedError("operation on tensors distributed on different device meshes")
        can_fast_propagate = can_fast_propagate and all_implicit_or_replicated(d._sharding)
    if not can_fast_propagate:
        return None
    # easy case, only implicit batching or replicated, so the rule is that things become
    # implicitly batched if one input is implicitly batch in that mesh dimension
    # otherwise they stay replicated
    if sharding_context is not None:
        sharding = sharding_context
    else:
        new_annotations = ['b' if any(e == 'b' for e in cross_input_annotations) else 'r' for cross_input_annotations in zip(*(d.sharding.sharding for d in dtensor_input))]
        sharding = Sharding(mesh, new_annotations)
    return sharding

def propagate_sharding(func, args, kwargs, dtensor_input: 'List[DTensor]', dtensor_results: 'List[DTensor]', sharding_context: 'Sharding', mesh):
    # fake cases
    key = str(func)
    if key in prules:
        args, kwargs = prules[key](args, kwargs, dtensor_input=dtensor_input, dtensor_results=dtensor_results, sharding_context=sharding_context)
        return args, kwargs

    sharding_to_find(func, str(args), str(kwargs), str(dtensor_results))
    # totally fake for now, propagates the sharding pointwise if all the input shardings are the same or an input is fully replicated.
    # will kinda do a ddp like thing (batch stays batched, weights stay replicated), but some ops would be unsafe
    if dtensor_input:
        assert sharding_context is None
        new_annotations = None
        for d in dtensor_input:
            if any(s != 'r' for s in d._sharding.sharding):
                if new_annotations is None:
                    new_annotations = d._sharding.sharding
                elif d._sharding.sharding != new_annotations:
                    raise NotImplementedError(f"mixed sharding annotations: {new_annotations} != {d._sharding.sharding}")
        if new_annotations is None:
            sharding = dtensor_input[0]._sharding
        else:
            sharding = Sharding(mesh, new_annotations)
    else:
        assert sharding_context
        sharding = sharding_context

    for r in dtensor_results:
        r._sharding = sharding

    return args, kwargs

def dtensor_dispatch(func, args=(), kwargs=None, sharding=None):
    if func is torch.ops.aten.detach.default:
        if not args[0].requires_grad:
            return args[0]
    stats.dispatched += 1
    worker = sharding.mesh.flat_workers[0] if sharding else None

    def stringify(t):
        if isinstance(t, DTensor):
            return 'DTensor'
        elif isinstance(t, torch.Tensor):
            return 'Tensor'
        else:
            return t

    def is_dtensor_no_tensors(x):
        if isinstance(x, DTensor):
            return True
        elif isinstance(x, torch.Tensor):
            raise NotImplementedError(f"mixed DTensor/local tensor {func}(args, kwargs)={tree_map(stringify, (args, kwargs))}")

    dtensors, unflatten = flatten((args, kwargs), is_dtensor_no_tensors)
    mesh = dtensors[0]._sharding.mesh if dtensors else sharding.mesh

    fast_prop_sharding = try_fast_propagate_sharding(dtensors, sharding, mesh)

    for d in dtensors:
        d.set_fake_is_first_worker(fast_prop_sharding is not None)

    fake_input_tensors = [d._fake for d in dtensors]
    if fast_prop_sharding is None:
        ctx = fake_mode
        stats.fake_tensor += 1
    else:
        ctx = nullcontext()

    with ctx:
        fake_args, fake_kwargs = unflatten(fake_input_tensors)
        result = func(*fake_args, **fake_kwargs)

    fake_result_dtensors, unflatten_result = flatten(result, is_tensor)
    fake_map = {id(f): i for i, f in enumerate(fake_input_tensors)}

    # sometimes operators return references to inputs, in which case the result should be the same DTensor object
    # otherwise we create a new DTensor with a new RemoteRef for the result
    result_dtensors = [dtensors[fake_map[id(fake)]] if id(fake) in fake_map else DTensor(fake, RemoteRef(), fast_prop_sharding, fast_prop_sharding is not None) for fake in fake_result_dtensors]

    if not fast_prop_sharding:
        modified_args, modified_kwargs = propagate_sharding(func, args, kwargs, dtensors, result_dtensors, sharding, mesh)
    else:
        modified_args, modified_kwargs = args, kwargs

    for i, r in enumerate(result_dtensors):
        assert r._sharding is not None, f"sharding unset for output {i} of {str(func)} fake outputs: {fake_result_dtensors}"

    def get_ref(x):
        return x if not isinstance(x, DTensor) else x._ref

    ref_args, ref_kwargs = tree_map(get_ref, modified_args), tree_map(get_ref, modified_kwargs)
    ref_results = [r._ref for r in result_dtensors]
    workers = mesh.flat_workers if fast_prop_sharding is None else mesh.flat_workers[1:]
    for worker in workers:
        worker.send_command(func, ref_args, ref_kwargs, ref_results)

    results = unflatten_result(result_dtensors)
    return results



class FakeTLS:
    pass
# thread local doesn't quite work right because backwards pass runs in another thread
# but we don't move the resource guard over there
tls = threading.local()

@contextmanager
def active_sharding(sharding, force=False):
    if not hasattr(tls, 'sharding'):
        tls.sharding = 'inactive'
    sharding = None if sharding is None else Sharding.lift(sharding)
    old_sharding = tls.sharding
    ctx = _ActiveSharding() if old_sharding == 'inactive' or force else nullcontext()
    with ctx:
        tls.sharding = sharding
        try:
            yield
        finally:
            tls.sharding = old_sharding

def _current_active_sharding():
    s = getattr(tls, 'sharding', None)
    if s == 'inactive':
        s = None
    return s

class _ActiveSharding(TorchDispatchMode):
    ignore = ['profiler._record_function_exit._RecordFunction']
    def __torch_dispatch__(self, func, types, args, kwargs):
        if getattr(tls, 'sharding', None) is None or str(func) in self.ignore:
            return func(*args, **kwargs)
        for x in tree_flatten((args, kwargs))[0]:
            if isinstance(x, torch.Tensor):
                return func(*args, **kwargs)
        assert tls.sharding != 'inactive', "_ActiveSharding is enabled but set to inactive?"
        return dtensor_dispatch(func, args, kwargs, sharding=tls.sharding)


class DTensor(BaseTensor):
    @staticmethod
    def __new__(
        cls,
        fake: torch.Tensor,
        ref: RemoteRef,
        sharding: 'Optional[Sharding]',
        fake_is_first_worker,
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
        r._ref = ref
        r._fake = fake
        assert sharding is None or isinstance(sharding, Sharding)
        r._sharding = sharding
        r._fake_is_first_worker = fake_is_first_worker
        return r

    def __init__(self, fake, ref, worker, fake_is_first_worker):
        pass

    @classmethod
    def to_remote(cls, t: torch.Tensor, sharding: 'Sharding'):
        sharding = Sharding.lift(sharding)
        f = fake_mode.from_tensor(t)
        r = RemoteRef()

        shape = sharding.mesh.shape

        batch_dim = 0

        def split_dim(i, to_split):
            nonlocal t
            sizes = t.size()
            split_adjusted = to_split + i
            d = sizes[split_adjusted]
            assert d % shape.size(i) == 0, "NOT EVENLY SPLIT"
            chunk_size = d // shape.size(i)
            t = t.reshape(*sizes[:split_adjusted], shape.size(i), chunk_size, *sizes[split_adjusted+1:])
            t = t.movedim(split_adjusted, i)
            return chunk_size

        for i, ann in enumerate(sharding.sharding):
            if ann == "r":
                t = t.expand(shape.size(i), *t.size())
                t = t.movedim(0, i)
            elif isinstance(ann, int):
                split_dim(i, ann)
            elif ann == "b":
                chunk_size = split_dim(i, batch_dim)
                sizes = f.size()
                index = tuple(slice(0, chunk_size) if i == batch_dim else slice(None) for i in range(f.dim()))
                f = f[index]
                batch_dim += 1
            else:
                raise NotImplementedError(f"Annotation: {ann}")

        if shape.dim() == 0:
            worker = sharding.mesh._workers.workers[shape.item()]
            worker.send_value(r, t)
        else:
            t = t.flatten(0, shape.dim() - 1)
            shape_flat = shape.flatten()
            for idx, local in zip(shape_flat, t):
                worker = sharding.mesh._workers.workers[idx]
                worker.send_value(r, local)

        return DTensor(f, r, sharding, False)

    def __repr__(self):
       return f"DTensor(sharding={self.sharding}, shape={list(self.shape)})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        return dtensor_dispatch(func, args, kwargs)

    def to_local(self):
        r = reconstruct_tensor(self)
        return Thunk(lambda: r())

    def set_fake_is_first_worker(self, b):
        if self._fake_is_first_worker == b:
            return
        stats.mode_change += 1
        if b:
            assert not any(isinstance(s, int) for s in self.sharding.sharding)
            req =  self.sharding.mesh.flat_workers[0].request_value(self._ref)
            self.sharding.mesh.flat_workers[0].del_value(self._ref)
            self._fake = req.wait()
        else:
            v = self._fake
            self._fake = fake_mode.from_tensor(v)
            self.sharding.mesh.flat_workers[0].send_value(self._ref, v)
        self._fake_is_first_worker = b

    def __del__(self):
        if sys is None:
            return # python shutdown
        if self._sharding is None:
            return # an error happening during construction and this wasn't initialized
        workers = self._sharding.mesh.flat_workers[1:] if self._fake_is_first_worker else self._sharding.mesh.flat_workers
        for worker in workers:
            worker.del_value(self._ref)

    @property
    def sharding(self):
        return self._sharding

    def to_sharding_(self, new_sharding):
        if not isinstance(new_sharding, Sharding):
            new_sharding = Sharding(self.sharding.mesh, new_sharding)
        new_sharding.apply_inplace(self)
        return self

def psum_(tensors, meshdim=0):
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]
    if not tensors:
        return
    for t in tensors:
        t.set_fake_is_first_worker(False)
    mesh = tensors[0]._sharding.mesh
    for i, d in enumerate(tensors):
        if d._sharding.mesh is not mesh:
            raise NotImplementedError("operation on tensors distributed on different device meshes")
        annotations = d.sharding.sharding.copy()
        if annotations[meshdim] != 'b':
            raise ValueError("tensor {i} not batched along {meshdim}")
        annotations[meshdim] = 'r'
        d._sharding = Sharding(mesh, annotations)

    pg = mesh._process_group(meshdim)
    refs = [t._ref.id for t in tensors]
    for worker in mesh.flat_workers:
        worker.all_reduce_coalesced(refs, pg)



def reconstruct_tensor(dtensor):
    annotations, mesh = dtensor._sharding.sharding, dtensor._sharding.mesh
    dims = []
    mesh_indexing = []
    for a in annotations:
        if a == 'r':
            mesh_indexing.append(0)
        elif isinstance(a, int) or a == '+' or a == 'b':
            mesh_indexing.append(slice(None))
            dims.append(a)
        else:
            raise NotImplementedError(f"Annotation {a}")


    mesh = WorkerMesh(mesh._workers, mesh.shape[tuple(mesh_indexing)])
    futures = [worker.request_value(dtensor._ref) if i > 0 or not dtensor._fake_is_first_worker else dtensor._fake for i, worker in enumerate(mesh.flat_workers)]
    first_real_dim = len(dims)

    def reconstruct():
        nonlocal first_real_dim
        local_values = [(f.wait() if isinstance(f, Future) else f).cpu() for f in futures]
        local_value = torch.stack(local_values)
        reshaped = (*mesh.shape.size(), *local_value.size()[1:])
        local_value = torch.reshape(local_value, reshaped)
        for i, d  in enumerate(dims):
            if d == '+': # efficiency wise it is better to do all_reduce first, then transfer from 1
                         # but this is useful for debugging
                local_value = local_value.sum(dim=0)
            elif d == 'b':
                local_value = local_value.movedim(0, first_real_dim - 1)
                local_value = local_value.flatten(first_real_dim - 1, first_real_dim)
            else:
                adjusted_dim = first_real_dim + d
                local_value = local_value.movedim(0, adjusted_dim - 1)
                local_value = local_value.flatten(adjusted_dim - 1, adjusted_dim)
            first_real_dim -= 1

        return local_value
    return reconstruct

def is_dtensor(x):
    return isinstance(x, DTensor)

# tree_flatten but also take a condition about which non-tree values we care about
def flatten(tree, cond=is_dtensor):
    r, spec = tree_flatten(tree)
    def unflatten(n):
        n_it = iter(n)
        return tree_unflatten([next(n_it) if cond(e) else e for e in r], spec)
    return [e for e in r if cond(e)], unflatten

class Manager:
    def __init__(self):
        pass

    def Worker(self, device):
        return ThreadWorker(self, device)

    def create_workers(self, shape) -> 'WorkerMesh':
        N = 1
        if isinstance(shape, int):
            shape = (shape,)
        for s in shape:
            N *= s
        device_count = torch.cuda.device_count()
        return WorkerMesh([self.Worker(device=i % device_count) for i in range(N)]).reshape(*shape)

_thunks = []

def wait_pending_callbacks():
    global _thunks
    for thunk in _thunks:
        thunk.wait()
    _thunks.clear()

class Thunk:
    def __init__(self, fn):
        self.fn = fn
        self.callbacks = []
        _thunks.append(self)

    def wait(self):
        if hasattr(self, 'value'):
            return self.value
        self.value = self.fn()
        for cb in self.callbacks:
            cb(self.value)
        return self.value
    def then(self, cb):
        if hasattr(self, 'value'):
            cb(self.value)
        else:
            self.callbacks.append(cb)


def to_local(obj):
    flat, unflatten = flatten(obj)
    reconstructs = [reconstruct_tensor(f) for f in flat]
    def fn():
        return unflatten([r() for r in reconstructs])
    return Thunk(fn)

class Future:
    def __init__(self):
        self.queue = Queue(maxsize=1)
    def set_value(self, v):
        self.queue.put_nowait((True, v))
    def set_exception(self, e):
        self.queue.put_nowait((False, e))
    def wait(self):
        if not hasattr(self, 'success'):
            self.success, self.value = self.queue.get()
        if not self.success:
            raise self.value
        return self.value

class Worker:
    pass


class ProcessGroup:
    def __init__(self, N):
        self.mailbox = [None]*N
        self.barrier = threading.Barrier(N)

_pg_map = {}

class ThreadWorker(Worker):
    def __init__(self, manager, device):
        self.manager = manager
        self.device = device
        self.queue = Queue()
        self.ref_to_tensor = {}
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while True:
            with torch.cuda.device(self.device):
                command, *args = self.queue.get()
                if command == 'send_command':
                    func, args, kwargs, results = args
                    def get_tensor(t):
                        if isinstance(t, RemoteRef):
                            return self.ref_to_tensor[t.id]
                        else:
                            return t
                    args = tree_map(get_tensor, args)
                    kwargs = tree_map(get_tensor, kwargs)
                    with active_sharding(None):
                        result = func(*args, **kwargs)
                    # list of references picks out the tensors from the result
                    flat_results, _ = tree_flatten(result)
                    real_results = [e for e in flat_results if isinstance(e, torch.Tensor)]
                    for real, ref in zip(real_results, results):
                        self.ref_to_tensor[ref.id] = real
                elif command == 'send_value':
                    ref, value = args
                    self.ref_to_tensor[ref.id] = value # copy?
                elif command == 'del_value':
                    ref, = args
                    del self.ref_to_tensor[ref.id]
                elif command == 'request_value':
                    ref, f = args
                    f.set_value(self.ref_to_tensor[ref.id])
                elif command == 'create_process_group':
                    rank, pg, pg_ref = args
                    self.ref_to_tensor[pg_ref.id] = (rank, pg)
                elif command == 'all_reduce_coalesced':
                    refs, pg_ref = args
                    rank, pg = self.ref_to_tensor[pg_ref.id]
                    for ref in refs:
                        pg.barrier.wait()
                        pg.mailbox[rank] = self.ref_to_tensor[ref].to('cuda:0')
                        pg.barrier.wait()
                        if rank == 0:
                            pg.mailbox[0] = torch.stack(pg.mailbox).sum(dim=0)
                        pg.barrier.wait()
                        self.ref_to_tensor[ref].copy_(pg.mailbox[0])
                else:
                    raise ValueError(f"unknown command {command}")

    def _send(self, *args):
        self.queue.put(args)

    def send_command(self, func, args, kwargs, results):
        self._send('send_command', func, args, kwargs, results)

    def request_value(self, ref: RemoteRef):
        f = Future()
        self._send('request_value', ref, f)
        return f

    def send_value(self, ref: RemoteRef, value: torch.Tensor):
        self._send('send_value', ref, value)

    def del_value(self, ref: RemoteRef):
        self._send('del_value', ref)

    def create_process_group(self, rank, pg, pg_ref):
        self._send('create_process_group', rank, pg, pg_ref)

    def all_reduce_coalesced(self, refs: List[int], pg_ref: RemoteRef):
        self._send('all_reduce_coalesced', refs, pg_ref)







class WorkerList:
    def __init__(self, workers: 'List[Worker]'):
        self.workers = workers
        self.pg = None

    @property
    def process_group(self):
        if self.pg is None:
            self.pg = RemoteRef()
            pg_obj = ProcessGroup(len(self.workers))
            for rank, worker in enumerate(self.workers):
                worker.create_process_group(rank, pg_obj, self.pg)
        return self.pg



class WorkerMesh:
    """
    A multi-dimensional array of devices used to specify shardings.
    Similar to the collective DTensor we have today
    """
    def __init__(self, workers: List[Worker], shape=None):
        if not isinstance(workers, WorkerList):
            workers = WorkerList(workers)
        self._workers = workers
        self.shape = torch.arange(len(workers.workers)) if shape is None else shape
        self.dim_to_pg = {}
        self.flat_workers =  [self._workers.workers[idx.item()] for idx in self.shape.flatten()]

    def reshape(self, *dims):
        return WorkerMesh(self._workers, self.shape.reshape(*dims))

    def __getitem__(self, elem):
        wm = WorkerMesh(self._workers, self.shape[elem])
        if wm.shape.dim() == 0:
            wm = wm.Sharding()
        return wm

    def select(self, dim, index):
        return WorkerMesh(self._workers, self.shape[index])

    def __repr__(self):
        return f'Mesh<{"x".join(str(s) for s in self.shape.size())}>'

    def _process_group(self, dim: int):
        def create_subgroup(pg, ranks):
            pg_object = ProcessGroup(len(ranks))
            for rank in ranks:
                w = self._workers.workers[rank]
                w.create_process_group(rank, pg_object, pg)

        if dim not in self.dim_to_pg:
            if self.shape.dim() == 1:
                assert dim == 0
                if len(self._workers.workers) == len(self.flat_workers):
                    self.dim_to_pg[dim] = self._workers.process_group
                else:
                    pg = RemoteRef()
                    ranks = [s.item() for s in self.shape]
                    create_subgroup(pg, ranks)
                    self.dim_to_pg[dim] =  pg
            else:
                pg = self.dim_to_pg[dim] = RemoteRef()
                flat_shape = self.shape.movedim(dim, -1).flatten(0, -2)
                for subgroup in flat_shape:
                    ranks = [s.item() for item in subgroup]
                    create_subgroup(pg, ranks)

        return self.dim_to_pg[dim]

    def Sharding(self, *annotations):
        return Sharding(self, list(annotations))

class Sharding:
    """
    A description of how a single tensor is sharded across devices.
    This is equivalent to our collective dtensor
    """
    mesh: WorkerMesh
    # one entry per dimension of device mesh,
    # which specifies how the tensor is represented
    # in that dimension
    sharding: List[Union[int, # shareded - e.g. the 0th dimension of
                              #   of the tensor is split across this dimesion
                              #   of the device mesh
                 Literal["r", # replicated - a copy of the data is stored
                              #   across this dimension of the device mesh
                         "b", # implicitly batched - each worker has a different copy
                              # of data along this dimension of the device mesh
                              # but it acts an an implicitly batched dimension for the
                              # purpose of operactors acting on the tensor
                         "+"  # partial sum - the value of the tensor is
                              #   the sum of the tensors stored along
                              #   this dimension of the device mesh
                         ]]]
    # note: it is also equivlent to specify a sharding
    # by saying if each dimension of the tensor is sharded and which
    # device mesh dimension it correspond to, but it still
    # requires saying whether the remaining device mesh dimensions are
    # replicated or sharded.

    def __init__(self, mesh, sharding):
        self.mesh = mesh
        self.sharding = sharding
        if isinstance(self.sharding, (str, int)):
            self.sharding = [self.sharding]
        assert len(self.sharding) == self.mesh.shape.dim(), "Mismatched sharding annotations to device mesh"

    def __repr__(self):
        return f"{self.sharding}"

    # for hashing
    @property
    def _key(self):
        return self.mesh, tuple(self.sharding)

    @staticmethod
    def lift(obj):
        if isinstance(obj, Worker):
            return Sharding._singleton_mesh(obj)
        elif isinstance(obj, Sharding):
            return obj
        else:
            raise ValueError(f"expected Sharding: {obj}")

    @staticmethod
    @cache
    def _singleton_mesh(worker):
        mesh = WorkerMesh([worker]).reshape(())
        return Sharding(mesh=mesh, sharding = [])

    @property
    def manager(self):
        return self.mesh.flat_workers[0].manager

    def change(self, mesh=None, sharding=None):
        return Sharding(mesh=mesh if mesh else self.mesh, sharding=sharding if sharding else self.sharding)

    def DTensor(self, t):
        return DTensor.to_remote(t, self)

    def apply_inplace(self, *tensors):
        reduce_pg_to_args = {}
        for t in tensors:
            if t.sharding.mesh is not self.mesh:
                raise NotImplementedError(f"Cross mesh transfer {t.sharding.mesh} is not {self.mesh}")
            for i, (olda, newa) in enumerate(zip(t.sharding.sharding, self.sharding)):
                if olda == newa:
                    continue
                if newa == '+':
                    raise ValueError(f"unexpected introduction of partial sum {t.sharding.sharding} -> {self.sharding}")
                if olda == '+' and newa == 'r':
                    # all reduce
                    pg = self.mesh._process_group(i)
                    if pg not in reduce_pg_to_args:
                        reduce_pg_to_args[pg] = []
                    reduce_pg_to_args[pg].append(t._ref.id)
                elif olda == '+' and isinstance(newa, int):
                    # reduce scatter
                    raise NotImplementedError("Reduce scatter")
                elif isinstance(olda, int) and newa == 'r':
                    raise NotImplementedError("all gather")
                elif olda == "r" and isinstance(newa, int):
                    raise NotImplementedError("drop some value")
            t._sharding = self

        for pg, args in reduce_pg_to_args.items():
            for worker in self.mesh.flat_workers:
                worker.all_reduce_coalesced(args, pg)

        if check_correctness_per_operator:
            for t in tensors:
                rem = t.to_local().wait()
                torch.testing.assert_close(t._fake, rem, atol=1e-03, rtol=1e-03)
                # don't let small differences accumulate over time when correctness testing
                t._fake.copy_(rem)
