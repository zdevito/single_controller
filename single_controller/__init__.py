import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten, tree_unflatten, tree_map
from typing import NamedTuple, List, Union, Literal
from contextlib import contextmanager, nullcontext
from .base_tensor import BaseTensor
from socket import socket, AF_INET, SOCK_STREAM
from uuid import uuid4
from subprocess import Popen, PIPE
import sys
import pickle
import asyncio
import threading
from concurrent.futures import Future
import traceback
import atexit
from functools import cache
import os
from torch._subclasses.fake_tensor import FakeTensorMode
import torch._dynamo
from torch._dynamo.backends.common import aot_autograd
from functorch.compile import min_cut_rematerialization_partition
import tempfile
import warnings

check_correctness_per_operator = False
simulate_function_calls = False
_py_compile = compile

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


@cache
def sharding_to_find(func, *args):
    # print("FAKE", str(func), func._schema)
    pass

def propagate_sharding(func, args, kwargs, dtensor_input: 'List[DTensor]', dtensor_results: 'List[DTensor]', sharding_context: 'Sharding'):
    key = str(func)
    mesh = dtensor_input[0]._sharding.mesh if dtensor_input else sharding_context.mesh
    for d in dtensor_input:
        if d._sharding.mesh is not mesh:
            raise NotImplementedError("operation on tensors distributed on different device meshes")

    if key in prules:
        args, kwargs = prules[key](args, kwargs, dtensor_input=dtensor_input, dtensor_results=dtensor_results, sharding_context=sharding_context)
        return mesh, args, kwargs

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

    return mesh, args, kwargs

_trace = None

def dtensor_dispatch(func, args=(), kwargs=None, sharding=None):
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

    with fake_mode:
        fake_input_tensors =  [d._fake for d in dtensors]
        fake_args, fake_kwargs = unflatten(fake_input_tensors)
        result = func(*fake_args, **fake_kwargs)

    fake_result_dtensors, unflatten_result = flatten(result, is_tensor)
    fake_map = {id(f): i for i, f in enumerate(fake_input_tensors)}

    # sometimes operators return references to inputs, in which case the result should be the same DTensor object
    # otherwise we create a new DTensor with a new RemoteRef for the result
    result_dtensors = [dtensors[fake_map[id(fake)]] if id(fake) in fake_map else DTensor(fake, RemoteRef(), None) for fake in fake_result_dtensors]
    mesh, modified_args, modified_kwargs = propagate_sharding(func, args, kwargs, dtensors, result_dtensors, sharding)

    for i, r in enumerate(result_dtensors):
        assert r._sharding is not None, f"sharding unset for output {i} of {str(func)} fake outputs: {fake_result_dtensors}"

    def get_ref(x):
        return x if not isinstance(x, DTensor) else x._ref

    ref_args, ref_kwargs = tree_map(get_ref, modified_args), tree_map(get_ref, modified_kwargs)
    ref_results = [r._ref for r in result_dtensors]
    if _trace is not None:
        m, cmds = _trace
        assert m is mesh, "multiple compiled mesh NYI"
        cmds.append((func, ref_args, ref_kwargs, ref_results, result))
    else:
        try:
            for worker in mesh.flat_workers:
                worker.send_command(func, ref_args, ref_kwargs, ref_results)
        except:
            print(fake_input_tensors)
            raise
    results = unflatten_result(result_dtensors)

    if check_correctness_per_operator:
        key = str(func)
        print(key, args, kwargs, result_dtensors)
        if "_." in key:
            for i, dtensor in enumerate(dtensors):
                rem = dtensor.to_local().wait()
                # print(dtensor._fake.sum(), rem.sum())
                if torch.all(torch.isfinite(dtensor._fake)) and torch.all(torch.isfinite(rem)):
                    torch.testing.assert_close(dtensor._fake, rem, atol=1e-03, rtol=1e-03)
                else:
                    pass #print("nonfinite present...")
                # don't let small differences accumulate over time when correctness testing
                try:
                    dtensor._fake.copy_(rem)
                except RuntimeError as e:
                    assert "unsupported operation" in str(e), "Weird tensor shapes cannot be moved around"
        for i, dtensor in enumerate(result_dtensors):
            rem = dtensor.to_local().wait()
            if 'aten._scaled_dot_product_efficient_attention.default' == key and i > 1:
                break
            # print(dtensor._fake.sum(), rem.sum())
            if torch.all(torch.isfinite(dtensor._fake)) and torch.all(torch.isfinite(rem)):
                torch.testing.assert_close(dtensor._fake, rem, atol=1e-03, rtol=1e-03)
            else:
                pass #print("nonfinite present...")
            # don't let small differences accumulate over time when correctness testing
            try:
                dtensor._fake.copy_(rem)
            except RuntimeError as e:
                assert "unsupported operation" in str(e), "Weird tensor shapes cannot be moved around"

    return results



class FakeTLS:
    pass
# thread local doesn't quite work right because backwards pass runs in another thread
# but we don't move the resource guard over there
tls = FakeTLS() # thread.local()

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

class PicklableFunc:
    def __init__(self, callable):
        if isinstance(callable, str):
            first, *parts = callable.split('.')
            callable = globals()[first]
            for p in parts:
                callable = getattr(callable, p)
        self.callable = callable
    def __call__(self, *args, **kwargs):
        return self.callable(*args, **kwargs)
    def __reduce__(self):
        if isinstance(self.callable, torch._ops.OpOverload):
            return PicklableFunc, ("torch.ops." + str(self.callable),)
        else:
            return PicklableFunc, (self.callable,)
    def __repr__(self):
        return repr(self.callable)
    def __str__(self):
        return str(self.callable)

class DTensor(BaseTensor):
    @staticmethod
    def __new__(
        cls,
        fake: torch.Tensor,
        ref: RemoteRef,
        sharding: 'Optional[Sharding]',
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
        return r

    def __init__(self, fake, ref, worker):
        pass


    def __tensor_flatten__(self):
        return ['_fake'], (self._ref, self._sharding)

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta):
        return inner_tensors['_fake']

    @classmethod
    def to_remote(cls, t: torch.Tensor, sharding: 'Sharding'):
        sharding = Sharding.lift(sharding)
        f = fake_mode.from_tensor(t)
        r = RemoteRef()

        result = DTensor(f, r, sharding)
        shape = sharding.mesh.shape

        for i, ann in enumerate(sharding.sharding):
            if ann == "r":
                t = t.expand(shape.size(i), *t.size())
                t = t.movedim(0, i)
            elif isinstance(ann, int):
                sizes = t.size()
                ann_adjusted = ann + i
                d = sizes[ann_adjusted]
                assert d % shape.size(i) == 0, "NOT EVENLY SPLIT"
                t = t.reshape(*sizes[:ann_adjusted], shape.size(i), d // shape.size(i), *sizes[ann_adjusted+1:])
                t = t.movedim(ann_adjusted, i)
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

        return result

    def __repr__(self):
       return f"DTensor(sharding={self.sharding}, shape={list(self.shape)})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        return dtensor_dispatch(func, args, kwargs)

    def to_local(self):
        return self._sharding.manager.schedule(reconstruct_tensor(self))

    def __del__(self):
        if sys is None or _trace is not None:
            return # pytho shutdown
        if self._sharding is None:
            return # an error happening during construction and this wasn't initialized
        for worker in self._sharding.mesh.flat_workers:
            worker.del_value(self._ref)

    @property
    def sharding(self):
        return self._sharding

    def to_sharding_(self, new_sharding):
        if not isinstance(new_sharding, Sharding):
            new_sharding = Sharding(self.sharding.mesh, new_sharding)
        if self.sharding.mesh is not new_sharding.mesh:
            raise NotImplementedError(f"Cross mesh transfer {self.sharding.mesh} is not {new_sharding.mesh}")
        for i, (olda, newa) in enumerate(zip(self.sharding.sharding, new_sharding.sharding)):
            if olda == newa:
                continue
            if newa == '+':
                raise ValueError(f"unexpected introduction of partial sum {self.sharding.sharding} -> {new_sharding.sharding}")
            if olda == '+' and newa == 'r':
                # all reduce
                pg = self.sharding.mesh._process_group(i)
                for worker in self.sharding.mesh.flat_workers:
                    worker.all_reduce(self._ref, pg)
            elif olda == '+' and isinstance(newa, int):
                # reduce scatter
                raise NotImplementedError("Reduce scatter")
            elif isinstance(olda, int) and newa == 'r':
                raise NotImplementedError("all gather")
            elif olda == "r" and isinstance(newa, int):
                raise NotImplementedError("drop some value")

        self._sharding = new_sharding
        if check_correctness_per_operator:
            rem = self.to_local().wait()
            torch.testing.assert_close(self._fake, rem, atol=1e-03, rtol=1e-03)
            # don't let small differences accumulate over time when correctness testing
            self._fake.copy_(rem)

        return self



def reconstruct_tensor(dtensor):
    annotations, mesh = dtensor._sharding.sharding, dtensor._sharding.mesh
    dims = []
    mesh_indexing = []
    for a in annotations:
        if a == 'r':
            mesh_indexing.append(0)
        elif isinstance(a, int) or a == '+':
            mesh_indexing.append(slice(None))
            dims.append(a)
        else:
            raise NotImplementedError(f"Annotation {a}")
    mesh = mesh[tuple(mesh_indexing)]
    futures = [worker.request_value(dtensor._ref) for worker in mesh.flat_workers]

    async def reconstruct():
        local_values = [await f for f in futures]
        local_value = torch.stack(local_values)
        reshaped = (*mesh.shape.size(), *local_value.size()[1:])
        local_value = torch.reshape(local_value, reshaped)
        for i, d  in enumerate(dims):
            if d == '+': # efficiency wise it is better to do all_reduce first, then transfer from 1
                         # but this is useful for debugging
                local_value = local_value.sum(dim=0)
            else:
                adjusted_dim = d + (len(dims) - i - 1)
                local_value = local_value.movedim(0, adjusted_dim)
                local_value = local_value.flatten(adjusted_dim, adjusted_dim + 1)
        return local_value
    return reconstruct()

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
        self.loop = asyncio.get_event_loop()
        self.workers = {} # uuid -> Worker

    def _start_loop(self):
        if hasattr(self, 'thread'):
            return
        self.host = "127.0.0.1"
        self.port = 12345
        self.shutdown_event = asyncio.Event()
        self.thread = threading.Thread(target=lambda: self.loop.run_until_complete(self.shutdown_event.wait()), daemon=True)
        self.thread.start()
        atexit.register(self.complete)
        self.schedule_void(self.accept_new_connections())
        self.server_started = asyncio.Event()

    async def accept_new_connections(self):
        # TODO: goto next port?
        server = await asyncio.start_server(self.new_connection, self.host, self.port)
        async with server:
            self.server_started.set()
            await server.serve_forever()

    async def new_connection(self, reader, writer):
        sz = int.from_bytes(await reader.read(8), 'little')
        secret = pickle.loads(await reader.read(sz))
        worker = self.workers[secret]
        await worker.connect(reader, writer)

    def create_worker(self, devices=None, local=False):
        secret = str(uuid4())
        if local:
            if devices is not None:
                warnings.warn("devices are ignored for local processes")
            self.workers[secret] = result = LocalWorker(self)
            return result
        env = os.environ.copy()
        if devices is not None:
            if isinstance(devices, int):
                devices = [devices]
            env['CUDA_VISIBLE_DEVICES'] = ','.join(str(d) for d in devices)


        self._start_loop()
        # we need to wait for the server to start serving
        # otherwise the new process will not be able to connect
        f = self.schedule(self.server_started.wait())
        f.result()

        # if we were only ever going to use local processes then it would easier to just directly
        # create pipes, but this uses sockets so that we can add remote processes without changing the
        # setup.
        self.workers[secret] = result = Worker(self)
        result.proc = Popen([sys.executable, '-m', 'single_controller.worker_process', self.host, str(self.port), secret], env=env)
        return result

    def schedule_void(self, co):
        async def wrap():
            try:
                await co
            except:
                traceback.print_exc()
        self.schedule(wrap())

    def schedule(self, awaitable):
        if hasattr(self, 'thread'):
            return asyncio.run_coroutine_threadsafe(awaitable, self.loop)
        else:
            # if we only have local workers
            r = self.loop.run_until_complete(awaitable)
            f = Future()
            f.set_result(r)
            return f


    def complete(self):
        if _last_event is not None:
            _last_event.wait()
        async def shutdown():
            for worker in self.workers.values():
                await worker.request_exit()
            self.shutdown_event.set()
        self.schedule_void(shutdown())
        for worker in self.workers.values():
            worker.wait()

# monkey patch...
_last_event = None
def _then(self, cb):
    global _last_event
    # track when the last callback that was registered ran
    _last_event = this_event = threading.Event()
    def run(x):
        cb(x.result())
        this_event.set()
    self.add_done_callback(run)

def _debug_wait_pending_callbacks():
    if _last_event is not None:
            _last_event.wait()

Future.then = _then
Future.wait = Future.result

def to_local(obj):
    flat, unflatten = flatten(obj)
    reconstructs = [reconstruct_tensor(f) for f in flat]
    manager = flat[0]._sharding.manager
    async def wrap():
        local_tensors = await asyncio.gather(*reconstructs)
        return unflatten(local_tensors)
    return manager.schedule(wrap())

class Worker:
    def __init__(self, manager):
        self.manager = manager
        self._dumps = pickle.dumps
        self._loads = pickle.loads
        self.new_commands = asyncio.Condition()
        self.commands_to_send = []
        self.pending_futures = []

    async def connect(self, r, w):
        await asyncio.gather(self.reader(r), self.writer(w))

    async def reader(self, r):
        while True:
            b = await r.read(8)
            sz = int.from_bytes(b, 'little')
            obj = self._loads(await r.readexactly(sz))
            f = self.pending_futures.pop(0)
            f.set_result(obj)

    async def writer(self, w):
        while True:
            try:
                b = self.commands_to_send.pop(0)
                w.write(len(b).to_bytes(8, 'little'))
                w.write(b)
            except IndexError:
                await w.drain()
                async with self.new_commands:
                    await self.new_commands.wait()

    def send_command(self, func, args, kwargs, results):
        self._write_pickle(('send_command', PicklableFunc(func), args, kwargs, results))

    def send_value(self, ref: RemoteRef, value: torch.Tensor):
        self._write_pickle(('send_value', ref, value))

    def del_value(self, ref: RemoteRef):
        self._write_pickle(('del_value', ref))

    def create_process_group(self, rank, worldsize, pg_ref):
        self._write_pickle(('create_process_group', rank, worldsize, pg_ref))

    def create_process_subgroup(self, orig_pg, participating_ranks, pg):
        self._write_pickle(('create_process_subgroup', orig_pg, participating_ranks, pg))

    def all_reduce(self, ref: RemoteRef, pg_ref: RemoteRef):
        self._write_pickle(('all_reduce', ref, pg_ref))

    def _write_pickle(self, obj, future=None):
        b = self._dumps(obj)
        self.commands_to_send.append(b)
        self.manager.schedule_void(self._notify_new_commands())

    async def _notify_new_commands(self):
        async with self.new_commands:
            self.new_commands.notify()

    def request_value(self, request: RemoteRef) -> 'asyncio.Future[torch.Tensor]':
        f = self.manager.loop.create_future()
        self.pending_futures.append(f)
        self.commands_to_send.append(self._dumps(('request_value', request)))
        self.manager.schedule_void(self._notify_new_commands())
        return f

    def wait(self):
        self.proc.wait()

    def request_exit(self):
        b = pickle.dumps(('exit',))
        self.commands_to_send.append(b)
        return self._notify_new_commands()

    def define_function(self, fn, code):
        self._write_pickle(('define_function', fn, code))

    def run_function(self, fn, arguments, results):
        self._write_pickle(('run_function', fn, arguments, results))

class LocalProcessGroup(NamedTuple):
    members: 'List[LocalWorker]'
    values: 'List[torch.tensor]'

_local_process_group_being_created = None


class BaseWorker:
    def define_function(self, fn, code):
        with tempfile.NamedTemporaryFile(delete=False, mode='w') as temp:
            temp.write(code)

        code = _py_compile(code, temp.name, 'exec')
        G = {'torch': torch, 'device': torch.device}
        exec(code, G)
        self.ref_to_tensor[fn.id] = G['run']

    def run_function(self, fn, args, result_refs):
        results = self.ref_to_tensor[fn.id](*(self.ref_to_tensor[a.id] for a in args))
        for r, ref in zip(results, result_refs):
            self.ref_to_tensor[ref.id] = r

class LocalWorker(BaseWorker):
    """
    Run in the local process rather than remotely
    """
    def __init__(self, manager):
        self.manager = manager
        self.ref_to_tensor = {}
    def send_command(self, func, args, kwargs, results):
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

    def run_function(self, fn, args, result_refs):
        with active_sharding(None):
            return super().run_function(fn, args, result_refs)

    def request_value(self, ref: RemoteRef):
        f = self.manager.loop.create_future()
        f.set_result(self.ref_to_tensor[ref.id])
        return f

    def send_value(self, ref: RemoteRef, value: torch.Tensor):
        self.ref_to_tensor[ref.id] = value

    def del_value(self, ref: RemoteRef):
        del self.ref_to_tensor[ref.id]

    def request_exit(self):
        async def noop():
            pass
        return noop()

    def create_process_group(self, rank, worldsize, pg_ref):
        global _local_process_group_being_created
        if _local_process_group_being_created is None:
            _local_process_group_being_created = LocalProcessGroup([], [])
        self.ref_to_tensor[pg_ref.id] = _local_process_group_being_created
        _local_process_group_being_created.members.append(self)
        if len(_local_process_group_being_created.members) == worldsize:
            _local_process_group_being_created = None

    def create_process_subgroup(self, orig_pg, participating_ranks, pg):
        if pg is None:
            return
        global _local_process_group_being_created
        if _local_process_group_being_created is None:
            _local_process_group_being_created = LocalProcessGroup([], [])
        self.ref_to_tensor[pg_ref.id] = _local_process_group_being_created
        _local_process_group_being_created.members.append(self)
        if len(_local_process_group_being_created.members) == len(participating_ranks):
            _local_process_group_being_created = None

    def all_reduce(self, ref: RemoteRef, pg_ref: RemoteRef):
        pg = self.ref_to_tensor[pg_ref.id]
        t = self.ref_to_tensor[ref.id]
        pg.values.append(t)
        if len(pg.values) == len(pg.members):
            r = torch.stack(pg.values).sum(dim=0)
            for m in pg.members:
                m.ref_to_tensor[ref.id] = r.clone()
            pg.values.clear()

    def wait(self):
        pass





class WorkerList:
    def __init__(self, workers: List[Worker]):
        self.workers = workers
        self.pg = None

    @property
    def process_group(self):
        if self.pg is None:
            self.pg = RemoteRef()
            for rank, worker in enumerate(self.workers):
                worker.create_process_group(rank, len(self.workers), self.pg)
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

    @property
    def flat_workers(self):
        return [self._workers.workers[idx.item()] for idx in self.shape.flatten()]

    def reshape(self, *dims):
        return WorkerMesh(self._workers, self.shape.reshape(*dims))

    def __getitem__(self, elem):
        return WorkerMesh(self._workers, self.shape[elem])

    def select(self, dim, index):
        return WorkerMesh(self._workers, self.shape[index])

    def __repr__(self):
        return f'Mesh<{"x".join(str(s) for s in self.shape.size())}>'

    def _process_group(self, dim: int):
        if dim not in self.dim_to_pg:
            if self.shape.dim() == 1:
                assert dim == 0
                self.dim_to_pg[dim] = self._workers.process_group
            else:
                pg = self.dim_to_pg[dim] = RemoteRef()
                flat_shape = self.shape.movedim(dim, -1).flatten(0, -2)
                for subgroup in flat_shape:
                    ranks = [s.item() for item in subgroup]
                    for i, w in enumerate(self._workers.workers):
                        w.create_process_subgroup(self._workers.process_group, ranks, pg if i in ranks else None)
        return self.dim_to_pg[dim]

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

    def __repr__(self):
        return f"{self.sharding}"

    # for hashing
    @property
    def _key(self):
        return self.mesh, tuple(self.sharding)

    @staticmethod
    def lift(obj):
        if isinstance(obj, (LocalWorker,Worker)):
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

    def from_local(self, t):
        return DTensor.to_remote(t, self)


# super janky "compile" to batch commands
# TODO: we don't support sending different commands to different workers yet
def _compile(graph, _):
    sharding_cache = {}
    print(graph.code)
    def wrapper(inputs):
        # ignore the function call and just simulate how the gradients were split
        if simulate_function_calls:
            return graph(*inputs)
        key = (*(a.sharding._key for a in inputs), _current_active_sharding())
        if key not in sharding_cache:
            global _trace
            mesh = inputs[0]._sharding.mesh
            commands = []
            _trace = (mesh, commands)
            formals = [DTensor(i._fake, RemoteRef(), i._sharding) for i in inputs]

            # check if something is causing the active_sharding resource guard to get lost here,
            # HACK is to force re-enabled. However a better solution would ensure that the arange and other things
            # just get captured with the sharding as a direct argument or something similar, because in real code
            # we might have captured different constructors with different active shardings...
            outputs_reference = graph(*formals)

            # delete these DTensor before tracing is over so that their __del__ doesn't send a message.
            formals = [f._ref for f in formals]
            _trace = None

            lines = []
            lines.append(f"def run({','.join(repr(f) for f in formals)}):")
            for func, args, kwargs, tensor_results, fake_results in commands:
                arg_list = [*(repr(a) for a in args), *(f'{name}={repr(value)}' for name, value in kwargs.items())]
                arg_strings = ', '.join(arg_list)
                lhs = [repr(r) for r in tensor_results]
                if isinstance(fake_results, (list, tuple)):
                    lhs_it = iter(lhs)
                    lhs = [next(lhs_it) if isinstance(fr, torch.Tensor) else "_" for fr in fake_results]

                lines.append(f"    {', '.join(lhs)} = torch.ops.{str(func)}({arg_strings})")
            lines.append(f"    return {','.join(repr(o._ref) for o in outputs_reference if o is not None)},") # outputs can be non-tensor?
            lines.append("")

            code = '\n'.join(lines)
            fn = RemoteRef()
            for worker in mesh.flat_workers:
                worker.define_function(fn, code)

            def impl(actuals):
                outputs = [DTensor(o._fake.clone(), RemoteRef(), o.sharding) if o is not None else None for o in outputs_reference]
                for worker in mesh.flat_workers:
                    worker.run_function(fn, [a._ref for a in actuals], [o._ref for o in outputs if o is not None])
                return outputs
            sharding_cache[key] = impl
        return sharding_cache[key](inputs)
    wrapper._boxed_call = True
    return wrapper

aot_dtensor = aot_autograd(fw_compiler=_compile, partition_fn=min_cut_rematerialization_partition)



def compile(*args, **kwargs):
    return torch.compile(*args, backend=aot_dtensor, **kwargs)
