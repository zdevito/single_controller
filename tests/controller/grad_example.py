import torch
import torch._inductor.metrics as metrics
from collections import defaultdict
import itertools
torch.set_default_device("cuda")

# import torch
from controller import (
    DeviceMesh,
    active_mesh,
    active_stream,
    Stream,
    fetch_shard,
    Future,
    Stream,
    get_active_stream,
)
from torch.fx.experimental.proxy_tensor import make_fx
from controller._testing import simulator_mesh, example_mesh
from controller.simulator import set_meta


from torch.utils._pytree import tree_map


NUM_GPUS=4

def call_layer(w, x):
    return torch.mm(x, w)

def initialize(num_gpus, stages_per_gpu, hidden_dim=4, local_batch_size=4, use_real=False):
    if use_real:
        device_mesh = example_mesh(hosts=1, gpus=num_gpus)
    else:
        device_mesh = simulator_mesh(hosts=1, gpus=num_gpus)
    meshes = [device_mesh(gpu=i) for i in range(num_gpus)]
    layers = [[] for _ in range(num_gpus)]
    for idx, mesh in enumerate(meshes):
        with active_mesh(mesh):
            for _ in range(stages_per_gpu):
                layers[idx].append(torch.randn(hidden_dim, hidden_dim, requires_grad=True))

    def dataloader():
        with active_mesh(meshes[0]):
            # a great dataloader
            return torch.randn(local_batch_size, hidden_dim)

    return device_mesh, meshes, layers, dataloader

from controller.gradient_generator import grad_generator, grad_function

@grad_function
def to_mesh(x: torch.Tensor, mesh: DeviceMesh):
    omesh = x.mesh
    def backward(grad_x: torch.Tensor):
        print(grad_x.mesh, omesh)
        return grad_x.to_mesh(omesh), None
    return x.to_mesh(mesh), backward

# coroutine that runs each microbatch, yields after each stage,
# yield True to indicate phases are over.
# might be better to have it yield more describitive info each time.
def run_mb(mb_num, layers, dataloader):
    mb = dataloader() # loads the microbatch
    grads = []
    for layer in layers:
        if mb.mesh != layer.mesh:
            grads.append(torch.autograd.graph.get_gradient_edge(mb))
            mb = to_mesh(mb, layer.mesh)
            yield
        with active_mesh(layer.mesh), set_meta('fw'), set_meta(str(mb_num)):
            # grad weight stage
            grads.append(layer)
            if mb.requires_grad:
                # grad input stage
                grads.append(torch.autograd.graph.get_gradient_edge(mb))
            mb = call_layer(layer, mb)
        yield

    yield True

    # could be a loop, but set_meta has to go around
    # the generator step so backward calc is recorded.
    gen = grad_generator(mb, list(reversed(grads)))
    while True:
        try:
            with set_meta('bw'), set_meta(str(mb_num)):
                next(gen)
            yield
        except StopIteration:
            break
    yield True

def fill_drain(num_microbatches, layers, dataloader):
    mbs = [run_mb(i, layers, dataloader) for i in range(num_microbatches)]
    for fb in range(2):
        for mb in mbs:
            while not next(mb):
                pass

def fill_drain_interleaved(num_microbatches, layers, dataloader):
    mbs = [run_mb(i, layers, dataloader) for i in range(num_microbatches)]
    for fb in range(2):
        finished = set() 
        for r in itertools.count(0):
            if len(finished) == len(mbs):
                break
            for _ in range(2 + fb):
                for i, mb in enumerate(mbs):
                    # introduce the initial skew.
                    # it is a skew of 2 for foward (step, send)
                    # and 3 for backward (gradi, gradw, send)
                    if r < i or i in finished:
                        continue
                    f = next(mb)
                    if f:
                        finished.add(i)
            

from controller.simulator import simulate_commands, chrome_events, visualize_events, analyze_events
device_mesh, meshes, layers, dataloader = initialize(num_gpus=4, stages_per_gpu=2, local_batch_size=32)

interleave = True
if interleave:
    stages_per_layer = len(layers[0])
    layers = [stage[i] for i in range(stages_per_layer) for stage in layers]
    fill_drain_interleaved(num_microbatches=4, layers=layers, dataloader=dataloader)
else:
    fill_drain(num_microbatches=4, layers=[l for s in layers for l in s], dataloader=dataloader)
events = simulate_commands(device_mesh.ctrl.backend.worker_commands)
visualize_events(events)
analyze_events(events)
