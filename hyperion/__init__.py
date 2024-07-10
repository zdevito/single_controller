import torch
from controller.simulator import set_meta
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
from torch.utils._pytree import tree_map_only, tree_flatten, tree_unflatten
from torch.func import functional_call


class Continuation(object):
    def __init__(self, continuation, values):
        self.continuation = continuation
        self.values = values
    
    def __call__(self, *args):
        return self.continuation(*args)

def explicit_module(mod):
    def mod_call(params, *args):
        return functional_call(mod, params, args)
    return explicit_autograd(mod_call)

def explicit_autograd(f):
    saved_values = {}
    def pack(x):
        idx = len(saved_values)
        saved_values[idx] = x.detach()
        return idx

    def unpack(id):
        val = saved_values[id]
        del saved_values[id]
        return val

    def wrapped_f(*fw_inps):
        tree_map_only(torch.Tensor, lambda x: x.requires_grad_(True), fw_inps)
        flat_inps, inp_spec = tree_flatten(fw_inps)
        diff_ins = flat_inps
        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            with set_meta("fw"):
                fw_outs = f(*fw_inps)
            
        def bw_call(*grad_outs):
            with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
                with set_meta("bw"):
                    grads = torch.autograd.grad(outputs = fw_outs, inputs=flat_inps, grad_outputs=grad_outs)
                    grads = tree_unflatten(grads, inp_spec)
                    return grads
    
        return fw_outs, Continuation(continuation=bw_call, values=saved_values)
    return wrapped_f

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
            return torch.randn(local_batch_size, hidden_dim, requires_grad=True)

    return device_mesh, meshes, layers, dataloader
    
def fill_drain(num_microbatches, meshes, layers, dataloader):
    pp_stages = len(meshes)
    bw_callbacks = {}
    bw_microbatches = []
    for mb_num in range(num_microbatches):
        mb = dataloader() # loads the microbatch
        with set_meta(str(mb_num)):
            for stage in range(0, pp_stages):
                with active_mesh(meshes[stage]):
                    mb = mb.to_mesh(meshes[stage]) # moves the microbatch to the current stage
                    for layer_idx, layer in enumerate(layers[stage]):
                        mb, bw_callback = explicit_autograd(call_layer)(layer, mb) # calls the current stage's layer
                        bw_callbacks[(mb_num, stage, layer_idx)] = bw_callback # saves bw callback for later
        bw_microbatches.append(mb)
        
    for mb_num, mb in enumerate(bw_microbatches):
        with set_meta(str(mb_num)):
            # Iterate from the back to the front
            for stage in range(pp_stages - 1, -1, -1):
                # Get the corresponding bw_callback
                with active_mesh(meshes[stage]):
                    for layer_idx in range(len(layers[stage]) -1, -1, -1):
                        gradWeight, mb = bw_callbacks[(mb_num, stage, layer_idx)](mb)
                        del bw_callbacks[(mb_num, stage, layer_idx)]
                    if stage != 0:
                        mb = mb.to_mesh(meshes[stage - 1])
