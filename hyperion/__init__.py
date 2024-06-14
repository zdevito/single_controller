import torch
from controller.simulator import set_meta

class Continuation(object):
    def __init__(self, continuation, values):
        self.continuation = continuation
        self.values = values
    
    def __call__(self, *args):
        return self.continuation(*args)

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
        for x in fw_inps:
            x.requires_grad_(True)
        diff_ins = fw_inps
        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            with set_meta("fw"):
                fw_outs = f(*fw_inps)
            
        def bw_call(*grad_outs):
            with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
                with set_meta("bw"):
                    return torch.autograd.grad(outputs = fw_outs, inputs=fw_inps, grad_outputs=grad_outs)
    
        return fw_outs, Continuation(continuation=bw_call, values=saved_values)
    return wrapped_f
