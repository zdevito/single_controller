from single_controller import DTensor, active_sharding, Manager, LocalWorker, to_local
import torch
m = Manager()
w = m.create_worker(local=False)

x = DTensor.to_remote(torch.ones(2, 2), sharding=w)
xx = x.add(x).add(x).max(dim=0)
f = xx[0].to_local()

assert (torch.allclose(torch.ones(2)*3, f.wait()))

with active_sharding(sharding=w):
    o = torch.ones(2) + x


lo = o.to_local()

lo.then(lambda f: print("CB", f))
assert torch.allclose(lo.wait(), torch.ones(2, 2) * 2)

to_local({'x': x, 'o': o, 'y': x + x + x}).then(print)

