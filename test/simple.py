from single_controller import DTensor, active_sharding, Manager, LocalWorker, to_local
import torch
real = True
if real:
    m = Manager()
    w = m.create_worker()
else:
    w = LocalWorker()

x = DTensor.to_remote(torch.ones(2, 2), sharding=w)
xx = x.add(x).add(x).max(dim=0)
f = xx[0].to_local()
print(f.wait())

with active_sharding(sharding=w):
    o = torch.ones(2) + x


x.to_local()
o.to_local().then(lambda f: print("CB", f))
to_local({'x': x, 'o': o, 'y': x + x + x}).then(print)
print("BEFORE WAIT")
w.wait_all()
