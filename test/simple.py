from single_controller import DTensor, ActiveSharding, Manager, LocalWorker
import torch
real = False
if real:
    m = Manager()
    w = m.create_worker()
else:
    w = LocalWorker()

x = DTensor.to_remote(torch.ones(2, 2), sharding=w)
xx = x.add(x).add(x).max(dim=0)
f = xx[0].to_local()
print(f.wait())

with ActiveSharding(sharding=w):
    o = torch.ones(2) + x


x.to_local()
o.to_local().add_done_callback(lambda f: print("CB", f.result()))

w.wait_all()
