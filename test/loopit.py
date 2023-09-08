from single_controller import DTensor, active_sharding, Manager, LocalWorker, to_local
import torch
real = True
if real:
    m = Manager()
    w = m.create_worker()
else:
    w = LocalWorker()

x = DTensor.to_remote(torch.zeros(2, 2), sharding=w)

for i in range(10000):
    if i % 100 == 0:
        x.to_local().then(print)
    x = x + 1

print(x.to_local().wait())
