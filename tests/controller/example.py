from controller import DeviceMesh, active_mesh, active_stream, Stream, fetch_shard, Future, Stream, get_active_stream
import torch
# this function helps get a local device mesh for testing
from controller._testing import example_mesh, mock_mesh
from controller.device_mesh import remote_function

mocked = False
if mocked:
    device_mesh = mock_mesh(hosts=2, gpus=2)
else:
    device_mesh = example_mesh(hosts=2, gpus=2)


# device meshes initially describe the hardware that 
# the job will use

# Basics
# ------

# This is normal torch, we create a local cpu tensor:
l = torch.rand(3, 4)

# however, we activate the device_mesh and tensors
# will be created across the mesh

with active_mesh(device_mesh):
    t = torch.rand(3, 4)

# for interactive use lets keep this device_mesh active
device_mesh.activate()

# user-defined remote functions
@remote_function('controller._testing.log')
def log(*args, **kwargs):
    None

@remote_function('controller._testing.has_nan')
def has_nan(t):
    return False

# run on workers
log("my tensor: %s", t)

# you can do mutable stuff
t.add_(1)

# devices still work as normal
t = t.cuda()


# Communication Operators
# -----------------------


# most comms turn into a 'reduce' with
# different settings, but we will have 
# syntax sugar for common things like 'all_gather'
# and 'all_to_all'

x = (device_mesh.rank('host') + 1) * 1000000 + (device_mesh.rank('gpu') + 1) * 1000 + torch.arange(6).reshape(2, 3) + 1
log("orig tensor:\n%s", x)

x = x.cuda()


t = x.reduce('gpu')

log("reduced tensor:\n%s", t)

# inplace

t.reduce_('host')

log("reduced tensor:\n%s", t)

# 'gather'
gathered = t.reduce('gpu', reduction='stack')
log("gathered tensor:\n%s", gathered)

# reduce-scatter
reduce_scatter = x.reduce('gpu', scatter=True)
log("before\n%s\nscattered:\n%s\n", x, reduce_scatter)


# Observing results on controller
# -------------------------------

# to get a value locally you can fetch the local
# value on a particular shard, which returns a future
l: Future = fetch_shard(reduce_scatter, dict(host=1, gpu=0))


with active_mesh(None):
    print(l.result())

# you don't always want tensor though, so you can pre
# process the value before sending it:
l = fetch_shard(reduce_scatter, dict(host=1, gpu=0), has_nan)
print(l.result())


# Moving Tensors
# -------

# you can select a subset of the devices with
# name-based indexing
host0 = device_mesh(host=0)
host1 = device_mesh(host=1)

with active_mesh(host0):
    a = torch.rand(2, 3, device='cuda')
    # send data from one mesh to another
    # there is not possibility of mismatched send/recv
    # because both are issued at the same time to 
    # the correct workers. So no possibility of deadlock!
    b = a.to_mesh(host1)

    # or slice a bigger mesh and send it to another
    c = t.slice_mesh(host=0).to_mesh(host1)

# the receiving host can then compute with those
# sent tensors
with active_mesh(host1):
    d = b + c
    ld = fetch_shard(d, dict(gpu=1))

with active_mesh(None):
    print(ld.result())



# Errors
# -------
# what happens if something goes wrong on a worker
# that we cannot statically know from the controller?

@remote_function('controller._testing.do_bogus_tensor_work')
def do_bogus_tensor_work(x, y, fail_rank=None):
    # this is wrong, the real function does x @ y
    return x + y


t = torch.rand(3, 4, device='cuda')

r = do_bogus_tensor_work(t, t)

print(fetch_shard(r).exception())

x = t + t
log("x: %s", x)

# only rank 1 will fail now
r = do_bogus_tensor_work(t, t, fail_rank=1)

reduced = r.reduce('gpu')

# notice the error is still the 
# op that started the failure
print(fetch_shard(reduced).exception())

# but we can still compute with things not dependent on the error
print(fetch_shard(t + t).exception())

# notice that t + t happened _after_ a reduce operator
# failed become one of the workers didn't have a valid tensor.
# with standard distributed, this would have been a deadlock!


# Streams
# -------
# express parallel. Each worker

# we have been using a default stream
default = get_active_stream()

# but we can create others
# on each worker, code on the same stream runs in order
# but code on different streams runs in parallel
comms = Stream('comms') # argument is just a debug name

# parallel compute
t = torch.rand(3, 4, device='cuda')
with active_stream(comms):
    t2 = torch.rand(3, 4, device='cuda')

# you can't directly use tensors across streams
# (when they are computed it is a race)
try:
    t.add(t2)
except Exception as e:
    print(e)

# but you can borrow a tensor to another stream
# this will insert the appropriate synchronization

t2_on_default = default.borrow(t2)
r = t.add(t2_on_default)

# while t2 has a borrow, you can't mutate it
# because that would race with the borrow.
# you can also borrow with mutate=True, 
# which allows the borrowing stream to mutate
# the tensor, but disables _reads_ of the tensor
# from other streams.
try:
    with active_stream(comms):
        t2.add_(1)
except Exception as e:
    print(e)

# when you are done, you have to explicitly
# tell the borrow it is done, because this will
# synchrionize t2's _memory_ back to the comms stream
t2_on_default.drop()


# drop() can actually be used on any tensor
# it says 'get rid of the memory for this tensor now'
# if it or any view of it is used later, that will
# cause an error. This is a good way to make assertions
# about the lifetime of tensors.

# just for local testing to shut down all the workers
device_mesh.exit()
