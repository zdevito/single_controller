import sys
import pickle
import importlib
import os
import zmq

context = zmq.Context(1)
s = context.socket(zmq.DEALER)
s.setsockopt(zmq.IDENTITY, os.getpid().to_bytes(4, byteorder='little'))
s.connect(sys.argv[1])
s.send(b'')
global_name, *args = pickle.loads(s.recv())
module, fn = global_name.rsplit('.', 1)
r = pickle.dumps(getattr(importlib.import_module(module), fn)(*args))
s.send(r)
