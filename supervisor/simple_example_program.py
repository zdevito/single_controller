import time
import random
import os
import zmq
import pickle


def start_message_queue():
    ctx = zmq.Context(1)
    sock = ctx.socket(zmq.DEALER)
    proc_id = int(os.environ['SUPERVISOR_IDENT']).to_bytes(8, byteorder='little')
    sock.setsockopt(zmq.IDENTITY, proc_id)
    sock.connect(os.environ['SUPERVISOR_PIPE'])
    sock.send(b'')
    return sock

sock = start_message_queue()

r = random.randint(1, 5)
print('rank', os.environ['RANK'], r)
print('rank', os.environ['RANK'], 'got', pickle.loads(sock.recv()))
print(os.environ['RANK'], 'sleeps ', r)
time.sleep(r)
sock.send(pickle.dumps(r))
