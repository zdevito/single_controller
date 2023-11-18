import time
import random
import os
import zmq
import pickle
from supervisor import get_message_queue


sock = get_message_queue()
r = random.randint(1, 5)
print('rank', os.environ['RANK'], r)
print('rank', os.environ['RANK'], 'got', pickle.loads(sock.recv()))
print(os.environ['RANK'], 'sleeps ', r)
time.sleep(r)
sock.send(pickle.dumps(r))
