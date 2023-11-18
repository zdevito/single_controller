
import os
import random
import time
import signal

rank = int(os.environ['RANK'])

for i in range(100):
    if rank == 0:
        print(f"iteration {i}")
    r = random.randrange(0, 200)
    if False and 0 == r:
        raise Exception(f"Rank {rank} failure!")
    elif 0 == r or 1 == r:
        # simulate losing the host
        print(f"Rank {rank} is losing its host!")
        os.kill(os.getppid(), signal.SIGTERM)
    time.sleep(1)
