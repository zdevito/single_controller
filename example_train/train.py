
import os
import random
import time
import signal

rank = int(os.environ['RANK'])
# for i in range(100):
#     if rank == 0:
#         print(f"iteration {i}")

#     if rank == 1 and i == 5:
#         if random.randrange(0, 2) == 0:
#             raise Exception(f"Rank {rank} failure!")
#         else:
#             print(f"Rank {rank} is losing its host!")
#             os.kill(os.getppid(), signal.SIGTERM)

#     time.sleep(1)
