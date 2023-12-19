
import os
import random
import time
import signal
import subprocess

# to check if it will clean up the subprocess when trainer exits
# new_proc = subprocess.Popen(['python', '-c', 'while True:\n  print(0)\n  import time; time.sleep(.3)\n'])

rank = int(os.environ['RANK'])
# for i in range(100):
#     if rank == 0:
#         print(f"iteration {i}")
#         if i == 5:
#             os.kill(os.getppid(), signal.SIGTERM)
#             #raise Exception(f"Rank {rank} failure!")
#     # if rank == 1 and i == 5:
#     #     if random.randrange(0, 2) == 0:
#     #         raise Exception(f"Rank {rank} failure!")
#     #     else:
#     #         print(f"Rank {rank} is losing its host!")
#     #         os.kill(os.getppid(), signal.SIGTERM)

#     time.sleep(1)
