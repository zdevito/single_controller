from supervisor import get_message_queue
import random
import os
import time


if __name__ == '__main__':
    rank = int(os.environ['RANK'])
    q = get_message_queue()
    health = random.randrange(0, 4)
    # if rank == 2:
    #     print(f"Rank {rank} is going to report unhealthy")
    #     health = 100
    #time.sleep(health)
    q.send_pyobj(health)
