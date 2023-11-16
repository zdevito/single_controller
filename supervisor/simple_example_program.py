import time
import random
import os

r = random.randint(1, 5)
print(os.environ['RANK'], 'sleeps ', r)
time.sleep(r)
