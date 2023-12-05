import sys
import subprocess
import signal
from contextlib import contextmanager
import time
import os

# this script simulates what a job scheduler might do.
# it will launch our host managers, and supervisor
# and then relaunch the host managers if the error
# to simulate the host manager moving to a new host

N = 4

# version that restarts hosts when they die
def restarting():
    supervise = subprocess.Popen([sys.executable, '-m', 'example_train.supervise', str(N)])
    def create_host():
        return subprocess.Popen([sys.executable, '-m', 'supervisor.host', 'tcp://localhost:55555'])
    hosts = [create_host() for i in range(N)]
    while True:
        if supervise.poll() is not None:
            if supervise.returncode != 0:
                for h in hosts:
                    h.send_signal(signal.SIGTERM)
            break
        for i, c in enumerate(hosts):
            if c.poll() is not None:
                if c.returncode != 0:
                    print(f'Host {i} manager exited, restarting...')
                    time.sleep(1)
                    hosts[i] = create_host()
        time.sleep(.1)

def emulate_mast_launch():
    # for testing the mast launcher entrypoint
    def create_host(i):
        env = {**os.environ}
        env['TW_TASK_ID'] = str(i)
        env['MAST_HPC_TASK_GROUP_HOSTNAMES'] = 'localhost'
        env['MAST_HPC_TASK_GROUP_SIZE'] = str(N)
        return subprocess.Popen([sys.executable,  '-m', 'example_train.mast'], env=env)
    hosts = [create_host(i) for i in range(N)]
    while hosts:
        finished = []
        status = [h.poll() for h in hosts]
        for i in range(len(hosts)):
            if status[i] is not None and status[i] != 0:
                print(f'Host {i} manager exited with {status[i]}, exiting...')
                for c in hosts:
                    c.send_signal(signal.SIGTERM)
                return
        hosts = [h for h, s in zip(hosts, status) if s is None]
        time.sleep(.1)

emulate_mast_launch()
