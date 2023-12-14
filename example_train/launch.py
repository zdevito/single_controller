import sys
import subprocess
import signal
from contextlib import contextmanager
import time
import os
import socket
# this script simulates what a job scheduler might do.
# it will launch our host managers, and supervisor
# and then relaunch the host managers if the error
# to simulate the host manager moving to a new host

N = 4

# version that restarts hosts when they die
def restarting():
    supervise = subprocess.Popen([sys.executable, '-m', 'example_train.supervise', str(N)])
    def create_host():
        return subprocess.Popen([sys.executable, '-m', 'supervisor.host', 'tcp://devgpu005.ncg1.facebook.com:55555'])
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

def emulate_mast_launch(to_launch):
    # for testing the mast launcher entrypoint
    def create_host(i):
        env = {**os.environ}
        env['TW_TASK_ID'] = str(i)
        env['MAST_HPC_TASK_GROUP_HOSTNAMES'] = socket.gethostname()
        env['TORCH_ELASTIC_SUPERVISOR'] = str(i == 0)
        env['MAST_HPC_TASK_GROUP_SIZE'] = str(N)
        return subprocess.Popen(to_launch, env=env)
    hosts = [create_host(i) for i in range(N)]
    print("PIDS", [h.pid for h in hosts])
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

if sys.argv[1:]:
    emulate_mast_launch(sys.argv[1:])
else:
    emulate_mast_launch((sys.executable, '-m', 'example_train.supervise'))

#restarting()
