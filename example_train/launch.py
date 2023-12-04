import sys
import subprocess
import signal
from contextlib import contextmanager
import time

# this script simulates what a job scheduler might do.
# it will launch our host managers, and supervisor
# and then relaunch the host managers if the error
# to simulate the host manager moving to a new host
N = 4
supervise = subprocess.Popen(['example_train', str(N)])
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
