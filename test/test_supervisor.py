from supervisor import Context
import sys
import subprocess
import signal
from supervisor import as_completed
from contextlib import contextmanager

N = 2
# this would typically be done by a SLURM-style launcher script.
@contextmanager
def create_hosts():
    procs = [subprocess.Popen([sys.executable, '-m', 'supervisor.host', 'tcp://localhost:55555']) for i in range(N)]
    try:
        yield
    except:
        for p in procs:
            p.send_signal(signal.SIGINT)
        raise
    finally:
        for p in procs:
            p.wait()

with create_hosts():
    ctx = Context()
    hosts = ctx.request_hosts(n=N).result()
    pg = ctx.create_process_group(hosts, args=['python', '-m', 'supervisor.simple_example_program'], npp=3)

    for p in pg:
        p.send('hello')

    try:
        for i, f in as_completed([p.recv(lambda x: isinstance(x, int)) for p in pg], enumerate=True, timeout=3):
            print(i, f.result())
    except TimeoutError:
        print("TIMEOUT")

    try:
        d = {p.returncode(): p for p in pg}
        for f in as_completed(d.keys(), timeout=1):
            p = d[f]
            print(p.rank, " FINISHED WITH CODE", f.result())
    except TimeoutError:
        print("TIMEOUT")

    ctx.return_hosts(hosts)

# TO PONDER LIST
# - how do we quickly launch processes that have different arguments per process such as pids
# - what does the real launcher script look like?
# - how does message filtering work (simple function?)
