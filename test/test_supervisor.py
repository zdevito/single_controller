from supervisor import Context
import sys
import subprocess
import signal
from concurrent.futures import as_completed
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

    futs = []
    for p in pg:
        f = p.returncode()
        f.p = p # YUCK, I DO NOT LIKE HOW AS
        futs.append(f)

    for f in as_completed(futs):
        code = f.result()
        print(f.p.rank, " FINISHED WITH CODE", code)

    ctx.return_hosts(hosts)

# TO PONDER LIST
# - how do we quickly launch processes that have different arguments per process such as pids
# - what does the real launcher script look like?
# - how does message filtering work (simple function?)
