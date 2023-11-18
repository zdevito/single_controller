from supervisor import Context, Host, Process, Future
from typing import List, Dict, Any
import sys
import subprocess
import signal
from supervisor import as_completed
from contextlib import contextmanager


def main(N):
    ctx = Context()
    # Acquire some host machines to run on.
    # For today's job schedulers (Slurm/MAST), this will work by
    # having the job scheduler launch the supervisor on one host,
    # and a host manager on every host.
    # The API to request hosts simply connects the supervisor
    # to the host managers as they check in.

    # In the future, the supervisor can run
    # externally to the worker machines and actually
    # request and wait for the hosts to get provided.
    hosts: List[Host] = ctx.request_hosts(n=N).result()

    while True:
        process_group: List[Process] = start_training(hosts)
        for f in as_completed([f.returncode() for f in process_group]):
            if f.result() != 0:
                # training has failed, clean up the training processes
                for p in process_group:
                    p.signal(signal.SIGTERM) # TODO: maybe have a broadcasting signal function.
                # policy-wise, there are a few ways to move forward
                # we can try to ensure all the signals were delivered
                # and the processes were killed, following up
                # with SIGKILL. Or we could just assume the message
                # made it, and continue to health checks.
                # The health check could then attempt to clean up
                # any stale processes and report unhealthy if
                # it cannot.


def start_training(hosts: List[Host]):
    # we will use 10% of machines as fallover machines.
    desired_run_size: int = int(.9*N)

    # The supervisor can now create processes on hosts.
    # We will start by running a health check on all of our machines
    # to find the 90% percentile machines and exclude the bottom 10%.

    pg: List[Process] = ctx.create_process_group(hosts, args=['python', '-m', 'example.health_check'], npp=1)

    health_responses: Dict[Future[Any], Process] = {p.recv(): p for p in pg}


    # as_completed returns messages as we receive them, avoiding waiting for stragglers.
    # if we do not hear from enough machines in 5 minutes, we assume
    # something about the cluster is unhealthy and then bail out entirely
    TIMEOUT = 5*60
    responding_machines = as_completed(health_responses.keys(), timeout=TIMEOUT)

    # some of the machines that report back might be completely unhealthy
    # lets remove them before we rank
    working_machines = filter(lambda x: healthy(x.result()), responding_machines)

    # But we don't have to hear from all of our machines before starting,
    # some might have major health check issues that will cause them to hang
    # for awhile, let's only sort the first 99% of machines by using zip
    # to stop iteration after to_sort machines have been received
    to_sort = int(.99 * N)
    working_machines = zip(range(to_sort), working_machines)

    to_rank: List[Score, Process] = [(f.result(), health_responses.pop(f))
                                     for _, f in response_generator]

    to_rank = sorted(to_rank)
    good, slow = to_rank[:desired_run_size], to_rank[desired_run_size:]
    good_hosts = [p.host for score, p in good]

    # Let's get training started.
    process_group = ctx.create_process_group(good_hosts, args=['python', '-m', 'example.train'], npp=8)

    # now simultaneously with training lets sort out what to do with our
    # stragglers. slow hosts are probably ok to keep, they responded
    # and are healthy. There is 1% of hosts we didn't wait for to cut tail latency.
    # Let's see if any have checked in anyway and should be considered healthy
    # but maybe slower.
    for f in as_completed(health_responses.keys(), timeout=30):
        if healthy(f.result()):
            health_responses.pop(f)

    # the remaining hosts in health_responses have either not responded
    # or are unhealthy.
    unhealthy_hosts = [proc.host for proc in health_responses.values()]

    # We will ask the these hosts get replaced in the job scheduler.
    # All work scheduled on them will be aborted/cancelled.

    ctx.replace_hosts(unhealthy_hosts)
    # Mechanically, this will cause the host manager on the host to
    # exit with an error condition. If MAST is in TASK-level scope,
    # this will cause mast to reschedule that host somewhere else
    # if it is unhealthy. That new instance of the host manager will
    # check in with the supervisor with assign it to unconnected host.

    return process_group


# this would typically be done by a SLURM-style launcher script.
@contextmanager
def simulate_hosts(N):
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

if __name__ == '__main__':
    N = 2
    with simulate_hosts(N):
        main(N)
