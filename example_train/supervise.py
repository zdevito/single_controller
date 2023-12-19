import logging
logging.basicConfig(format='%(asctime)s %(levelname)s:%(name)s:%(message)s', level=logging.INFO)

from supervisor import Context, Host, Process, Future
from typing import List, Dict, Any, Set
import sys
import subprocess
import signal
from supervisor import as_completed
from supervisor.launchers import mast
from contextlib import contextmanager
import time
import os

logger = logging.getLogger(__name__)

def start_training(ctx, N: int, hosts: Set[Host], npp: int, run_fraction=.5, rank_fraction=.75):
    # we will use 10% of machines as fallover machines.
    desired_run_size: int = int(run_fraction*N)

    # The supervisor can now create processes on hosts.
    # We will start by running a health check on all of our machines
    # to find the 90% percentile machines and exclude the bottom 10%.

    logger.info(f"starting health checks host {len(hosts)} hosts")
    pg: List[Process] = ctx.create_process_group(hosts, args=[sys.executable, '-m', 'example_train.health_check'], processes_per_host=1, name='health_check')

    health_responses: Dict[Future[Any], Process] = {p.recv(): p for p in pg}


    # as_completed returns messages as we receive them, avoiding waiting for stragglers.
    # if we do not hear from enough machines in 5 minutes, we assume
    # something about the cluster is unhealthy and then bail out entirely
    TIMEOUT = 60*5
    responding_machines = as_completed(health_responses.keys(), timeout=TIMEOUT)

    # some of the machines that report back might be completely unhealthy
    # lets remove them before we rank
    working_machines = filter(lambda x: healthy(x), responding_machines)

    # But we don't have to hear from all of our machines before starting,
    # some might have major health check issues that will cause them to hang
    # for awhile, let's only sort the first 99% of machines by using zip
    # to stop iteration after to_sort machines have been received
    to_sort = int(rank_fraction * N)
    working_machines = zip(range(to_sort), working_machines)

    to_rank: List[Score, Process] = [(f.result(), health_responses.pop(f))
                                     for _, f in working_machines]

    logger.info(f"Found {len(to_rank)} hosts that passed health checks, ranking...")
    to_rank = sorted(to_rank, key=lambda x: x[0])

    # TODO NEXT: if we end up with too few hosts to start,
    # try to replace them and retry, because right now we get
    # stuck with hosts that have disconnected but not marked for replacement
    # yet
    if len(to_rank) < desired_run_size:
        raise Exception('Not enough healthy hosts')

    good, slow = to_rank[:desired_run_size], to_rank[desired_run_size:]
    good_hosts = [p.host for score, p in good]

    logger.info(f"Chose hosts: {[p.rank for _, p in good]}")

    # Let's get training started.
    logger.info(f"Launching {npp*desired_run_size} processes")
    env = {
        'OMP_NUM_THREADS': '1',
        'TORCH_NCCL_ASYNC_ERROR_HANDLING': os.environ.get('TORCH_NCCL_ASYNC_ERROR_HANDLING', '1'),
        'MASTER_ADDR': f'{good_hosts[0].hostname().result()}.facebook.com',
        'MASTER_PORT': str(50687),
    }
    process_group = ctx.create_process_group(good_hosts, args=[sys.executable, '-m', 'example_train.train'], processes_per_host=npp, env=env, name='train')

    # now simultaneously with training lets sort out what to do with our
    # stragglers. slow hosts are probably ok to keep, they responded
    # and are healthy. There is 1% of hosts we didn't wait for to cut tail latency.
    # Let's see if any have checked in anyway and should be considered healthy
    # but maybe slower.
    return process_group, good_hosts

    try:
        for f in as_completed(health_responses.keys(), timeout=60*5):
            if healthy(f):
                health_responses.pop(f)
    except TimeoutError:
        pass

    # the remaining hosts in health_responses have either not responded
    # or are unhealthy.
    unhealthy_hosts = [proc.host for proc in health_responses.values()]
    logger.info(f"Replacing unhealthy hosts: {unhealthy_hosts}")
    # We will ask the these hosts get replaced in the job scheduler.
    # All work scheduled on them will be aborted/cancelled.

    hosts.difference_update(unhealthy_hosts)
    hosts.update(ctx.replace_hosts(unhealthy_hosts))
    # Mechanically, this will cause the host manager on the host to
    # exit with an error condition. If MAST is in TASK-level scope,
    # this will cause mast to reschedule that host somewhere else
    # if it is unhealthy. That new instance of the host manager will
    # check in with the supervisor with assign it to unconnected host.

    return process_group, good_hosts

def healthy(score):
    return score.exception() is None and score.result() < 4


def train_with_size(ctx, hosts):
    hosts = set(hosts)
    npp = 8
    N = len(hosts)
    logger.info(f"Starting training with {N} hosts, {npp} processes per host.")
    # Acquire some host machines to run on.
    # For today's job schedulers (Slurm/MAST), this will work by
    # having the job scheduler launch the supervisor on one host,
    # and a host manager on every host.
    # The API to request hosts simply connects the supervisor
    # to the host managers as they check in.

    # In the future, the supervisor can run
    # externally to the worker machines and actually
    # request and wait for the hosts to get provided.
    complete = False
    while not complete:
        process_group, current_hosts = start_training(ctx, N, hosts, npp=npp, run_fraction=1, rank_fraction=1)
        logger.info(f"Process group size {len(process_group)}")
        complete = True
        for f in as_completed([f.returncode() for f in process_group]):
            if f.exception() is not None or f.result() != 0:
                logger.info("Training has failed, attempting restart...")
                # training has failed, clean up the training processes
                for p in process_group:
                    p.signal(signal.SIGTERM) # TODO: maybe have a broadcasting signal function.
                disconnected = [h for h in current_hosts if h.connection_lost()]
                hosts.difference_update(disconnected)
                hosts.update(ctx.replace_hosts(disconnected))
                # policy-wise, there are a few ways to move forward
                # we can try to ensure all the signals were delivered
                # and the processes were killed, following up
                # with SIGKILL. Or we could just assume the message
                # made it, and continue to health checks.
                # The health check could then attempt to clean up
                # any stale processes and report unhealthy if
                # it cannot.
                complete = False
                break
    logger.info(f"Training exited successfully.")
    return list(hosts)

def main(N, port):
    #ctx = Context(port=port, log_format='/tmp/dedicated_{name}.log')
    ctx = Context(port=port)
    hosts: List[Host] = ctx.request_hosts(n=N)

    # if we don't warmup, then we can get started before all of our
    # other hosts have checked in.
    logger.info(f"Beginning warmup")
    hosts = train_with_size(ctx, hosts)
    logger.info(f"Warmup finished.")

    to_train = 1
    while to_train <= N:
        hosts[:to_train] = train_with_size(ctx, hosts[:to_train])
        to_train *= 2
    ctx.shutdown()

if __name__ == '__main__':
    mast(main)
