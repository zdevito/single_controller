import os
from contextlib import ExitStack, contextmanager
from weakref import WeakKeyDictionary
from supervisor import Context, HostConnected
from supervisor.host import Host as HostManager
from functools import partial
from threading import Thread
from controller.backend import ProcessBackend
from controller.controller import Controller
from controller.backend import MockBackend
from controller.ndslice import NDSlice
from controller.simulator import SimulatorBackend
from controller import world_mesh, active_mesh
from controller.device_mesh import DeviceMesh
import signal
from time import sleep
import logging
import torch

logger = logging.getLogger(__name__)

# code used for testing but useful to have importable (e.g. can refer to remote functions)
def do_bogus_tensor_work(x, y, fail_rank=None):
    if fail_rank is not None and int(os.environ["RANK"]) != fail_rank:
        return x
    return x @ y

def log(*args, **kwargs):
    logger.info(*args, **kwargs)

def has_nan(t):
    return torch.isnan(t).any().item()

class LocalContext:
    def __init__(self):
        self._all_hosts = WeakKeyDictionary()
        self.cleanup = ExitStack()
        self._process_cache = {}

    @contextmanager
    def _get_context(self, N, gpu_per_host):
        ctx = Context()
        ctx.request_hosts(N)
        threads = []
        # we want ctx to start its listener threads
        # before creating the hosts because
        # initialization will happen faster in this case
        sleep(0)

        def run_host(host: HostManager):
            host.run_event_loop_forever()

        for _ in range(N):
            host = HostManager("tcp://127.0.0.1:55555", start_new_session=False)
            self._all_hosts[host] = True
            thread = Thread(target=partial(run_host, host), daemon=True)
            thread.start()
            threads.append(thread)

        connections = ctx.messagefilter(HostConnected)
        hosts = [connections.recv(timeout=1).sender for _ in range(N)]
        store = ProcessBackend._create_store()
        processes = ProcessBackend._create_pg(ctx, hosts, gpu_per_host, store, _restartable=True)
        yield ctx, hosts, processes
        for p in processes:
            p.signal(signal.SIGTERM)
        ctx.shutdown()
        for th in threads:
            th.join(timeout=1)
            if th.is_alive():
                raise TimeoutError()

    def _processes(self, N, gpu_per_host):
        key = (N, gpu_per_host)
        if key not in self._process_cache:
            self._process_cache[key] = self.cleanup.enter_context(self._get_context(N, gpu_per_host))
        return self._process_cache[key]
    
    def close(self):
        self.cleanup.close()
    
    def interrupt(self):
        for host in self._all_hosts.keys():
            for proc in host.process_table.values():
                os.killpg(proc.subprocess.pid, signal.SIGKILL)

    @contextmanager
    def local_device_mesh(self, N, gpu_per_host, activate=True):
        ctx, hosts, processes = self._processes(N, gpu_per_host)
        dm = world_mesh(ctx, hosts, gpu_per_host, _processes=processes)
        if activate:
            with active_mesh(dm):
                yield dm
        else:
            yield dm
        dm.ctrl.shutdown()

    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.cleanup.close()


def _example_mesh(hosts: int, gpus: int):
    with LocalContext() as c, c.local_device_mesh(hosts, gpus, activate=False) as dm:
        yield dm

def example_mesh(hosts: int, gpus: int):
    it = _example_mesh(hosts, gpus)
    dm = next(it)
    def exit():
        try:
            next(it)
        except StopIteration:
            pass
    dm.exit = exit
    return dm

def mock_mesh(hosts: int, gpus: int, verbose=True):
    backend = MockBackend(hosts*gpus, verbose=verbose)
    ctrl = Controller(backend)
    dm = DeviceMesh(ctrl, NDSlice(0, [hosts, gpus], [gpus, 1]), ('host', 'gpu'))
    dm.exit = lambda: None
    return dm

def simulator_mesh(hosts: int, gpus: int):
    backend = SimulatorBackend(hosts*gpus)
    ctrl = Controller(backend)
    dm = DeviceMesh(ctrl, NDSlice(0, [hosts, gpus], [gpus, 1]), ('host', 'gpu'))
    dm.exit = lambda: None
    return dm
