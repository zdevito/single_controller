import socket
import os
import subprocess
import logging
from .host import main
import sys
PORT = 55555

logger = logging.getLogger(__name__)

def mast(supervise):
    N = int(os.environ['MAST_HPC_TASK_GROUP_SIZE'])
    my_host_name = socket.gethostname()
    hostname_0 = sorted(os.environ['MAST_HPC_TASK_GROUP_HOSTNAMES'].split(','))[0]
    hostname_0 = socket.getfqdn(hostname_0)
    # used to fake a mast launch locally to test this script
    if 'TORCH_ELASTIC_SUPERVISOR' in os.environ:
        is_supervisor = 'True' == os.environ['TORCH_ELASTIC_SUPERVISOR']
    else:
        is_supervisor = hostname_0 == my_host_name
    supervisor_addr = f'tcp://{hostname_0}:{PORT}'
    logger.info("hostname %s, supervisor host is %s, supervisor=%s", my_host_name, hostname_0, is_supervisor)
    if is_supervisor:
        # local host manager on supervisor machine
        host_process = subprocess.Popen([sys.executable, '-m', 'supervisor.host', supervisor_addr])
        try:
            supervise(N, PORT)
        except:
            host_process.kill()
            host_process.wait()
            raise
    else:
        # host manager on non-supervisor machine
        host_process = subprocess.Popen([sys.executable, '-m', 'supervisor.host', supervisor_addr])
        result = host_process.wait()
        if result != 0:
            # Until we can use HPC_TASK level restarts, workaround it,
            # by not letting host machines report an error even when the machine fails.
            logger.info(f"Host manager exited with non-zero code {result}, but we are exiting cleanly so the fleet wide job doesn't end.")
