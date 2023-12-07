import os
import subprocess
import sys
import socket
import logging
logger = logging.getLogger()
N = int(os.environ['MAST_HPC_TASK_GROUP_SIZE'])
hostname_0 = sorted(os.environ['MAST_HPC_TASK_GROUP_HOSTNAMES'].split(','))[0]

my_host_name = socket.gethostname()
if 'LAUNCH_FAKE' in os.environ:
    zero = int(os.environ['TW_TASK_ID']) == 0
else:
    zero = hostname_0 in my_host_name

logger.warn("HOSTNAME %s %s", my_host_name, zero)
port = 55555
addr = f'tcp://{hostname_0}.facebook.com:{port}'
if zero:
    host_process = subprocess.Popen([sys.executable, '-m', 'supervisor.host', addr])
    from .supervise import main
    main(N, port)
else:
    from supervisor.host import main
    main(addr)
