import os
import subprocess
import sys
import socket
import logging
logger = logging.getLogger()
N = int(os.environ['MAST_HPC_TASK_GROUP_SIZE'])
hostname_0 = os.environ['MAST_HPC_TASK_GROUP_HOSTNAMES'].split(',')[0]

my_host_name = socket.gethostname()
addr = f'tcp://{hostname_0}.facebook.com:55555'
zero = hostname_0 in my_host_name
logger.warn("HOSTNAME %s %s", my_host_name, zero)
if zero:
    host_process = subprocess.Popen([sys.executable, '-m', 'supervisor.host', addr])
    from .supervise import main
    main(N)
else:
    from supervisor.host import main
    main(addr)
