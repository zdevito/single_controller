import os
import subprocess
import sys
global_host_rank = int(os.environ['TW_TASK_ID'])
N = int(os.environ['MAST_HPC_TASK_GROUP_SIZE'])
hostname_0 = os.environ['MAST_HPC_TASK_GROUP_HOSTNAMES'].split(',')[0]

addr = f'tcp://{hostname_0}:55555'
if global_host_rank == 0:
    host_process = subprocess.Popen([sys.executable, '-m', 'supervisor.host', addr])
    from .supervise import main
    main(N)
else:
    from supervisor.host import main
    main(addr)
