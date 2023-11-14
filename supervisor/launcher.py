import sys
import subprocess

server = subprocess.Popen([sys.executable, '-m', 'supervisor'])
clients = [subprocess.Popen([sys.executable, '-m', 'client', 'tcp://localhost:55555']) for i in range(int(sys.argv[1]))]
print(f"PIDS: {server.pid}, {[c.pid for c in clients]}")
r = [client.wait() for client in clients]
server.wait()
