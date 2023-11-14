import zmq
import time
import pickle
from client import popen
import threading
import traceback

context = zmq.Context(1)
HEARTBEAT_LIVENESS = 3     # 3..5 is reasonable
HEARTBEAT_INTERVAL = 1.0

backend = context.socket(zmq.ROUTER)
backend.bind('tcp://*:55555')

frontend = context.socket(zmq.ROUTER)
frontend.bind('inproc://futures')

poller = zmq.Poller()
poller.register(backend, zmq.POLLIN)
poller.register(frontend, zmq.POLLIN)

class Client:
    def __init__(self, name):
        self.name = name
        self.heartbeat()
    def heartbeat(self):
        self.expiry = time.time() + HEARTBEAT_INTERVAL * HEARTBEAT_LIVENESS
    def __repr__(self):
        return f"Client({self.name})"

workers = {}
free_worker_names = []
hosts_to_allocate = []

process_waits = {}
process_completes = {}


class Frontend:
    def allocate_host(self, ref):
        hosts_to_allocate.append(ref)
    def create_process(self, ref, host, args):
        m = pickle.dumps( ('popen', ref, 'client.hello', workers[host].name, args) )
        backend.send_multipart([host, m])
    def wait_process(self, ref, proc):
        if proc in process_completes:
            v = process_completes.pop(proc)
            frontend.send_multipart([ref, v])
        else:
            process_waits[proc] = ref

frontend_interface = Frontend()

def supervisor():
    while True:
        for sock, _ in poller.poll(timeout=HEARTBEAT_INTERVAL*1000*2):
            if sock is backend:
                f, msg = backend.recv_multipart()
                if f not in workers:
                    workers[f] = Client(msg.decode())
                    free_worker_names.append(f)
                    print(f"New Client: {f}")
                else:
                    workers[f].heartbeat()
                    if len(msg):
                        action, *args = pickle.loads(msg)
                        if action == 'started':
                            ref, = args
                            frontend.send_multipart([ref, pickle.dumps(ref)])
                        elif action == 'finished':
                            ref, code, msg = args
                            v = pickle.dumps((code,msg))
                            if ref in process_waits:
                                proc = process_waits.pop(ref)
                                frontend.send_multipart([proc, v])
                            else:
                                process_completes[ref] = v
                    else:
                        print(f"{workers[f]} Heartbeat")
            elif sock is frontend:
                ref, msg  = frontend.recv_multipart()
                msg = pickle.loads(msg)
                print(msg)
                method, *args = msg
                print(method, args)
                getattr(frontend_interface, method)(ref, *args)

        while free_worker_names and hosts_to_allocate:
            name, ref = free_worker_names.pop(0), hosts_to_allocate.pop(0)
            frontend.send_multipart([ref, pickle.dumps(name)])

        t = time.time()
        # priority queue would be log(N)
        expired = [(key, client) for key, client in workers.items() if client.expiry < t]
        for key, client in expired:
            print(f"Lost {client}")
            del workers[key]

        print(workers, free_worker_names, hosts_to_allocate, process_waits, process_completes)

        # for i in range(4):
        #     global message, proc_ref
        #     if t > message:
        #         message += 4
        #         for k, worker in workers.items():
        #             m = pickle.dumps(('popen', proc_ref, 'client.hello', worker.name, 1))
        #             backend.send_multipart([k, m])
        #             proc_ref += 1


supervisor_thread = threading.Thread(target=supervisor, daemon=True)
supervisor_thread.start()



def _future():
    fut = context.socket(zmq.DEALER)
    fut.connect('inproc://futures')
    return fut

def allocate_host():
    f = _future()
    f.send(pickle.dumps(('allocate_host',)))
    return f

def create_process(host, args):
    f = _future()
    f.send(pickle.dumps(('create_process', host, args)))
    return f

def wait_process(proc):
    f = _future()
    f.send(pickle.dumps(('wait_process', proc)))
    return f


def wait(f):
    return pickle.loads(f.recv())

def some_hosts():
    for h in [allocate_host() for _ in range(2)]:
        yield wait(h)

def some_procs(hosts):
    for h in hosts:
        procs = [create_process(h, 1) for _ in range (2)]
        for p in procs:
            yield wait(wait_process(wait(p)))

class Future:
    def __init__(self, p):
        self.p = _future()

for p in some_procs(some_hosts()):
    print(p)

# supervisor_thread.join()
