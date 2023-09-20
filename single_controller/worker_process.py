from socket import AF_INET, SOCK_STREAM, socket
from . import RemoteRef, BaseWorker
import pickle
import sys
import torch
from time import sleep
from torch.utils._pytree import tree_flatten, tree_unflatten, tree_map

# log what is happening
verbose = False

no_response = object()

class LocalWorker(BaseWorker):
    def __init__(self, host, port, secret):
        self.socket = socket(AF_INET, SOCK_STREAM)
        self.socket.connect((host, int(port)))
        self.ofile = self.socket.makefile("wb")
        self.ifile = self.socket.makefile("rb")
        self._write_pickle(secret)
        self.ofile.flush()
        self.ref_to_tensor = {}

    def run(self):
        while True:
            method, *args = self._read_pickle()
            if verbose:
                if method == "define_function":
                    fn, body = args
                    print(f"method: define_function {fn}\n{body}")
                else:
                    print("method:", method, args)
            if method == 'exit':
                return
            elif method == 'request_value':
                ref, = args
                result = self.ref_to_tensor[ref.id]
                self._write_pickle(result)
                self.ofile.flush()
            else:
                getattr(self, method)(*args)

    def send_command(self, func, args, kwargs, results):
        def get_tensor(t):
            if isinstance(t, RemoteRef):
                return self.ref_to_tensor[t.id]
            else:
                return t
        args = tree_map(get_tensor, args)
        kwargs = tree_map(get_tensor, kwargs)
        result = func(*args, **kwargs)
        flat_results, _ = tree_flatten(result)
        real_results = [e for e in flat_results if isinstance(e, torch.Tensor)]
        for real, ref in zip(real_results, results):
            self.ref_to_tensor[ref.id] = real


    def send_value(self, ref: RemoteRef, value: torch.Tensor):
        self.ref_to_tensor[ref.id] = value

    def del_value(self, ref: RemoteRef):
        del self.ref_to_tensor[ref.id]

    def _write_pickle(self, obj):
        b = pickle.dumps(obj)
        sz = len(b).to_bytes(8, 'little')
        self.ofile.write(sz)
        self.ofile.write(b)

    def _read_pickle(self):
        sz = int.from_bytes(self.ifile.read(8), 'little')
        return pickle.loads(self.ifile.read(sz))

    def create_process_group(self, rank, world_size, pg_ref):
        torch.distributed.init_process_group('nccl', init_method='tcp://127.0.0.1:12346', rank=rank, world_size=world_size)
        self.ref_to_tensor[pg_ref.id] = None

    def create_process_subgroup(self, orig_pg, participating_ranks, pg):
        pg = self.ref_to_tensor[orig_pg.id]
        assert pg is None, "subgroup must be created from default group..."
        r = torch.distributed.new_group(ranks=participating_ranks, backend='nccl')
        if pg is not None:
            self.ref_to_tensor[pg.id] = r

    def all_reduce(self, ref: RemoteRef, pg_ref: RemoteRef):
        pg = self.ref_to_tensor[pg_ref.id]
        t = self.ref_to_tensor[ref.id]
        torch.distributed.all_reduce(t, group=pg)

if __name__ == "__main__":
    _, host, port, secret = sys.argv
    w = LocalWorker(host, port, secret)
    w.run()
