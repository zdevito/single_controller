from socket import AF_INET, SOCK_STREAM, socket
from . import RemoteRef
import pickle
import sys
import torch
from time import sleep
from torch.utils._pytree import tree_flatten, tree_unflatten, tree_map


no_response = object()

class LocalWorker:
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
            print("method:", method, args)
            if method == 'exit':
                return
            result = getattr(self, method)(*args)
            if result is not no_response:
                self._write_pickle(result)
                self.ofile.flush()

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
        return no_response

    def request_value(self, ref: RemoteRef):
        return self.ref_to_tensor[ref.id]

    def send_value(self, ref: RemoteRef, value: torch.Tensor):
        self.ref_to_tensor[ref.id] = value
        return no_response

    def del_value(self, ref: RemoteRef):
        del self.ref_to_tensor[ref.id]
        return no_response

    def _write_pickle(self, obj):
        b = pickle.dumps(obj)
        sz = len(b).to_bytes(8, 'little')
        self.ofile.write(sz)
        self.ofile.write(b)

    def _read_pickle(self):
        sz = int.from_bytes(self.ifile.read(8), 'little')
        return pickle.loads(self.ifile.read(sz))

if __name__ == "__main__":
    _, host, port, secret = sys.argv
    w = LocalWorker(host, port, secret)
    w.run()
