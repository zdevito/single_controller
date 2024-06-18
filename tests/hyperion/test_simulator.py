from unittest import main, TestCase
import torch
from controller._testing import simulator_mesh
from controller.simulator import simulate_commands
from controller import (
    DeviceMesh,
    active_mesh,
    active_stream,
    Stream,
    fetch_shard,
    Future,
    Stream,
    get_active_stream,
)
import random
import math

def get_max_runtime(events):
    end_time = 0
    for worker_events in events.values():
        for event in worker_events:
            end_time = max(end_time, event.end)
    return end_time
        

class TestSimulator(TestCase):
    def test_mm(self):
        mesh = simulator_mesh(hosts=1, gpus=2)
        with active_mesh(mesh):
            x = torch.randn(5, 5)
            torch.mm(x, x)
        events = simulate_commands(mesh.ctrl.backend.worker_commands)
        tm = get_max_runtime(events)
        self.assertEqual(tm, 250)

    def test_mm_parallel(self):
        mesh = simulator_mesh(hosts=1, gpus=2)
        slice0 = mesh(gpu=0)
        slice1 = mesh(gpu=1)
        with active_mesh(slice0):
            x = torch.randn(5, 5)
            torch.mm(x, x)

        with active_mesh(slice1):
            x = torch.randn(5, 5)
            torch.mm(x, x)
        events = simulate_commands(mesh.ctrl.backend.worker_commands)
        tm = get_max_runtime(events)
        self.assertEqual(tm, 250)

    def test_mm_sequential(self):
        mesh = simulator_mesh(hosts=1, gpus=2)
        slice0 = mesh(gpu=0)
        slice1 = mesh(gpu=1)
        with active_mesh(slice0):
            x = torch.randn(5, 5)
            x = torch.mm(x, x)
            x = x.to_mesh(slice1)
        with active_mesh(slice1):
            torch.mm(x, x)

        events = simulate_commands(mesh.ctrl.backend.worker_commands)
        tm = get_max_runtime(events)
        self.assertEqual(tm, 500)
