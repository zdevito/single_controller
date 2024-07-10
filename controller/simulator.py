import torch
from collections import defaultdict
from copy import deepcopy
import contextlib
from typing import Optional, List, Sequence, NamedTuple, Tuple, Union
from .backend import MockBackend
from .ndslice import NDSlice

class Event:
    def __init__(self, start, runtime, meta):
        self.start = start
        self.runtime = runtime
        self.end = start + runtime
        self.meta = meta

    def __repr__(self):
        return f"E(meta={self.meta}, start={self.start:.2f}, end={self.end:.1f})"

def simulate_commands(worker_commands):
    worker_queues = deepcopy(worker_commands)
    worker_events = defaultdict(list)
    processed_events = {}
    def all_input_events(work_item):
        if work_item.collectives is None:
            return work_item.inputs
        else:
            return [x for item in work_item.collectives for x in item.inputs]

    def all_processed(work_items):
        return all([inp in processed_events for inp in work_items])

    def all_workers_involved(work_item):
        if work_item.collectives is None:
            return [work_item.worker]
        else:
            return [item.worker for item in work_item.collectives]

    def all_tasks_involved(work_item):
        if work_item.collectives is None:
            return [work_item]
        else:
            return work_item.collectives

    def is_first_in_queue(work_item):
        assert len(worker_queues[work_item.worker]) > 0
        return worker_queues[work_item.worker][0] == work_item

    while any(len(v) > 0 for v in worker_queues.values()):
        progress = False
        for worker, work_items in worker_queues.items():
            if len(work_items) == 0:
                continue
            if not all_processed(all_input_events(work_items[0])):
                continue
                
            if not all([is_first_in_queue(i) for i in all_tasks_involved(work_items[0])]):
                continue

            progress = True
            input_events = [processed_events[x] for x in all_input_events(work_items[0])]
            stream_events = []
            for cur_worker in all_workers_involved(work_items[0]):
                if len(worker_events[cur_worker]) > 0:
                    stream_events.append(worker_events[cur_worker][-1])
            start_time = max([i.end for i in input_events + stream_events], default=0)
            # if start_time >= 5e9 and "comms" in str(worker):
            #     import ipdb; ipdb.set_trace()
            all_workers = all_workers_involved(work_items[0])
            for cur_worker in all_workers:
                cur_task = worker_queues[cur_worker].pop(0) 
                new_event = Event(start_time, cur_task.runtime, cur_task.meta)
                worker_events[cur_worker].append(new_event)
                for output_id in cur_task.outputs:
                    processed_events[output_id] = new_event

        if not progress:
            import ipdb; ipdb.set_trace()
            raise RuntimeError(f"Stuck. {worker_queues}")
        

    return worker_events

def chrome_events(worker_events):
    trace = []
    for worker, events in worker_events.items():
        for event in events:
            trace.append({
                "name": event.meta,
                "cat": "compute",
                "ph": "X",
                "ts": event.start * 1000,
                "dur": event.runtime * 1000,
                "pid": worker,
                "tid": worker
            })
    import json
    with open("trace.json", "w") as f:
        json.dump({"traceEvents": trace}, f, indent=4)

def visualize_events(worker_events):
    import pandas as pd
    import plotly.graph_objs as go

    # Convert the data to a DataFrame
    records = []
    for key, events in worker_events.items():
        for event in events:
            records.append({
                'Process': key,
                'Event': event.meta,
                'Start': event.start,
                'End': event.end,
                'Duration': event.end - event.start
            })
    df = pd.DataFrame(records)

    # Create Gantt chart using plotly.graph_objs
    fig = go.Figure()

    fw_list = [
        "#0000FF",  # Blue
        "#1E90FF",  # Dodger Blue
        "#00BFFF",  # Deep Sky Blue
        "#5F9EA0",  # Cadet Blue
        "#4682B4",  # Steel Blue
        "#87CEFA",  # Light Sky Blue
        "#6495ED",  # Cornflower Blue
        "#4169E1"   # Royal Blue
    ]
    bw_list = [
        "#FF0000",  # Red
        "#FF4500",  # Orange Red
        "#FF1493",  # Deep Pink
        "#FF69B4",  # Hot Pink
        "#DB7093",  # Pale Violet Red
        "#B22222",  # Firebrick
        "#8B0000",  # Dark Red
        "#FF6347"   # Tomato
    ]
    
    # Map each event to a color
    
    def get_color(metas):
        color = None
        if "collective" in metas:
            return "green"

        if "fw" in metas:
            for meta in metas:
                if meta.isdigit():
                    return fw_list[int(meta) % len(fw_list)]
        elif "bw" in metas:
            for meta in metas:
                if meta.isdigit():
                    return bw_list[int(meta) % len(fw_list)]
        return "red"


    for process in df['Process'].unique():
        process_df = df[df['Process'] == process]
        for _, row in process_df.iterrows():
            color = get_color(row['Event'])
            fig.add_trace(go.Bar(
                x=[row['Duration']],
                y=[str(process)],
                base=[row['Start']],
                orientation='h',
                name=' '.join(row['Event']),
                hoverinfo='name+x',
                marker=dict(
                    color=color,
                ),
                showlegend=False  # Hide default legend
            ))

    # Add custom legend
    annotations = []
    legend_x = 0.95
    legend_y = 1.0

    fig.update_layout(
        title='Timeline Visualization',
        xaxis_title='Time',
        yaxis_title='Process',
        barmode='stack',
        # annotations=annotations,
        showlegend=False,  # Disable the default legend
        yaxis=dict(
            autorange='reversed'  # Reverse the y-axis
        )
    )

    # Show the plot
    fig.show()

def analyze_events(worker_events):
    start_time = 0
    end_time = 0
    for worker in worker_events.values():
        for event in worker:
            end_time = max(end_time, event.end)
    used_time = 0
    for event in worker_events[0]:
        used_time += event.runtime
    print(f"End Time: ", end_time)
    print(f"Used Time: {100 * used_time/end_time:.2f}%")
    print(f"Bubble size: {100 - 100 * used_time/end_time:.2f}%")
        
from .messages import CallFunction, SendTensor, DeleteRefs, CommandGroup, BorrowCreate, Reduce, BorrowFirstUse, BorrowDrop

META_VAL = []

@contextlib.contextmanager
def set_meta(new_value):
    # Sets the metadata for any tasks created under this
    global META_VAL
    META_VAL.append(new_value)
    try:
        yield
    finally:
        META_VAL.pop()

def get_fake_tensor(x):
    if isinstance(x, torch.Tensor):
        return x._fake
    return x
from torch.utils._pytree import tree_map

def get_runtime(msg):
    if isinstance(msg, CallFunction):
        if msg.function.path == 'torch.ops.aten.mm.default':
            from torch.utils.flop_counter import FlopCounterMode
            cur_args = tree_map(get_fake_tensor, msg.args)
            with FlopCounterMode(display=False) as mode:
                torch.ops.aten.mm.default(*cur_args)
            counted_flops = mode.get_total_flops()
            return counted_flops
        elif msg.function.path == 'torch.ops.aten.addmm.default':
            from torch.utils.flop_counter import FlopCounterMode
            cur_args = tree_map(get_fake_tensor, msg.args)
            with FlopCounterMode(display=False) as mode:
                torch.ops.aten.addmm.default(*cur_args)
            counted_flops = mode.get_total_flops()
            return counted_flops

        return 0
    if isinstance(msg, SendTensor):
        if msg.from_ranks == msg.to_ranks:
            return 0
        return 0
    if isinstance(msg, Reduce):
        return get_fake_tensor(msg.local_tensor_ref).numel() * COLLECTIVE_SPEED
    raise NotImplementedError()

class Task:
    def __init__(self, inputs, outputs, runtime: float, meta, worker, collectives=None):
        self.inputs = inputs
        self.outputs = outputs
        self.runtime = runtime
        self.meta = [meta] + META_VAL
        self.worker = worker
        self.collectives = collectives
        if self.collectives is not None:
            self.collectives.append(self)

    def __repr__(self):
        return f"T({self.runtime})"

from torch.utils._pytree import tree_leaves
BORROW_ID = -1
COLLECTIVE_SPEED = 1e3
class SimulatorBackend(MockBackend):
    def __init__(self, world_size: int, collective_speed=1e3):
        super().__init__(world_size, verbose = False)
        self.worker_commands = defaultdict(list)
        self.borrow_last_use = {}
        global COLLECTIVE_SPEED
        COLLECTIVE_SPEED = collective_speed

    def send(self, ranks: Union[NDSlice, List[NDSlice]], msg) -> None:
        super().send(ranks, msg)
        # breakpoint()
        if isinstance(ranks, NDSlice):
            ranks = [ranks]

        def get_ids(tree):
            ids = []
            for arg in tree_leaves(tree):
                if isinstance(arg, torch.Tensor):
                    ids.append(arg.ref)
            return ids

        def process_msg(msg):
            if isinstance(msg, CallFunction):
                inputs = get_ids(msg.args)
                outputs = get_ids(msg.results)
                meta = f"call: {msg.function.path}"
                def track_borrow_last_use(x):
                    if x in self.borrow_last_use:
                        self.borrow_last_use[x] = outputs[0]
                tree_map(track_borrow_last_use, inputs)
                for rank in ranks:
                    for worker in rank:
                        self.worker_commands[(worker, msg.stream.name)].append(Task(inputs=inputs, outputs=outputs, runtime=get_runtime(msg), meta=meta, worker=(worker, msg.stream.name)))
            elif isinstance(msg, SendTensor):
                inputs = get_ids(msg.tensor)
                outputs = get_ids(msg.result)
                meta = "send"
                collectives = []
                if msg.from_ranks == msg.to_ranks:
                    collectives = None
                for rank in ranks:
                    for worker in rank:
                        self.worker_commands[worker].append(Task(inputs=inputs, outputs=outputs, runtime=get_runtime(msg), meta=meta, worker=worker,  collectives=collectives))
            elif isinstance(msg, DeleteRefs):
                # not modeling memory for now
                return
            elif isinstance(msg, BorrowCreate):
                inputs = get_ids(msg.tensor)
                outputs = get_ids(msg.result)
                meta="borrow"
                # import ipdb; ipdb.set_trace()
                for rank in ranks:
                    for worker in rank:
                        self.worker_commands[(worker, msg.from_stream.name)].append(Task(inputs=inputs, outputs=outputs, runtime=0, meta=meta, worker=(worker, msg.from_stream.name)))
                return
            elif isinstance(msg, Reduce):
                inputs = get_ids(msg.local_tensor_ref)
                outputs = get_ids(msg.result)
                meta = "collective"
                collectives = []
                for rank in ranks:
                    for worker in rank:
                        self.worker_commands[(worker, msg.stream_ref.name)].append(Task(inputs=inputs, outputs=outputs, runtime=get_runtime(msg), meta=meta, worker=(worker, msg.stream_ref.name), collectives=collectives))
                return
            elif isinstance(msg, BorrowFirstUse):
                global BORROW_ID
                meta = "borrow_first_use"
                # for rank in ranks:
                #     for worker in rank:
                #         self.worker_commands[(worker, msg.to_stream.name)].append(Task(inputs=[msg.borrow], outputs=[BORROW_ID], runtime=0, meta=meta, worker=(worker, msg.to_stream.name)))
                self.borrow_last_use[msg.borrow] = 0
                # BORROW_ID -= 1 
                return
            elif isinstance(msg, BorrowDrop):
                meta="borrow_drop"
                for rank in ranks:
                    for worker in rank:
                        # self.worker_commands[(worker, msg.to_stream.name)].append(Task(inputs=[], outputs=[BORROW_ID], runtime=0, meta=meta, worker=(worker, msg.to_stream.name)))
                        if msg.borrow in self.borrow_last_use:
                            # import ipdb; ipdb.set_trace()
                            self.worker_commands[(worker, msg.from_stream.name)].append(Task(inputs=[self.borrow_last_use[msg.borrow]], outputs=[], runtime=0, meta=meta, worker=(worker, msg.from_stream.name)))
                return
            else:
                print(msg)
                raise NotImplementedError()

        if isinstance(msg, CommandGroup):
            for msg in msg.commands:
                process_msg(msg)
        else:
            process_msg(msg)
        # print(f"rank: {ranks}")
        # print(f"msg: {msg}")
        # print()

    def recvready(self, timeout: Optional[float]) -> Sequence[Tuple[int, NamedTuple]]:
        r = tuple(self.responses)
        self.responses.clear()
        return r

    def recv(self, timeout: Optional[float]) -> Tuple[int, NamedTuple]:
        if self.responses:
            return self.responses.popleft()
        raise TimeoutError()
