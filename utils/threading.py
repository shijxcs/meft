import queue
import threading
from typing import *

import torch
import torch.cuda
from torch import Tensor


class TaskProcessor:
    def __init__(self, device="cuda"):
        self.device = device
        self.running = False
        self.threads = []
        self.streams = []

    def start(self, num_workers: int = 2):
        self.running = True
        self.threads.clear()
        self.streams.clear()
        self.task_queue = queue.Queue(num_workers)

        if not (isinstance(num_workers, int) and num_workers > 0):
            raise TypeError("Invalid type of `num_workers`, must be a positive integer.")

        for _ in range(num_workers):
            stream = torch.cuda.Stream(device=self.device)
            thread = threading.Thread(
                target=self._worker_loop,
                args=(stream,),
                daemon=True,
            )
            thread.start()
            self.streams.append(stream)
            self.threads.append(thread)

    def _worker_loop(self, stream: torch.cuda.Stream):
        while self.running:
            try:
                func, args, kwargs, outputs = self.task_queue.get(timeout=0.1)
                try:
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        results = func(*args, **kwargs)

                        if isinstance(results, tuple):
                            for out, res in zip(outputs, results):
                                if isinstance(res, Tensor):
                                    out.copy_(res, non_blocking=True)
                        elif isinstance(results, Tensor):
                            outputs.copy_(results, non_blocking=True)

                except Exception as e:
                    print(f"Error processing task: {e}")

                finally:
                    self.task_queue.task_done()

            except queue.Empty:
                continue

    def submit(self, func: Callable, args: Tuple = None, kwargs: Mapping[str, Any] = None, outputs: Any = None):
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}
        self.task_queue.put((func, args, kwargs, outputs))

    def join(self):
        self.task_queue.join()
        for stream in self.streams:
            torch.cuda.current_stream().wait_stream(stream)

    def stop(self):
        self.running = False
        for thread in self.threads:
            thread.join()
