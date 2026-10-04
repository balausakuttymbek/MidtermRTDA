"""Incremental acquisition: tail a JSON-lines file and yield new records as micro-batches."""
import json
import os
import time


def tail_jsonl(path, poll=0.2, idle_timeout=2.0, start_timeout=5.0):
    """Yield lists of new records as they appear in `path`.

    Only complete lines are consumed (a half-written line waits for the next poll).
    Stops when no new data arrived for `idle_timeout` seconds.
    """
    t_wait = time.time()
    while not os.path.exists(path):
        if time.time() - t_wait > start_timeout:
            return
        time.sleep(0.05)

    offset, last_data = 0, time.time()
    while True:
        with open(path, "rb") as f:
            f.seek(offset)
            data = f.read()
        batch = []
        cut = data.rfind(b"\n")
        if cut >= 0:
            offset += cut + 1
            for line in data[:cut + 1].decode("utf-8").splitlines():
                try:
                    batch.append(json.loads(line))
                except json.JSONDecodeError:
                    pass                       # malformed line: skip
        if batch:
            last_data = time.time()
            yield batch
        elif time.time() - last_data > idle_timeout:
            return
        else:
            time.sleep(poll)
