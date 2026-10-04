"""Entry point: generator thread -> incremental acquisition -> processing -> detection -> storage."""
import argparse
import os
import threading
import time

import pandas as pd

from acquisition import tail_jsonl
from dashboard import make_assets, print_events, print_status
from detector import AnomalyDetector
from evaluate import evaluate
from generator import run_writer
from processing import StreamProcessor
from storage import ResultStore


def parse_args():
    p = argparse.ArgumentParser(description="Smart-campus real-time stream processing")
    p.add_argument("--steps", type=int, default=720, help="time steps (each = 10 s, 3 rooms)")
    p.add_argument("--delay", type=float, default=0.02, help="seconds between ticks of the live source")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data", default="data/stream.jsonl")
    p.add_argument("--out", default="results")
    p.add_argument("--window", type=int, default=60, help="records kept per room")
    p.add_argument("--roll", type=int, default=12, help="rolling window for indicators")
    p.add_argument("--print-every", type=int, default=10, help="print dashboard every N batches")
    p.add_argument("--no-source", action="store_true", help="do not start generator; read existing --data file")
    return p.parse_args()


def main():
    a = parse_args()
    os.makedirs(os.path.dirname(a.data) or ".", exist_ok=True)
    truth_path = os.path.join(os.path.dirname(a.data) or ".", "injected_anomalies.csv")
    store = ResultStore(a.out)
    proc, det = StreamProcessor(a.window, a.roll), AnomalyDetector()

    if not a.no_source:
        if os.path.exists(a.data):
            os.remove(a.data)
        threading.Thread(target=run_writer, args=(a.data, a.steps, a.delay, a.seed, truth_path),
                         daemon=True).start()

    t_start, all_events, batches = time.perf_counter(), [], 0
    for batch in tail_jsonl(a.data):
        t0 = time.perf_counter()
        new = proc.update(batch)
        events = det.detect(new) if len(new) else pd.DataFrame()
        elapsed = time.perf_counter() - t0
        store.append_processed(new)
        store.append_events(events)
        batches += 1
        if len(events):
            all_events.append(events)
            print_events(events)
        if batches % a.print_every == 0:
            print_status(batches, len(batch), len(batch) / max(elapsed, 1e-9), elapsed * 1000, proc)

    total = time.perf_counter() - t_start
    events_df = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame(
        columns=["timestamp", "room", "rule", "metric", "value", "severity", "detail"])
    summary = {"batches": batches, "valid_records": proc.seq, "cleaning": proc.stats,
               "events_total": len(events_df),
               "events_by_rule": events_df["rule"].value_counts().to_dict() if len(events_df) else {},
               "wall_time_s": round(total, 2)}
    if os.path.exists(truth_path):
        summary["detection_vs_ground_truth"] = evaluate(truth_path, events_df)
    store.save_summary(summary)
    make_assets(a.out)
    print("\n=== DONE ===")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
