"""Smart-campus sensor stream generator.

Simulates classroom sensors (temperature, CO2, humidity) and appends
JSON-lines records to a file one "tick" at a time, like a live source.
Dirty data and anomalies are injected on purpose; ground truth is saved
so the detector can be evaluated later.
"""
import json
import math
import time
from datetime import datetime, timedelta

import numpy as np

STEP_SECONDS = 10                       # one reading per room every 10 s (simulated time)
START = datetime(2026, 10, 1, 8, 0, 0)

# room -> list of (start_step, end_step, people) lessons
ROOMS = {
    "A101": [(60, 400, 28)],
    "A102": [(200, 500, 15)],
    "B201": [(100, 600, 8)],
}


def _occupancy(room, step):
    return sum(n for a, b, n in ROOMS[room] if a <= step < b)


def _heating_fault(room, step):
    """Slow temperature drift in A102 (HVAC failure) - caught by threshold rule, not z-score."""
    if room != "A102":
        return 0.0
    if 520 <= step < 640:
        return 0.08 * (step - 520)
    if 640 <= step < 700:
        return 9.6 - 0.16 * (step - 640)
    return 0.0


def generate_ticks(n_steps=720, seed=42, start=START):
    """Yield (list_of_records, list_of_injected_truth) for each time step."""
    rng = np.random.default_rng(seed)
    co2_state = {r: 430.0 for r in ROOMS}
    for step in range(n_steps):
        ts = start + timedelta(seconds=step * STEP_SECONDS)
        tick, truth = [], []
        for room in ROOMS:
            occ = _occupancy(room, step)
            co2_state[room] += (420 + 35 * occ - co2_state[room]) * 0.012 + rng.normal(0, 4)
            temp = (21 + 0.05 * occ + 0.6 * math.sin(2 * math.pi * step / n_steps)
                    + _heating_fault(room, step) + rng.normal(0, 0.1))
            hum = 40 + 0.25 * occ + rng.normal(0, 0.6)
            rec = {"timestamp": ts.isoformat(), "room": room,
                   "temperature": round(temp, 2),
                   "co2": round(co2_state[room], 1),
                   "humidity": round(hum, 1)}

            r = rng.random()
            kind = None
            if r < 0.004:
                kind = "co2_spike"
                rec["co2"] = round(rec["co2"] + rng.uniform(400, 700), 1)
            elif r < 0.007:
                kind = "temp_spike"
                rec["temperature"] = round(rec["temperature"] + rng.uniform(6, 10), 2)
            elif r < 0.011:
                kind = "sensor_fault"          # dirty data: impossible value
                rec["co2"] = -999
            elif r < 0.015:
                kind = "missing_value"         # dirty data: null
                rec["humidity"] = None
            elif r < 0.019:
                kind = "duplicate"             # dirty data: record sent twice
                tick.append(dict(rec))
            if kind:
                truth.append({"timestamp": rec["timestamp"], "room": room, "kind": kind})
            tick.append(rec)
        yield tick, truth


def run_writer(path, n_steps=720, delay=0.02, seed=42, truth_path=None):
    """Append each tick to `path` (JSON lines), sleeping `delay` s between ticks."""
    all_truth = []
    with open(path, "a", encoding="utf-8") as f:
        for tick, truth in generate_ticks(n_steps, seed):
            for rec in tick:
                f.write(json.dumps(rec) + "\n")
            f.flush()
            all_truth.extend(truth)
            if delay:
                time.sleep(delay)
    if truth_path:
        import pandas as pd
        pd.DataFrame(all_truth, columns=["timestamp", "room", "kind"]).to_csv(truth_path, index=False)
