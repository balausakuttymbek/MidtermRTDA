"""Persist processed rows, events and run summary."""
import json
import os

import pandas as pd

from processing import OUT_COLS


class ResultStore:
    def __init__(self, out_dir="results", reset=True):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.processed_path = os.path.join(out_dir, "processed.csv")
        self.events_path = os.path.join(out_dir, "events.csv")
        if reset:
            for p in (self.processed_path, self.events_path):
                if os.path.exists(p):
                    os.remove(p)

    def _append(self, df, path):
        df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)

    def append_processed(self, df):
        if len(df):
            self._append(df[OUT_COLS], self.processed_path)

    def append_events(self, df):
        if len(df):
            self._append(df, self.events_path)

    def save_summary(self, summary):
        with open(os.path.join(self.out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
