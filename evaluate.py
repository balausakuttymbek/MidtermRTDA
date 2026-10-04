"""Compare detected spikes with the injected ground truth."""
import pandas as pd

METRIC_OF = {"co2_spike": "co2", "temp_spike": "temperature"}


def evaluate(truth_csv, events_df):
    truth = pd.read_csv(truth_csv, parse_dates=["timestamp"])
    truth = truth[truth["kind"].isin(METRIC_OF)]
    true_keys = {(r.room, r.timestamp, METRIC_OF[r.kind]) for r in truth.itertuples()}
    z = events_df[events_df["rule"] == "ZSCORE_SPIKE"] if len(events_df) else events_df
    found = {(r.room, r.timestamp, r.metric) for r in z.itertuples()} if len(z) else set()
    tp, fn, fp = len(true_keys & found), len(true_keys - found), len(found - true_keys)
    return {"injected_spikes": len(true_keys), "detected_true": tp, "missed": fn,
            "false_alarms": fp,
            "precision": round(tp / (tp + fp), 3) if tp + fp else None,
            "recall": round(tp / (tp + fn), 3) if tp + fn else None}
