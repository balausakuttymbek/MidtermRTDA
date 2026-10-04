"""Cleaning + real-time indicators with Pandas / NumPy."""
import numpy as np
import pandas as pd

METRICS = ["temperature", "co2", "humidity"]
VALID_RANGE = {"temperature": (-10, 50), "co2": (300, 5000), "humidity": (0, 100)}
STD_FLOOR = {"temperature": 0.3, "co2": 10.0, "humidity": 1.5}   # ~sensor noise; avoids huge z on quiet windows
DERIVED = ["ma", "std", "min", "max", "z", "diff"]
OUT_COLS = (["timestamp", "room"] + METRICS
            + [f"{m}_{d}" for m in METRICS for d in DERIVED] + ["imputed"])


class StreamProcessor:
    """Keeps a sliding buffer per room and computes indicators for each new micro-batch."""

    def __init__(self, window=60, roll=12):
        self.window = window          # records kept per room
        self.roll = roll              # rolling window for indicators
        self.buffer = pd.DataFrame()
        self.seen = set()             # (room, timestamp) already processed -> duplicate filter
        self.seq = 0
        self.stats = {"received": 0, "duplicates": 0, "missing_values": 0, "invalid_values": 0}

    # ---------- cleaning ----------
    def clean(self, batch):
        df = pd.DataFrame(batch).reindex(columns=["timestamp", "room"] + METRICS)
        self.stats["received"] += len(df)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp", "room"])

        keys = list(zip(df["room"], df["timestamp"]))
        fresh = np.array([k not in self.seen for k in keys], dtype=bool)
        keep = fresh & ~df.duplicated(subset=["room", "timestamp"]).to_numpy()
        self.stats["duplicates"] += int((~keep).sum())
        self.seen.update(k for k, ok in zip(keys, keep) if ok)
        df = df[keep].copy()

        for m in METRICS:
            df[m] = pd.to_numeric(df[m], errors="coerce")
            lo, hi = VALID_RANGE[m]
            self.stats["missing_values"] += int(df[m].isna().sum())
            bad = df[m].notna() & ~df[m].between(lo, hi)
            self.stats["invalid_values"] += int(bad.sum())
            df.loc[bad, m] = np.nan
        df["imputed"] = df[METRICS].isna().any(axis=1)
        return df

    # ---------- indicators ----------
    def _roll(self, buf, m, stat, min_periods, shift=0):
        def f(s):
            out = getattr(s.rolling(self.roll, min_periods=min_periods), stat)()
            return out.shift(shift) if shift else out
        return buf.groupby("room")[m].transform(f)

    def update(self, batch):
        """Process one micro-batch; return the new rows enriched with indicators."""
        new = self.clean(batch)
        if new.empty:
            return new
        first_seq = self.seq
        new["seq"] = np.arange(self.seq, self.seq + len(new))
        self.seq += len(new)

        buf = pd.concat([self.buffer, new], ignore_index=True)
        buf = buf.sort_values(["room", "timestamp"])
        buf[METRICS] = buf.groupby("room")[METRICS].ffill()          # fill gaps with last valid value
        buf = buf.groupby("room", group_keys=False).tail(self.window)

        for m in METRICS:
            buf[f"{m}_ma"] = self._roll(buf, m, "mean", 3)
            buf[f"{m}_std"] = self._roll(buf, m, "std", 3)
            buf[f"{m}_min"] = self._roll(buf, m, "min", 1)
            buf[f"{m}_max"] = self._roll(buf, m, "max", 1)
            prev_mean = self._roll(buf, m, "mean", 5, shift=1)       # stats of PREVIOUS readings
            prev_std = self._roll(buf, m, "std", 5, shift=1).clip(lower=STD_FLOOR[m])
            buf[f"{m}_z"] = (buf[m] - prev_mean) / prev_std
            buf[f"{m}_diff"] = buf.groupby("room")[m].diff()

        self.buffer = buf
        out = buf[buf["seq"] >= first_seq].sort_values(["timestamp", "room"])
        return out[OUT_COLS + ["seq"]]

    def rate_per_min(self, window_s=300):
        """Event rate: valid records per minute over the last `window_s` of stream time."""
        rates = {}
        for room, g in self.buffer.groupby("room"):
            t = g["timestamp"]
            rates[room] = (t > t.max() - pd.Timedelta(seconds=window_s)).sum() / (window_s / 60)
        return rates

    def latest(self):
        """Current per-room snapshot for the console dashboard."""
        last = self.buffer.sort_values("timestamp").groupby("room").tail(1).set_index("room")
        snap = last[["timestamp", "temperature", "co2", "humidity",
                     "co2_ma", "co2_std", "co2_min", "co2_max"]].copy()
        snap["rate/min"] = pd.Series(self.rate_per_min())
        snap["timestamp"] = snap["timestamp"].dt.strftime("%H:%M:%S")
        return snap.reset_index()
