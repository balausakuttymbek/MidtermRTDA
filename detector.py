"""Rule-based event / anomaly detection."""
import pandas as pd

from processing import METRICS


class AnomalyDetector:
    """Rules:
      1. ZSCORE_SPIKE      - |z| >= z_thr versus the previous rolling window (any metric, point anomaly)
      2. CO2_WARNING       - CO2 >= 1000 ppm  (state rule, hysteresis)
      3. CO2_CRITICAL      - CO2 >= 1200 ppm
      4. TEMP_OUT_OF_RANGE - temperature outside 18-28 C
      5. HUMIDITY_OUT_OF_RANGE - humidity outside 20-70 %
    State rules fire once when the condition holds for `min_consec` consecutive
    readings and re-arm only after the value returns past a hysteresis margin.
    """

    def __init__(self, z_thr=5.0, min_consec=3):
        self.z_thr = z_thr
        self.min_consec = min_consec
        self.active, self.count = set(), {}
        self.rules = {
            "CO2_WARNING": ("co2", lambda v: v >= 1000, lambda v: v < 970, "warning"),
            "CO2_CRITICAL": ("co2", lambda v: v >= 1200, lambda v: v < 1170, "critical"),
            "TEMP_OUT_OF_RANGE": ("temperature", lambda v: v < 18 or v > 28,
                                  lambda v: 18.3 <= v <= 27.7, "warning"),
            "HUMIDITY_OUT_OF_RANGE": ("humidity", lambda v: v < 20 or v > 70,
                                      lambda v: 21 <= v <= 69, "warning"),
        }

    def detect(self, df):
        events = []
        for row in df.sort_values("timestamp").itertuples(index=False):
            # point rule
            for m in METRICS:
                z = getattr(row, f"{m}_z")
                if pd.notna(z) and abs(z) >= self.z_thr:
                    events.append((row.timestamp, row.room, "ZSCORE_SPIKE", m,
                                   getattr(row, m), "critical", f"z={z:.1f}"))
            # state rules
            for rule, (metric, is_on, is_off, severity) in self.rules.items():
                v = getattr(row, metric)
                key = (row.room, rule)
                if pd.isna(v):
                    continue
                if is_on(v):
                    self.count[key] = self.count.get(key, 0) + 1
                    if self.count[key] >= self.min_consec and key not in self.active:
                        self.active.add(key)
                        events.append((row.timestamp, row.room, rule, metric, v, severity,
                                       f">= {self.min_consec} consecutive readings"))
                elif is_off(v):
                    self.count[key] = 0
                    self.active.discard(key)
                elif key not in self.active:
                    self.count[key] = 0
        return pd.DataFrame(events, columns=["timestamp", "room", "rule", "metric",
                                             "value", "severity", "detail"])
