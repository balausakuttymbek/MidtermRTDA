"""Console output and result assets (tables + figures)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def print_status(batch_no, n_batch, throughput, latency_ms, processor):
    snap = processor.latest()
    s = processor.stats
    print(f"\n=== batch {batch_no} | +{n_batch} records | {throughput:,.0f} rec/s | "
          f"latency {latency_ms:.1f} ms | total valid {processor.seq} | dup {s['duplicates']} "
          f"| missing {s['missing_values']} | invalid {s['invalid_values']}")
    print(snap.to_string(index=False, float_format=lambda x: f"{x:.1f}"))


def print_events(events):
    for r in events.itertuples():
        print(f"  [ALERT {r.severity.upper():8}] {r.timestamp:%H:%M:%S} {r.room} "
              f"{r.rule} {r.metric}={r.value:.1f} ({r.detail})")


def _md(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def make_assets(out_dir):
    proc = pd.read_csv(f"{out_dir}/processed.csv", parse_dates=["timestamp"])
    try:
        ev = pd.read_csv(f"{out_dir}/events.csv", parse_dates=["timestamp"])
    except (FileNotFoundError, pd.errors.EmptyDataError):
        ev = pd.DataFrame(columns=["timestamp", "room", "rule", "metric", "value", "severity", "detail"])

    # table 1: statistics per room / metric
    rows = []
    for room, g in proc.groupby("room"):
        for m in ["temperature", "co2", "humidity"]:
            rows.append({"room": room, "metric": m, "n": len(g), "mean": round(g[m].mean(), 1),
                         "std": round(g[m].std(), 1), "min": round(g[m].min(), 1),
                         "max": round(g[m].max(), 1)})
    stats = pd.DataFrame(rows)
    stats.to_csv(f"{out_dir}/table_statistics.csv", index=False)
    open(f"{out_dir}/table_statistics.md", "w").write(_md(stats))

    # table 2: events by rule and room
    if len(ev):
        by_rule = pd.crosstab(ev["rule"], ev["room"]).reset_index()
        by_rule.to_csv(f"{out_dir}/table_events.csv", index=False)
        open(f"{out_dir}/table_events.md", "w").write(_md(by_rule))

    # figure 1: A101 timeline, figure 2: A102 timeline (temperature fault)
    for room in ["A101", "A102"]:
        g = proc[proc["room"] == room]
        e = ev[ev["room"] == room]
        fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for a, m, unit, limits in zip(ax, ["co2", "temperature", "humidity"],
                                      ["ppm", "°C", "%"], [(1000, 1200), (18, 28), (20, 70)]):
            a.plot(g["timestamp"], g[m], color="lightgray", lw=0.8, label="raw")
            a.plot(g["timestamp"], g[f"{m}_ma"], color="tab:blue", lw=1.6, label="moving average")
            for lim in limits:
                a.axhline(lim, color="tab:orange", ls="--", lw=0.8)
            em = e[e["metric"] == m]
            sp = em[em["rule"] == "ZSCORE_SPIKE"]
            st = em[em["rule"] != "ZSCORE_SPIKE"]
            a.scatter(sp["timestamp"], sp["value"], color="red", marker="x", s=50, label="z-score spike", zorder=5)
            a.scatter(st["timestamp"], st["value"], color="darkorange", marker="v", s=50, label="threshold alert", zorder=5)
            a.set_ylabel(f"{m} ({unit})")
        ax[0].set_title(f"Room {room}: real-time indicators and detected events")
        ax[0].legend(loc="upper left", fontsize=8)
        ax[-1].set_xlabel("time")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(f"{out_dir}/fig_{room}_timeline.png", dpi=150)
        plt.close(fig)

    # figure 3: CO2 moving average for all rooms
    fig, a = plt.subplots(figsize=(10, 4))
    for room, g in proc.groupby("room"):
        a.plot(g["timestamp"], g["co2_ma"], label=room)
    a.axhline(1000, color="tab:orange", ls="--", lw=0.8)
    a.set_ylabel("CO2 moving average (ppm)")
    a.set_title("CO2 by room")
    a.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(f"{out_dir}/fig_co2_all_rooms.png", dpi=150)
    plt.close(fig)
