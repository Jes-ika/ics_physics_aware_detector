import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def generate(n=6000, seed=17):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2025-01-01", periods=n, freq="s")
    pump = np.zeros(n, dtype=int)
    valve = np.zeros(n, dtype=float)
    level = np.zeros(n)
    flow = np.zeros(n)
    pressure = np.zeros(n)
    label = np.zeros(n, dtype=int)

    level[0] = 0.5
    valve[:] = 0.75
    for i in range(n):
        # Repeating operating cycles, creating legitimate operating-mode changes.
        pump[i] = int((i // 450) % 2 == 0)
        if (i // 900) % 2 == 1:
            valve[i] = 0.45
        else:
            valve[i] = 0.8

        expected_flow = 0.12 + 0.58 * pump[i] * valve[i]
        flow[i] = np.clip(expected_flow + rng.normal(0, 0.025), 0, 1)
        expected_pressure = 0.2 + 0.62 * flow[i]
        pressure[i] = np.clip(expected_pressure + rng.normal(0, 0.025), 0, 1)
        if i > 0:
            level[i] = np.clip(level[i-1] + 0.025 * (flow[i] - 0.28) + rng.normal(0, 0.004), 0, 1)

    # Three injected synthetic scenarios, held out conceptually from normal training:
    # 1) manipulated level sensor; 2) actuator/flow inconsistency; 3) pressure anomaly.
    attacks = [(3900, 4150, "level"), (4550, 4800, "flow"), (5200, 5400, "pressure")]
    for start, end, kind in attacks:
        label[start:end] = 1
        if kind == "level":
            level[start:end] = np.clip(level[start:end] + np.linspace(0.18, 0.38, end-start), 0, 1)
        elif kind == "flow":
            flow[start:end] = np.clip(flow[start:end] * 0.25, 0, 1)
        elif kind == "pressure":
            pressure[start:end] = np.clip(pressure[start:end] + 0.35, 0, 1)

    return pd.DataFrame({
        "timestamp": ts, "label": label, "level": level, "flow": flow,
        "pressure": pressure, "pump_on": pump, "valve_open": valve
    })

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/demo_ics.csv")
    parser.add_argument("--rows", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate(args.rows, args.seed)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} rows to {out.resolve()}")
    print(f"Label counts: {df['label'].value_counts().to_dict()}")

if __name__ == "__main__":
    main()
