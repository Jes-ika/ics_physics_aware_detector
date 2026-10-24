from pathlib import Path
import pandas as pd

FEATURES = ["level", "flow", "pressure", "pump_on", "valve_open"]
CONTINUOUS = ["level", "flow", "pressure"]
ACTUATORS = ["pump_on", "valve_open"]

def load_dataset(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"timestamp", "label", *FEATURES}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", *FEATURES]).sort_values("timestamp").reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    return df

def physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Illustrative consistency features. Replace with validated process equations for real ICS."""
    out = pd.DataFrame(index=df.index)
    # Level dynamics should broadly agree with inflow/outflow and pump/valve state.
    out["level_delta"] = df["level"].diff().fillna(0)
    out["flow_delta"] = df["flow"].diff().fillna(0)
    out["pressure_delta"] = df["pressure"].diff().fillna(0)
    # Heuristic residual: flow is expected to increase when the pump is on and valve is open.
    expected_flow = 0.15 + 0.55 * df["pump_on"] * df["valve_open"]
    out["flow_actuator_residual"] = (df["flow"] - expected_flow).abs()
    # Heuristic residual: pressure broadly tracks flow in this synthetic process.
    expected_pressure = 0.2 + 0.65 * df["flow"]
    out["pressure_flow_residual"] = (df["pressure"] - expected_pressure).abs()
    # A large level change without corresponding flow change can be suspicious.
    out["level_flow_mismatch"] = (out["level_delta"] - 0.08 * out["flow_delta"]).abs()
    return out.replace([float("inf"), float("-inf")], 0).fillna(0)
