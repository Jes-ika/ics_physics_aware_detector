
import pandas as pd
from src.data_utils import load_dataset, physics_features

df = load_dataset("artifacts/scored_events.csv")
physics = physics_features(df)

idx = df["combined_score"].idxmax()

print("Suspicious event:")
print(df.loc[idx].to_string())

print("\nPhysics features around the event:")
start = max(0, idx - 3)
end = min(len(df), idx + 4)

print(
    pd.concat(
        [df.loc[start:end-1, ["timestamp", "label", "level",
                              "flow", "pressure", "pump_on",
                              "valve_open"]],
         physics.loc[start:end-1]],
        axis=1
    ).to_string(index=False)
)
