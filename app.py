from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="ICS Physics-Aware Detector", layout="wide")
st.title("ICS Physics-Aware Anomaly Detection")
st.caption("Research prototype — synthetic demo results are not evidence of real-world detection performance.")

metrics_path = Path("artifacts/metrics.json")
scores_path = Path("artifacts/scored_events.csv")
if not metrics_path.exists() or not scores_path.exists():
    st.warning("No trained artifacts found yet. Run the setup and training commands in README.md first.")
    st.stop()

metrics = json.loads(metrics_path.read_text())
df = pd.read_csv(scores_path)
df["timestamp"] = pd.to_datetime(df["timestamp"])
threshold = metrics["proposed_model_test"]["threshold"]
alerts = df[df["alert"] == 1]

# Evaluate only the held-out test period.
test_start = int(len(df) * 0.75)
test_df = df.iloc[test_start:].copy()

test_alerts = int(test_df["alert"].sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows analysed", f"{len(df):,}")
c2.metric("Test alerts", f"{test_alerts:,}")
c3.metric("Test F1", f"{metrics['proposed_model_test']['f1']:.3f}")
c4.metric("Test recall", f"{metrics['proposed_model_test']['recall']:.3f}")

st.caption(
    f"Metrics are evaluated on the held-out test set "
    f"({len(test_df):,} rows), not on the entire dataset."
)

st.subheader("Sensor and actuator trends")
signals = st.multiselect("Signals", ["level", "flow", "pressure", "pump_on", "valve_open"],
                         default=["level", "flow", "pressure"])
fig = px.line(df, x="timestamp", y=signals, title="Recorded ICS signals")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Anomaly score")
score_fig = px.line(df, x="timestamp", y="combined_score", title="Combined temporal + physics-inspired score")
score_fig.add_hline(y=threshold, line_dash="dash", annotation_text="Validation-calibrated threshold")
st.plotly_chart(score_fig, use_container_width=True)

st.subheader("Alerts and evidence")
show_anomalies = st.checkbox("Show only labelled synthetic attack windows", value=False)
view = df[df["label"] == 1] if show_anomalies else alerts
st.dataframe(view[["timestamp", "label", "level", "flow", "pressure", "pump_on",
                   "valve_open", "temporal_score", "physics_score", "combined_score", "alert"]].head(500),
             use_container_width=True, hide_index=True)
st.download_button("Download scored events CSV", data=df.to_csv(index=False),
                   file_name="scored_events.csv", mime="text/csv")

with st.expander("Research metrics"):
    st.json(metrics)
