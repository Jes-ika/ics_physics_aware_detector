# ICS Physics-Aware Anomaly Detection

A research prototype for self-supervised, physics-aware detection of anomalous behaviour in industrial control system (ICS) time series.

## Important scope

This repository starts with a **synthetic demo dataset** so the pipeline can run immediately. Synthetic results are only a software smoke test; they are **not evidence** that the method detects real cyber-physical attacks. Replace or supplement the synthetic data with an authorised public dataset such as SWaT or HAI before reporting research results.

## Research workflow

1. Generate a reproducible synthetic sensor/actuator dataset.
2. Train a self-supervised forecasting model using normal training data only.
3. Compute forecast residuals and physics-inspired consistency features.
4. Compare with an Isolation Forest baseline.
5. Evaluate on a time-separated test set containing held-out synthetic attack patterns.
6. View metrics and anomaly scores in a Streamlit dashboard.

## Setup (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.generate_data --output data/demo_ics.csv
python -m src.train --data data/demo_ics.csv
streamlit run app.py
```

If PowerShell blocks activation, use:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.generate_data --output data/demo_ics.csv
.\.venv\Scripts\python.exe -m src.train --data data/demo_ics.csv
.\.venv\Scripts\streamlit.exe run app.py
```

## Dataset format

CSV columns:
- `timestamp`: ordered timestamp
- `label`: 0 = normal, 1 = injected anomaly (evaluation only; never used for model fitting)
- `level`: process-level sensor
- `flow`: process flow sensor
- `pressure`: process pressure sensor
- `pump_on`: actuator state (0 or 1)
- `valve_open`: actuator state (0 to 1)

For a real dataset, edit `src/data_utils.py` to map source columns to the canonical schema. Document the mapping and keep labels out of self-supervised training.

## Research cautions

- Use chronological splits; do not randomly split overlapping time windows.
- Calibrate thresholds on a separate normal validation period, not on the final test labels.
- Report precision, recall, F1, PR-AUC where appropriate, false alarms per hour, and detection delay.
- Hold out entire attack scenarios or attack runs when possible.
- Compare the proposed method to simple baselines.
- Do not claim zero-day detection; say "detection of held-out attack scenarios under the evaluated benchmark."
- Physics features here are illustrative consistency features, not a validated mechanistic model of a real plant.
