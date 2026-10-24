import argparse, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, average_precision_score, roc_auc_score
from src.data_utils import load_dataset, FEATURES, physics_features
from src.models import train_forecaster, forecast_scores, fit_isolation_forest, isolation_scores

def safe_metrics(y, scores, threshold):
    pred = (scores >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    result = {"precision": float(p), "recall": float(r), "f1": float(f1),
              "threshold": float(threshold), "alerts": int(pred.sum())}
    if len(np.unique(y)) > 1:
        result["pr_auc"] = float(average_precision_score(y, scores))
        result["roc_auc"] = float(roc_auc_score(y, scores))
    else:
        result["pr_auc"] = None
        result["roc_auc"] = None
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/demo_ics.csv")
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--output-dir", default="artifacts")
    args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.data)
    n = len(df)
    # Chronological split: first 60% normal training, next 15% normal validation where available,
    # final 25% held-out test. Training excludes all labelled anomalies.
    train_end = int(n * .60)
    val_end = int(n * .75)
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    train_df = train_df[train_df.label == 0]
    if len(train_df) < args.window + 20:
        raise ValueError("Insufficient normal training rows.")

    physics = physics_features(df)
    all_features = pd.concat([df[FEATURES].reset_index(drop=True), physics.reset_index(drop=True)], axis=1)
    # Fit scaling on normal training data only.
    scaler = StandardScaler()
    train_raw = train_df[FEATURES].to_numpy(dtype=np.float32)
    scaler.fit(train_raw)
    values = scaler.transform(df[FEATURES].to_numpy(dtype=np.float32)).astype(np.float32)

    # Train self-supervised predictor only on normal training sequences.
    model = train_forecaster(scaler.transform(train_df[FEATURES].to_numpy(dtype=np.float32)).astype(np.float32),
                             window=args.window, epochs=args.epochs)
    # Score validation/test with rolling windows from each full chronological stream; no model fitting on them.
    idx, temporal_scores, per_feature = forecast_scores(model, values, window=args.window)
    score_frame = pd.DataFrame({"row_idx": idx, "temporal_score": temporal_scores})
    # Physics score scaled using training distribution only.
    phys_cols = list(physics.columns)
    phys_scaler = StandardScaler().fit(physics.iloc[:train_end][phys_cols])
    phys_scaled = phys_scaler.transform(physics[phys_cols])
    physics_score = np.mean(np.square(phys_scaled), axis=1)
    # Diagnostic: identify extreme physics-feature contributions.
    physics_contributions = np.square(phys_scaled)

    print("\nLargest standardized physics-feature contributions:")
    for row_idx in np.argsort(physics_score)[-5:][::-1]:
        feature_idx = int(np.argmax(physics_contributions[row_idx]))
        print(
            f"row={row_idx}, "
            f"feature={phys_cols[feature_idx]}, "
            f"contribution={physics_contributions[row_idx, feature_idx]:.4f}, "
            f"physics_score={physics_score[row_idx]:.4f}"
        )
    # Combine model forecast error with physics-consistency residual score.
    temporal_full = np.full(n, np.nan)
    temporal_full[idx] = temporal_scores
    # Robustly normalise scores using normal validation rows only.
    val_normal = (df.index < val_end) & (df.label.to_numpy() == 0) & np.isfinite(temporal_full)
    def robust_z(x, mask):
        med = np.nanmedian(x[mask])
        mad = np.nanmedian(np.abs(x[mask] - med)) + 1e-8
        return (x - med) / (1.4826 * mad)
    temporal_z = robust_z(temporal_full, val_normal)
    physics_z = robust_z(physics_score, (df.index < val_end) & (df.label.to_numpy() == 0))
    combined = np.nan_to_num(temporal_z, nan=0.0) + np.nan_to_num(physics_z, nan=0.0)
    # Threshold is set from normal validation scores only, not test labels.
    val_scores = combined[(df.index >= train_end) & (df.index < val_end) & (df.label.to_numpy() == 0)]
    threshold = float(np.quantile(val_scores, .99)) if len(val_scores) else float(np.quantile(combined[:train_end], .99))
    valid_test = (df.index >= val_end) & np.isfinite(combined)
    test_metrics = safe_metrics(df.loc[valid_test, "label"].to_numpy(), combined[valid_test], threshold)

    # Isolation Forest baseline on raw + physics features; fit normal training only.
    train_comb = pd.concat([train_df[FEATURES].reset_index(drop=True),
                            physics.loc[train_df.index].reset_index(drop=True)], axis=1).to_numpy()
    baseline_scaler, baseline = fit_isolation_forest(train_comb)
    base_scores_all = isolation_scores(baseline_scaler, baseline, all_features.to_numpy())
    base_val = base_scores_all[(df.index >= train_end) & (df.index < val_end) & (df.label.to_numpy() == 0)]
    base_threshold = float(np.quantile(base_val, .99)) if len(base_val) else float(np.quantile(base_scores_all[:train_end], .99))
    base_metrics = safe_metrics(df.loc[valid_test, "label"].to_numpy(), base_scores_all[valid_test], base_threshold)

    result = {
        "project": "Self-Supervised, Physics-Aware ICS Anomaly Detection",
        "dataset": str(args.data),
        "rows": int(n),
        "train_rows_normal": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "window": args.window,
        "proposed_model_test": test_metrics,
        "isolation_forest_test": base_metrics,
        "note": "Synthetic data results are a pipeline smoke test only; not evidence of real-world attack detection."
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    np.save(out / "combined_scores.npy", combined)
    np.save(out / "isolation_scores.npy", base_scores_all)
    torch.save(model.state_dict(), out / "forecast_model.pt")
    joblib.dump({"feature_scaler": scaler, "physics_scaler": phys_scaler,
                 "baseline_scaler": baseline_scaler, "baseline_model": baseline,
                 "threshold": threshold, "baseline_threshold": base_threshold,
                 "window": args.window, "features": FEATURES}, out / "artifacts.joblib")
    scored = df.copy()
    scored["temporal_score"] = temporal_full
    scored["physics_score"] = physics_score
    scored["combined_score"] = combined
    scored["baseline_score"] = base_scores_all
    scored["alert"] = (combined >= threshold).astype(int)
    scored.to_csv(out / "scored_events.csv", index=False)
    print(json.dumps(result, indent=2))
    print(f"Saved artifacts to {out.resolve()}")

if __name__ == "__main__":
    main()
