# Research Plan

## Research question
Can self-supervised temporal representations combined with physics-inspired consistency features improve detection of held-out cyber-physical attack scenarios in industrial control systems, while controlling false alarms during legitimate operating-mode changes?

## Proposed hypotheses
- H1: The combined temporal + physics-aware score performs better than an Isolation Forest baseline on held-out scenarios.
- H2: Physics-aware residuals provide incremental value over temporal prediction error alone.
- H3: The detector can produce traceable alerts with practical inference latency.

These are hypotheses, not assumed outcomes.

## Required experiments
1. Isolation Forest baseline.
2. Temporal forecasting model without physics features.
3. Combined temporal + physics-inspired score.
4. Ablation: remove physics score.
5. Threshold sensitivity analysis using normal validation data only.
6. Hold out complete attack runs/scenarios where dataset labels permit.
7. Test legitimate operating-mode transitions separately.
8. Report precision, recall, F1, PR-AUC, false alarms per hour, detection delay, runtime and memory.

## Important limitations
- Synthetic dataset is for engineering smoke tests only.
- Physics residuals in the demo are illustrative heuristics, not validated plant equations.
- Public datasets can contain leakage risks, class imbalance, limited scenarios and benchmark-specific artifacts.
- "Unseen attack" means attack scenarios withheld from training in the defined experiment, not guaranteed zero-day detection.
