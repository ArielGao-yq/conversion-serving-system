# conversion-serving-system

A conversion-prediction model rebuilt as a complete production ML system:
reproducible feature/training pipelines, MLflow experiment tracking,
a FastAPI serving layer in Docker, and drift + latency monitoring.

**Status:** Week 1 — scope, data, baseline.
**Data:** UCI Bank Marketing (public). No proprietary data.

## Baseline results
| Model | AUC | Log-loss | Brier |
|---|---|---|---|
| Logistic regression (baseline) | — | — | — |

## Ship gate (end of Week 4)
- [ ] Pipelines as importable code, tracked in MLflow
- [ ] FastAPI /predict + /health, Dockerized, CI green
- [ ] Drift report + load test (p50/p95 latency) in README
- [ ] One-command run from a clean checkout

## Week 2 — Model comparison

All runs use the frozen temporal split v1 and identical preprocessing
(`src.features.build_preprocessor`); only the classifier changes.
Experiments are tracked in MLflow (experiment: `conversion-serving-system`).
Pre-registered bar for beating the baseline: AUC ≥ 0.768 AND log-loss < 0.570.

| Run | Model | Features | AUC | Log-loss | Brier | Mean predicted prob. |
|---|---|---|---|---|---|---|
| baseline_logreg_v1 | Logistic regression | all 19 | 0.748 | 0.570 | 0.191 | 25.7% |
| gradient_boosting_v1 | HistGradientBoosting (lr 0.1, max_iter 100) | all 19 | 0.634 | 0.826 | 0.253 | 10.4% |
| gradient_boosting_namacro_v1 | HistGradientBoosting (lr 0.1, max_iter 100) | 14 (macro features dropped) | 0.659 | 0.756 | 0.241 | 13.1% |

Reference points: the test period's actual positive rate is 30.8%
(train: 6.4%). A constant prediction at the test rate scores
log-loss 0.618 / Brier 0.213 — gradient boosting is worse than that
constant on both.

### Decision: model_v1 is the logistic regression

Gradient boosting fails the bar by a wide margin, and the failure is
diagnosed as temporal drift, not an implementation bug. Its Train AUC
is 0.807 (it fits the training period well), but in the test period the
macro features move outside their training ranges entirely
(euribor3m and nr.employed lie wholly below their train ranges).
A tree model cannot extrapolate beyond its training support, so test
rows land in edge leaves learned under the old regime; the linear
model extends the trend instead. The ablation supports this reading:
dropping the macro features narrows the train–test AUC gap
(0.173 → 0.120) and improves test metrics, but gradient boosting
still trails the baseline — the residual gap is the regime/base-rate
shift itself. Per the Week 1 decision rule, we ship the
better-calibrated model and document drift as the binding constraint.

### Diagnostic: AUC by age band (logistic regression, test split)

| Age band | n | Positive rate | AUC |
|---|---|---|---|
| <30 | 1,945 | 30.2% | 0.755 |
| 30–39 | 3,051 | 27.4% | 0.754 |
| 40–49 | 1,359 | 27.0% | 0.780 |
| 50–59 | 1,037 | 34.7% | 0.720 |
| 60+ | 846 | 45.9% | 0.609 |

Ranking quality is stable under 60 but materially weaker for the 60+
band. Production caveat: if deployed, performance must be monitored
per age band; scores should not be compared across bands directly.

### Tests

7 tests guard the feature pipeline and the frozen split
(`tests/test_features.py`, `tests/test_data.py`). Run `python -m pytest`.
