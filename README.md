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
