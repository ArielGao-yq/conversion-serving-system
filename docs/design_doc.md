# Design Doc — Project 1: Conversion Prediction Service

## 1. Problem
Predict whether a bank client will subscribe to a term deposit, so the bank can
decide whom to call. One prediction per client, at the moment the bank decides
to call — not after the call.

## 2. Data
- **Dataset:** UCI Bank Marketing, `bank-additional-full.csv` (41,188 rows, 21 columns).
  Downloaded by `scripts/download_data.py`. License: CC BY 4.0.
- **Target:** `y` = subscribed ("yes" / "no"). Positive rate ≈ 11% overall — but
  it drifts sharply over time: 6.4% in the train slice vs 30.8% in the test
  slice (see §6). Any probability benchmark must use the test slice's own rate.
- **Prediction time:** the moment the bank decides to call. Only features known
  *before* the call may be used.

## 3. Leakage audit — what is excluded and why
- **`duration` — DROPPED.** Length of the final call, known only after it ends.
  Evidence from Session B: rows with `duration == 0` are always `y = "no"`, and
  mean duration differs sharply by outcome. A model using it reaches AUC ~0.95
  by reading the answer, not by predicting.
- All other features are known at or before prediction time (client attributes,
  campaign history, contact timing, macro indicators).
- Quirks: "unknown" is a recorded category, not a missing-value marker;
  `pdays = 999` means never contacted in a previous campaign (a sentinel, not
  a real day count).

## 4. Features (19, after dropping `duration`)
All columns except `y` and `duration`. Numeric columns are scaled
(`StandardScaler`); categorical columns are one-hot encoded
(`OneHotEncoder(handle_unknown="ignore")`), so a category seen only in test
gets all-zero columns instead of crashing.

## 5. Split — frozen, never re-split
Temporal, by row order (data is chronologically ordered, May 2008 – Nov 2010):
first 80% train (32,950 rows), last 20% test (8,238 rows).
Frozen as v1: `data/splits/v1/{train,test}_index.csv`. Weeks 2–4 inherit this split.

## 6. Baseline model (Session C)
- Logistic regression in one sealed sklearn `Pipeline` (preprocessing + model),
  `max_iter=1000`. No tuning. This is the ancestor of Week 2's sealed `model_v1`.
- Results on the frozen test split (run Nov 6, 2026):
  - AUC: **0.748** (near 0.95 would mean `duration` leaked back in — it didn't)
  - Log-loss: **0.570**
  - Brier score: **0.191**
  - Benchmarks, computed at the *test* slice's base rate (30.8%): a dumb model
    predicting 0.308 for everyone scores log-loss 0.618 / Brier 0.213 — the
    baseline beats both bars. (Benchmarks computed at the overall 11% rate —
    0.35 / 0.098 — are the wrong reference for a temporal split.)
  - Calibration note: mean predicted probability on test is 25.7% vs an actual
    30.8% — the model under-predicts the later period by ~5 points because the
    base rate itself shifted (train 6.4% → test 30.8%). Ranking (AUC) is
    unaffected. This gap is exactly what Week 4 drift monitoring is for.
- The same three numbers are recorded in the Project 1 README results table.
  Every Week 2 model has to beat this baseline — or document why it doesn't.

## 7. Serving mode
- Single-request JSON API. "Live traffic" is simulated from the held-out test chunk.
- Week 2: the same steps as one sealed Pipeline, `model_v1`, tracked in MLflow
  (≥3 runs compared).
- Week 3: FastAPI `/predict` + `/health` with input validation, Docker, CI.

## 8. Success metrics
- **Week 1 gate:** baseline trained on the frozen split; AUC / log-loss /
  calibration logged in the README; this design doc written.
- **Week 2 gate:** a stronger model (gradient boosting or small PyTorch) beats
  the baseline, or a documented reason it doesn't; tests for the feature transforms.
- **Ship gate (Nov 30):** a user can open the repo, run one command,
  call the endpoint, and read latency, drift, and model results without asking
  anything.

## 9. Review decisions (30-minute review — decided Nov 8, 2026)
1. **"Beats the baseline" (Week 2):** AUC ≥ 0.768 (+0.02 over 0.748) on the frozen
   v1 test split AND log-loss < 0.570. (+0.01 would be within ~1.5 SE of the AUC
   estimate on 8,238 rows — not distinguishable from luck.) If gradient boosting
   clears neither bar, ship the better-calibrated model and document why:
   temporal drift, not model class, is the binding constraint.
2. **Latency target for /predict:** p50 < 50 ms, p95 < 200 ms per single request,
   measured on the local Docker deployment in Week 4's load test
   (percentiles after warm-up, not averages).
3. **Retraining trigger:** drift-first — PSI ≥ 0.25 on any monitored feature, or
   mean predicted probability shifted > 5 points from its reference level for a
   full monitoring window; monthly scheduled retrain as the production backstop.
   Week 4 implements detection + alarm on the simulated stream and one
   demonstrated retrain. (Motivated by the baseline's own 6.4% → 30.8% base-rate
   shift between train and test periods.)
4. **Fairness/privacy:** no feature exclusions for `model_v1` — the dataset is
   anonymized and all 19 features are the benchmark standard. Production caveat:
   `age`, `marital`, `job` are proximity-to-protected-class features and would
   require a fairness audit before any real deployment. Week 2 adds AUC by age
   band as a reported diagnostic (not a gate).

