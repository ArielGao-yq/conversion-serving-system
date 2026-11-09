# Data Card — UCI Bank Marketing (bank-additional-full)

## Source
- UCI Machine Learning Repository, dataset 222 — Bank Marketing
  (Moro, Cortez & Rita, 2014). Collected from a Portuguese bank's
  direct phone marketing campaigns, May 2008 – November 2010.
- File: `bank-additional-full.csv` (41,188 rows, 21 columns), fetched by
  `scripts/download_data.py` from the UCI archive. License: CC BY 4.0.
- Raw data is NOT committed to this repo (see `.gitignore`); only the
  frozen split indices under `data/splits/` are versioned.

## Prediction task
- **Target:** `y` — whether the client subscribed to a term deposit
  ("yes" / "no"). Positive rate ≈ 11% overall.
- **Prediction time:** the moment the bank decides to call the client.
  Only features knowable at or before that moment may be used.

## Leakage
- **`duration` (length of the final call) is EXCLUDED.** It is known only
  after the call ends — the very event being predicted. Evidence
  (`scripts/check_data.py`): every row with `duration == 0` has
  `y == "no"`, and mean duration is far higher for subscribers.
  A model using it reaches AUC ~0.95 by reading the answer.

## Quirks and sentinels
- `"unknown"` is a recorded category (e.g. in `job`, `education`), not a
  missing-value marker. Keep it as its own category.
- `pdays = 999` means the client was never contacted in a previous
  campaign — a sentinel value, not a real day count.
- Macro indicators (`euribor3m`, `emp.var.rate`, `nr.employed`, …) drift
  substantially over the sample period (financial crisis), which is why
  the split is temporal and base rates differ by slice.

## Split
- Temporal, by row order: first 80% train (32,950), last 20% test (8,238).
- Frozen as v1: `data/splits/v1/{train,test}_index.csv` (row ids only),
  created by `scripts/make_split.py`. Train positive rate 6.4%,
  test positive rate 30.8% — see the design doc's calibration note.
