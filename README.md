# Smart irrigation under sensor faults: stress testing and fault-aware learning

Code for the paper:

> **Robust Smart Irrigation under Sensor Faults: Stress Testing and Fault-Aware Learning for IoT Greenhouses**
> B. El Khalyly and Z. Ibn Batouta, 2026 (under review).

We take a LightGBM irrigation classifier and break its inputs on purpose. The faults are missing values, noise, calibration bias, a frozen sensor and slow drift, and we measure how much the decisions degrade.
We then retrain the same model on a training set that already contains some corrupted rows and check how much of the loss it recovers.

## What the pipeline does

1. **`src/prep3.py`** keeps the labelled rows, parses both timestamp formats, drops unusable rows and exact duplicate feature–label vectors. That leaves 48,642 rows, 208 positives and 17 days. It then sets the calendar day as the grouping variable.
2. **`src/eval3.py`**:
   - Builds five `StratifiedGroupKFold` folds by day, deterministic with no shuffling, so no day is ever split between training and validation.
   - Fits a standard LightGBM model per fold.
   - Fits a fault-aware model per fold on that fold's training data plus 50% extra rows. A third of the extra rows have one sensor set to missing, a third get Gaussian noise (0.5 σ) and a third get a ±1 σ bias.
   - Injects five fault families into the validation folds only, with five seeds each. Every fault parameter is computed from the training fold.
   - Runs the single-sensor criticality test.
   - Runs a one-sided paired Wilcoxon test over the 25 stressed realizations.
3. **`src/figs3.py`** draws the figures.

## Run it in Google Colab or Jupyter

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/badrelkhalylyphd-boop/irrigation-sensor-fault-robustness/blob/main/reproduce_colab.ipynb)

`reproduce_colab.ipynb` runs the whole experiment in one notebook. It installs the pinned library versions; on Colab the session restarts once, after which you choose *Run all* again. On Colab, first upload `greenhouse_dataset.csv` in the Files panel (it is read from `/content`; if it is missing the notebook asks for it). Locally, the copy in `data/` is used, and ends with a table that compares every number reported in the paper with the current run and marks each one ✅ or ❌.


## Quick start (command line)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python reproduce.py
```

The run takes a few minutes. Outputs in `outputs/`:

| File | Content | Paper |
|---|---|---|
| `irrigation_clean.parquet` | cleaned labelled data with `DateGroup` | Sec. III-A |
| `article3_results.csv` | every (seed, training, fault) run with F1, MCC, recall, precision, BAcc, PR-AUC, confusion counts | Tables I–II, Figs. 1–2 |
| `article3_sensor_criticality.csv` | full single-channel faults per sensor | Table III, Fig. 3 |
| console output | Wilcoxon p-values and mean gains | Sec. IV-E |
| `figures/*.png` | figures | Figs. 1–3 |

## Expected results

Our own outputs are in `results/` for comparison.

| | Standard | Fault-aware |
|---|---|---|
| Clean F1 (%) | 46.57 | 48.50 |
| Mean F1 over 25 stressed runs (%) | 36.11 | 44.39 |
| Retention of clean F1 | 77.5% | 91.5% |
| Mean F1 under missing values (%) | 11.16 | 44.26 |

One-sided Wilcoxon (fault-aware > standard, n = 25):

| Metric | p-value |
|---|---|
| F1 | 6.99 × 10⁻⁵ |
| MCC | 1.39 × 10⁻³ |
| PR-AUC | 0.86 (not significant) |

Under complete loss of Soil Percent, the standard model's F1 drops to 1.3% and the fault-aware model's to 21.9%.

## Reproducibility notes

- Verified on Windows 11, Python 3.11.9, with the versions in `requirements.txt`.
- The grouped folds use `StratifiedGroupKFold(n_splits=5)` **without shuffling**. An earlier draft of the supplementary document listed `shuffle=True, random_state=42`. That was a documentation error: the reported numbers come from the deterministic split, and this repository uses it.
- With only 17 days and most positives on 18–20 May, the grouped estimate depends on which days land in which fold. The paired design (same folds, same faults, both models) is what makes the comparison between standard and fault-aware training meaningful, not the absolute F1 level.
- Fault seeds follow the scheme in `eval3.py` (`10000 + seed*100 + fold*10 + fault_index` for the global test, and `9000 + …` for the criticality test). Augmentation seeds are `500 + fold`.

## Data

See [`data/README.md`](data/README.md).

## Citation

Please cite the paper (reference to be updated on publication) and the dataset:

```
I. Keok, "IoT Greenhouse Irrigation Sensor Dataset," Mendeley Data, V1, 2026, doi: 10.17632/2tvpsht4s4.1.
```

## License

Code: MIT (see `LICENSE`). Data: CC BY 4.0, from the original authors.
