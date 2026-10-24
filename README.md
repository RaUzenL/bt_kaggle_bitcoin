# BT4012 Kaggle Bitcoin competition

Predicting illicit Bitcoin transactions (ROC-AUC). Training data covers time steps 1–35 and the test set steps 36–49, so every model is validated **forward in time** on three checks:

| Check | Fit on | Score on |
|---|---|---|
| `steps_22_28` | labelled steps 1–21 | steps 22–28 |
| `steps_29_35` | labelled steps 1–28 | steps 29–35 |
| `14_step_check` | the steps 1–21 model | steps 22–35 |

Settings are selected on the mean ROC-AUC of the two 7-step checks; the 14-step check is reported only.

## Repository layout

```
e1234567.ipynb                         final submission model, end to end
experiments/baselines_hgb_lightgbm.ipynb   baselines on the 165 supplied features: HGB, LightGBM, time-weighted HGB
submission/                            the two final submission files (written by e1234567.ipynb)
requirements.txt                       pinned dependencies
```

`e1234567.ipynb` covers data checks and preprocessing, the validation checks, graph feature engineering and two-round graph stacking (D2 LightGBM and D2 HGB), the stacked MLP, validation of every component and blend ratio, and the final submissions:

| File | LGBM : HGB : MLP rank blend |
|---|---|
| `submission/graph_blend_lgbm_hgb_mlp_311.csv` | 3:1:1 (top submission) |
| `submission/graph_blend_lgbm_hgb_mlp_111.csv` | 1:1:1 |

Running the notebook also creates `stacked_mlp.py` (written by the notebook) and `work/` (intermediate stage-2 features and MLP predictions, about 60 MB); both are regenerated on every run.

## Reproduce

### 1. Environment
Python **3.13.1**. Create a virtual environment and install the pinned packages:

```bash
python3.13 -m venv .venv
```

```bash
.venv/bin/pip install -r requirements.txt
```

**Apple Silicon (macOS 14 or later):** pip installs a NumPy build that uses Apple Accelerate. The results were produced with the OpenBLAS build of the same version; replace it to reproduce every score exactly:

```bash
.venv/bin/pip download numpy==2.3.2 --platform macosx_11_0_arm64 --only-binary=:all: --no-deps -d wheels
```

```bash
.venv/bin/pip install --force-reinstall --no-deps wheels/numpy-2.3.2-cp313-cp313-macosx_11_0_arm64.whl
```

Register the environment as the Jupyter kernel the notebooks expect (`bt4012`):

```bash
.venv/bin/python -m ipykernel install --user --name bt4012 --display-name "Python (BT4012)"
```

### 2. Data
Download the competition files and place them in `bt-4012-competition-2026/` at the repository root:

```
bt-4012-competition-2026/train.csv
bt-4012-competition-2026/test.csv
bt-4012-competition-2026/txs_edgelist.csv
bt-4012-competition-2026/sample_submission.csv
```

The data folder is not tracked in git.

### 3. Run
Open `e1234567.ipynb` with the `Python (BT4012)` kernel and run all cells, or run it headless from the repository root:

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 e1234567.ipynb
```

| Notebook | Approximate run time (8-core Apple Silicon CPU) |
|---|---|
| `e1234567.ipynb` | about 15 minutes |
| `experiments/baselines_hgb_lightgbm.ipynb` | about 2 minutes |

Everything runs on CPU; no GPU is needed.

### Reproducibility notes
- **Seeds** are fixed throughout; re-running gives identical submission files with the pinned versions.
- **scikit-learn must be 1.7.1.** Other versions build different HistGradientBoosting trees.
- **LightGBM and PyTorch in one process** can deadlock on macOS (both bundle OpenMP), so `e1234567.ipynb` runs the PyTorch MLP step in a separate Python process (`stacked_mlp.py`).
- The MLP uses 8 PyTorch threads; a different thread count changes its results slightly.
