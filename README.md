# BT4012 Kaggle Bitcoin competition: final model

Predicts the probability of illicit Bitcoin transactions (using the ROC-AUC score). The final submissions are rank blends of graph-stacking models.

Competition: https://www.kaggle.com/competitions/bt-4012-competition-2026/overview

Please import the competition data folder under the root of this repo (see Step 3).

## Contents

| Path | What it is |
|---|---|
| `e1297793.ipynb` | Final model, end to end: data checks, graph features, stacking, stacked MLP, blends, submission files |
| `experiments/baselines_hgb_lightgbm.ipynb` | Baselines on the raw 165 features (HistGradientBoosting, LightGBM, time weighting) |
| `submission/` | The two final submission files |
| `requirements.txt` | Python packages with pinned versions |

## Reproduce the results

### Step 1: Install Python 3.13 (required)
This project needs **Python 3.13** (3.13.1 was used to produce the results). **Python 3.14 and other versions will not work**, because `scikit-learn==1.7.1` has no build for them and the installation will fail.

Install Python 3.13 before going further. On macOS you can run `brew install python@3.13`, or download it from python.org.

Check that it is available:

```bash
python3.13 --version
```

If this command is not found, install Python 3.13 first. Do not continue with another version.

On Windows, use `py -3.13 --version` and `py -3.13 -m venv .venv` in place of the `python3.13` commands.

### Step 2: Download this repository
```bash
git clone https://github.com/RaUzenL/bt_kaggle_bitcoin.git
```

```bash
cd bt_kaggle_bitcoin
```

### Step 3: Add the competition data
The data is not in this repository. Download it from the Kaggle competition page, then:

1. Create a folder named `bt-4012-competition-2026` inside `bt_kaggle_bitcoin`.
2. Put these four files inside it: `train.csv`, `test.csv`, `txs_edgelist.csv`, `sample_submission.csv`.

The folder should look like this:

```
bt_kaggle_bitcoin/
├── bt-4012-competition-2026/
│   ├── train.csv
│   ├── test.csv
│   ├── txs_edgelist.csv
│   └── sample_submission.csv
├── e1297793.ipynb
├── experiments/
└── ...
```

### Step 4: Create an environment and install the packages
Create a virtual environment inside the project, using Python 3.13:

```bash
python3.13 -m venv .venv
```

On Windows, use `py -3.13 -m venv .venv` instead.

Activate it:
- macOS or Linux: `source .venv/bin/activate`
- Windows (PowerShell): `.venv\Scripts\Activate.ps1`

Check that the environment uses Python 3.13. The output must start with `Python 3.13`. If it shows another version, delete `.venv` and create it again with `python3.13`:

```bash
python --version
```

Install the pinned packages:

```bash
pip install -r requirements.txt
```

Register the notebook kernel. The notebooks are set to use a kernel called `bt4012`:

```bash
python -m ipykernel install --user --name bt4012 --display-name "Python (BT4012)"
```

### Step 5: Run the final model
**Option A: Jupyter (recommended for viewing each step).** Start Jupyter:

```bash
jupyter lab
```

Open `e1297793.ipynb`, choose the kernel **Python (BT4012)**, then select **Run → Run All Cells**.

**Option B: command line.** From the project folder:

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 e1297793.ipynb
```

The run takes roughly 10–15 minutes on an 8-core laptop CPU. The MLP step starts its own Python process automatically, so you don't need to do anything extra for it.

The notebook writes two files to `submission/`:
- `graph_blend_lgbm_hgb_mlp_311.csv` (top submission)
- `graph_blend_lgbm_hgb_mlp_111.csv` (second final pick)

### Step 6 (optional): Run the baselines
```bash
cd experiments
```

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 baselines_hgb_lightgbm.ipynb
```

```bash
cd ..
```

### Step 7: Check your results
The notebook prints a table of blend ratios. These are the values to expect (ROC-AUC):

| Model | Steps 22–28 | Steps 29–35 | 14-step check (22–35) |
|---|---|---|---|
| Stacked MLP alone | 0.97359 | 0.99513 | 0.94743 |
| Blend 3:1:1 | 0.94777 | 0.99881 | 0.88502 |
| Blend 1:1:1 | 0.95606 | 0.99860 | 0.90151 |





## Notes
- **Seeds are fixed** in every notebook, so re-running on the same machine gives the same numbers.
- **The MLP uses 8 CPU threads.** A machine with a different core count can give slightly different MLP results.
- **Run times are approximate** and depend on your machine.
- **Only the final model is in this repository.** The earlier experiments are not included.

## AI Declaration

Claude helped to improve the readability of the code, draft this README, and summarise the public leaderboard scores of the submissions for me to decide. I made the final decisions on the models, validation design, blend ratios and submissions without the help of Claude.
