# DVC + Project guide (step-by-step)

This document explains, in simple steps, how this project is organized and how to run the pipeline using DVC and the Python scripts in `src/`.
It's written so a school student can follow along.

## What this project does (short)

- It takes raw CSV data from `data/raw/`, cleans and preprocesses it, creates features (vectorized data), trains an XGBoost model, saves the model to `models/`, and writes evaluation metrics to `results/`.

## Project files (what to look at)

- `data/raw/` - original raw CSV files (train/test).
- `data/processed/` - cleaned and processed CSV files.
- `data/vectorized/` - data after feature engineering (ready for the model).
- `models/xgb_model.pkl` - trained model file.
- `results/evaluation_metrics.json` - JSON file with evaluation numbers (accuracy, etc.).
- `src/` - all Python scripts. Each script corresponds to one step in the pipeline:
  - `src/data_ingestion.py` - downloads or copies raw data into `data/raw`.
  - `src/data_preprocess.py` - cleans raw data and writes `data/processed/*.csv`.
  - `src/feature_engineering.py` - produces vectorized features in `data/vectorized/`.
  - `src/model_building.py` - trains a model and writes `models/xgb_model.pkl`.
  - `src/model_evaluation.py` - loads the model and data, computes metrics and writes `results/evaluation_metrics.json`.
- `dvc.yaml`, `dvc.lock`, `params.yaml` - DVC pipeline definitions and parameters.
- `requirements.txt` - Python packages the project needs.

## Simple setup (one-time on your machine)

1. Install Python (3.8+ recommended).
2. Open PowerShell in the project folder (where this file is). Example: `c:\Users\use\Desktop\dvc`.
3. Create and activate a virtual environment, then install packages:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt
```

4. Install DVC if you don't have it (one-time):

```powershell
pip install dvc
```

Note: If the repository already has DVC initialized and tracked files (it has `dvc.yaml`), you don't need to run `dvc init`. If you are starting fresh you can run `dvc init`.

Additional notes about the virtual environment and NLTK:

```powershell
python -m venv .venv
# To activate in PowerShell (may require changing execution policy):
.\.venv\Scripts\Activate
# If activation is blocked, you can still install packages using the venv pip directly:
.venv\Scripts\pip.exe install -r requirements.txt
```

- The preprocessing code uses NLTK corpora (`stopwords`, `wordnet`). After installing packages run this once to download required NLTK data (or run it from a Python REPL):

```powershell
.venv\Scripts\python.exe -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```

- If you want to avoid running the downloader manually, you can run `dvc repro` after the packages are installed; the preprocessing script will raise a clear error telling you which NLTK resource is missing and you can then run the downloader command above.

## How the DVC pipeline maps to `src/` (easy list)

Each pipeline stage runs one `src/` script and reads/writes files in `data/`, `models/`, or `results/`.

1. data_ingestion
	- Script: `src/data_ingestion.py`
	- Produces: `data/raw/*.csv`
	- DVC stage example (only if you need to add it manually):

```powershell
dvc stage add -n data_ingestion -d src/data_ingestion.py -o data/raw python src/data_ingestion.py
```

2. data_preprocess
	- Script: `src/data_preprocess.py`
	- Reads: `data/raw/` and writes `data/processed/`
	- Stage example:

```powershell
dvc stage add -n data_preprocess -d src/data_preprocess.py -d data/raw -o data/processed python src/data_preprocess.py
```

3. feature_engineering
	- Script: `src/feature_engineering.py`
	- Reads: `data/processed/` and writes `data/vectorized/`
	- Stage example:

```powershell
dvc stage add -n feature_engineering -d src/feature_engineering.py -d data/processed -o data/vectorized python src/feature_engineering.py
```

4. model_building
	- Script: `src/model_building.py`
	- Reads: `data/vectorized/` and writes `models/xgb_model.pkl`
	- Stage example:

```powershell
dvc stage add -n model_building -d src/model_building.py -d data/vectorized -o models/xgb_model.pkl python src/model_building.py
```

5. model_evaluation
	- Script: `src/model_evaluation.py`
	- Reads: `models/xgb_model.pkl` and `data/vectorized/`, writes metrics to `results/evaluation_metrics.json`
	- Stage example (also marks the metrics file so `dvc metrics show` can read it):

```powershell
dvc stage add -n model_evaluation -d src/model_evaluation.py -d models/xgb_model.pkl -d data/vectorized --metrics results/evaluation_metrics.json python src/model_evaluation.py
```

## Run the whole pipeline (recommended)

If the DVC pipeline (`dvc.yaml`) is already defined, the easiest way to run everything in order is:

```powershell
dvc repro
```

`dvc repro` will check which stages are out-of-date and run only those. It follows the order of dependencies so you don't need to run each script manually.

## View the pipeline graph

To see a picture of the pipeline (which stage depends on which), run:

```powershell
dvc dag
```

This prints a simple graph showing the stages and their order.

## See tracked metrics

After running evaluation, show the metrics DVC knows about:

```powershell
dvc metrics show
```

This reads `results/evaluation_metrics.json` (if set up in the DVC stage) and prints it in a readable form.

## Run a single script for testing

If you're debugging one step, run the Python script directly. Example:

```powershell
python src/data_preprocess.py
```

This is useful for quick checks. But when the pipeline is tracked by DVC it's better to use `dvc repro` so outputs and dependencies are recorded.

## Common simple troubleshooting tips (for beginners)

- If a script fails because a file is missing, check the input directories listed above (`data/raw`, `data/processed`, `data/vectorized`).
- If Python can't find a package, ensure your virtual environment is activated and `pip install -r requirements.txt` completed without errors.
- If DVC can't find a stage, open `dvc.yaml` to see the stage names. You can also run `dvc status` to check what's missing/out-of-date.

## Quick checklist (one-liner sequence)

1. Create venv & install: `python -m venv .venv; .\\.venv\\Scripts\\Activate; pip install -r requirements.txt`
2. (Optional) dvc init if starting new: `dvc init`
3. Run pipeline: `dvc repro`
4. View DAG: `dvc dag`
5. Show metrics: `dvc metrics show`
