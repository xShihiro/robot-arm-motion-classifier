# EuroFusion-AI

Classify 3D movement trajectories (circle, diagonal left/right, horizontal, vertical) using handcrafted features and a Random Forest classifier. The same model powers both the full training script and the demo predictor.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate            # on Windows: .venv\Scripts\activate
python -m pip install --upgrade pip  # (optional) ensure the latest pip version is installed
pip install -r requirements.txt      # runtime deps
pip install -r requirements-dev.txt  # formatting/linting tools (optional)
```
- Tested with Python 3.11; use 3.10+ for best compatibility.

## Data
- Default labeled data: `dataset_combined/`
- Demo (unlabeled) data: `fake_demo_data/`
- To point at different data, change `DATA_DIRECTORY` in `src/data_preprocessing.py` (and `DEMO_DIRECTORY` in `src/demo.py`) or pass alternative paths when calling the helper functions.

## How to run (Random Forest)
- Train + evaluate (with grid search, metrics, optional tree viz):
  ```bash
  python src/random_forest.py
  ```
- Train on the labeled set and predict the demo files:
  ```bash
  python src/demo.py
  ```

Both scripts assume you run them from the repo root so imports resolve (`python src/...`).

## Configuration
- Random Forest settings live in `RF_CONFIG` inside `src/random_forest.py` (augmentation on/off, number of augmentations, CV folds, model hyperparameters, tree visualization toggle).
- `src/demo.py` pulls its defaults from `RF_CONFIG`; adjust there for demo-specific overrides.
- Augmentation strategies are class-specific in `src/data_augmentation.py`.

## Project layout
- `src/data_preprocessing.py` – load labeled movement files into class lists
- `src/data_augmentation.py` – class-specific augmentations
- `src/feature_extraction.py` – handcrafted feature vectors + train/dev/test split helper
- `src/random_forest.py` – Random Forest training, grid search, evaluation, optional tree plotting
- `src/demo.py` – train RF and predict unlabeled demo files

## Notes
- Plots require a display; on headless runs set a non-interactive backend (e.g., `MPLBACKEND=Agg`) to avoid errors.
- To refresh the environment: delete `.venv`, recreate it, reinstall from the requirements files.
