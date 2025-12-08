# EuroFusion-AI

Classify 3D movement trajectories (circle, diagonal left/right, horizontal, vertical) using handcrafted features and a Random Forest classifier. The same model powers both the full training script and the demo predictor.

## Feature Representation

Each movement trajectory is transformed into a 42-dimensional handcrafted feature vector capturing its geometric and kinematic structure. The feature set includes:

- **Total per-axis displacement** and full 3D **path length**
- **Extreme-point axis lengths** (maximal span in x/y/z) and **normalized axis-direction vectors**
- **Peak displacements** from the starting position along each axis
- **Bounding-box extents** and **aspect-ratios** across all axis pairs
- **Per-axis movement fractions** and directional balance terms
- **Loopiness** (ratio of path length to straight-line axis length) to distinguish curved from linear motions
- **Radius statistics** (mean and variance of radial distances) for detecting circular trajectories
- Additional squared and cross-axis terms to capture nonlinear and correlated spatial effects easier

All features are translation-invariant and robust to noise, providing strong geometric separation between circle, diagonal, horizontal, and vertical classes.

## Hyperparameter Selection

Model hyperparameters were selected using `GridSearchCV` with 5-fold cross-validation on the training portion of the labeled dataset.  
The search explored different model complexities (tree depth, number of estimators), regularization strengths (minimum samples per split/leaf), and feature subsampling strategies.

The best configuration –  
`n_estimators=200`, `max_depth=7`, `max_features="sqrt"`, `min_samples_leaf=1`, `min_samples_split=2` –  
consistently produced high accuracy on both development and test sets, indicating good generalization.

For the final demo model, these hyperparameters are fixed and the classifier is retrained on **all** available labeled data to maximize predictive performance on unseen trajectories.

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
- To refresh the environment: delete `.venv`, recreate it, reinstall from the requirements files.
