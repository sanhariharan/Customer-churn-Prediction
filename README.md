# Customer Churn Prediction — ANN (TensorFlow)

A production-ready, reproducible project for predicting customer churn using an Artificial Neural Network (ANN) built with TensorFlow and packaged with a Streamlit inference UI.

This repository contains the full experiment (data prep, model training, evaluation) and a lightweight web app for non-technical stakeholders to interact with the model.

---

## Key highlights

- Model type: Keras Sequential ANN (Dense layers)
- Task: Binary classification — predict if a customer will churn
- Training notebook: `experiments.ipynb`
- Inference app: `app.py` (Streamlit)
- Saved artifacts: `model.h5`, `scaler.pkl`, `label_encoder_gender.pkl`, `onehot_encoder_geo.pkl`
- Logs: TensorBoard logs under `logs/fit/`

---

## Quick start (Windows / PowerShell)

1. Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Launch the Streamlit inference app (ensure `model.h5` and pickle files are in the project root):

```powershell
streamlit run app.py
```

4. To inspect or re-train the model, open `experiments.ipynb` in Jupyter or VS Code and run cells sequentially.

---

## Project structure

- `Churn_Modelling.csv` — original dataset used for training.
- `experiments.ipynb` — notebook for preprocessing, training, and saving artifacts.
- `app.py` — Streamlit application for interactive predictions.
- `model.h5` — trained Keras model (HDF5 format).
- `scaler.pkl`, `label_encoder_gender.pkl`, `onehot_encoder_geo.pkl` — preprocessing artifacts saved with pickle.
- `requirements.txt` — dependency list.
- `logs/fit/` — TensorBoard logs from training runs.

---

## Reproducible training (overview)

The notebook performs these steps:

1. Load and clean data (drop identifiers: `RowNumber`, `CustomerId`, `Surname`).
2. Encode categorical variables:
   - `Gender` using `LabelEncoder` (saved to `label_encoder_gender.pkl`).
   - `Geography` using `OneHotEncoder` (saved to `onehot_encoder_geo.pkl`).
3. Split into train and test sets (`train_test_split`, test_size=0.2).
4. Fit `StandardScaler` on training data and transform both train/test (saved to `scaler.pkl`).
5. Build a Sequential ANN with Dense layers and `sigmoid` output for binary classification.
6. Compile with Adam optimizer and binary cross-entropy loss; train with `EarlyStopping` and `TensorBoard` callbacks.
7. Save the final model to `model.h5`.

Notes:
- For strict experiment hygiene, fit encoders/scalers only on the training set (the notebook currently does this for the scaler; ensure the same for encoders if you refactor).
- Use stratified splitting (`stratify=y`) when `Exited` is imbalanced.

---

## Inference contract

To produce correct model inputs the following ordering and transformations are required:

Inputs (expected columns before scaling) — order used in `app.py`:

1. `CreditScore` (int)
2. `Gender` (label-encoded using `label_encoder_gender.pkl`)
3. `Age` (int)
4. `Tenure` (int)
5. `Balance` (float)
6. `NumOfProducts` (int)
7. `HasCrCard` (0/1)
8. `IsActiveMember` (0/1)
9. `EstimatedSalary` (float)
10+. Geography one-hot columns in the same order as `onehot_encoder_geo.get_feature_names_out(['Geography'])`

Output:
- Single float probability in [0, 1] representing churn risk. The Streamlit UI uses a 0.5 threshold for human-friendly messaging.

Common error modes:
- Missing artifacts (`model.h5`, pickles) will cause the app to fail on start.
- Unseen categorical values at prediction time — configure `OneHotEncoder(handle_unknown='ignore')` if you expect this.

---

## Model evaluation (recommended additions)

After training, compute these metrics on the hold-out test set:

- Accuracy (baseline)
- Precision / Recall / F1
- ROC AUC (recommended for imbalanced classes)
- Confusion matrix and Precision-Recall curve

Small code snippet (in notebook) to evaluate:

```python
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
preds = (model.predict(X_test) > 0.5).astype(int)
print(classification_report(y_test, preds))
print('ROC AUC:', roc_auc_score(y_test, model.predict(X_test)))
print('Confusion matrix:\n', confusion_matrix(y_test, preds))
```

---

## Production & deployment notes

- Consider saving the model in TensorFlow SavedModel format (`model.save('saved_model')`) for better compatibility across TF versions.
- Bundle preprocessing and model into a single pipeline (e.g., `sklearn.pipeline.Pipeline` or a custom wrapper) and persist that to ensure inference parity.
- Add input validation for the Streamlit app and graceful error messages when artifacts are missing.

---

## Reproducibility

- Pin exact package versions (create a `requirements.lock` or `pip freeze > requirements.lock`).
- Record the random seeds used for NumPy/TensorFlow/Python to reproduce training runs.

Example to freeze current environment (PowerShell):

```powershell
pip freeze > requirements.lock
```

---

## Next steps (recommended)

1. Convert preprocessing into a `ColumnTransformer`/`Pipeline` and store the entire pipeline.
2. Add a `predict.py` CLI for batch inference and unit tests for preprocessing steps.
3. Extend evaluation in `experiments.ipynb` with plots and an explainability step (SHAP or LIME).
4. Add CI checks (formatting, linting, basic unit tests) and a minimal GitHub Actions workflow.

---

## License & credits

- Author: San Hariharan (as credited in the app UI)
- Choose a license and add a `LICENSE` file (MIT is a common permissive option).

If you'd like, I can also:
- Refactor the notebook to use a reproducible `Pipeline` and add evaluation plots.
- Add the `predict.py` script and unit tests.

Tell me which one you want next and I will implement it.
