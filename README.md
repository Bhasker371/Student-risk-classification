# Student Risk Predictor

Predicts whether a student is at risk of failing (final grade below 10/20)
using only demographic and behavioral factors — no exam grades required,
so at-risk students can be flagged for early intervention before any
grades exist.

This is an original implementation built around the general idea of a
student academic-risk classifier: its own feature set, model comparison,
experiment tracking setup, and Streamlit UI, built from scratch rather
than derived from any other project's code.

---

## Dataset

[UCI Machine Learning Repository — Student Performance Data Set](https://archive.ics.uci.edu/dataset/320/student+performance)
(Cortez, P. & Silva, A., 2008). `data/student-mat.csv` contains the
Portuguese "Mathematics" course results, used here under the dataset's
public academic-use terms.

"At risk" is defined as `G3 < 10` (the Portuguese passing threshold, out
of 20). The final and prior-period grades (`G1`, `G2`, `G3`) are excluded
from the feature set on purpose — using only information available
before any grades exist keeps the model useful for early intervention
rather than just restating a grade the school already knows.

## Model

- `sklearn.pipeline.Pipeline`: `ColumnTransformer` (`StandardScaler` for
  numeric features, `OneHotEncoder` for categorical features) feeding a
  classifier.
- Trains and compares **Logistic Regression** and **Random Forest**,
  selecting the better model by F1 score.
- Every run logs params, metrics (accuracy, F1, ROC-AUC), and the model
  artifact to **MLflow**.

## Experiment tracking

Set these environment variables to log to a remote MLflow server (e.g.
[DagsHub](https://dagshub.com)):

```
MLFLOW_TRACKING_URI=https://dagshub.com/<your-username>/student-risk-classification.mlflow
MLFLOW_TRACKING_USERNAME=<your-username>
MLFLOW_TRACKING_PASSWORD=<your-dagshub-token>
```

Keep these in a local `.env` (already gitignored) — never commit them.
If unset, training falls back to a local SQLite-backed MLflow store
(`mlflow.db`, also gitignored), so the pipeline runs fully offline.

## Setup

```bash
python -m venv venv
source venv/bin/activate  # venv\Scripts\activate on Windows
pip install -r requirements.txt

python train.py            # trains, logs to MLflow, saves model/student_risk_pipeline.joblib
streamlit run streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## Project structure

```
student-risk-predictor/
├── data/student-mat.csv       # public UCI dataset
├── train.py                   # trains, evaluates, logs to MLflow
├── streamlit_app.py           # serves predictions + explains them
├── model/                     # saved joblib artifact (created by train.py)
├── requirements.txt
├── .gitignore
└── README.md
```
