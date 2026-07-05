"""
Trains a student academic-risk classifier on the public UCI Student
Performance dataset (Cortez & Silva, 2008).

"At risk" is defined as a final grade (G3) below 10 (out of 20, the
Portuguese passing threshold). G1/G2 (earlier-period grades) are dropped
from the feature set on purpose: the goal is to flag at-risk students
early, using only demographic/behavioral signals available before any
grades exist, not to leak the outcome itself.

Experiment tracking targets MLflow (optionally backed by DagsHub) via
environment variables:
    MLFLOW_TRACKING_URI
    MLFLOW_TRACKING_USERNAME
    MLFLOW_TRACKING_PASSWORD
If these are not set, MLflow falls back to a local ./mlruns/ file store,
so training works fully offline with no external account required.
"""

import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).parent / "data" / "student-mat.csv"
MODEL_DIR = Path(__file__).parent / "model"
MODEL_PATH = MODEL_DIR / "student_risk_pipeline.joblib"

NUMERIC_FEATURES = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences",
]
CATEGORICAL_FEATURES = [
    "school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob",
    "reason", "guardian", "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Download it from the UCI "
            "Machine Learning Repository (Student Performance dataset) and "
            "place student-mat.csv in the data/ folder."
        )
    return pd.read_csv(DATA_PATH, sep=";")


def build_pipeline(classifier) -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(steps=[("preprocess", preprocess), ("model", classifier)])


def evaluate(pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }
    if hasattr(pipeline, "predict_proba") and y_test.nunique() > 1:
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
    return metrics


def configure_tracking():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        db_path = Path(__file__).parent / "mlflow.db"
        local_store = f"sqlite:///{db_path}"
        mlflow.set_tracking_uri(local_store)
        print(f"MLFLOW_TRACKING_URI not set — using local store at {local_store}")
    mlflow.set_experiment("student-risk-classification")


def main():
    configure_tracking()

    df = load_data()
    df["at_risk"] = (df["G3"] < 10).astype(int)

    X = df[FEATURES]
    y = df["at_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=2000),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    best_name, best_pipeline, best_metrics = None, None, {"f1": -1}

    for name, clf in candidates.items():
        pipeline = build_pipeline(clf)
        with mlflow.start_run(run_name=name):
            pipeline.fit(X_train, y_train)
            metrics = evaluate(pipeline, X_test, y_test)

            mlflow.log_param("model_type", name)
            mlflow.log_param("n_features", len(FEATURES))
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)

            signature = infer_signature(X_train, pipeline.predict(X_train))
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                signature=signature,
                input_example=X_train.head(3),
            )

            print(f"{name}: {metrics}")

        if metrics["f1"] > best_metrics["f1"]:
            best_name, best_pipeline, best_metrics = name, pipeline, metrics

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"\nBest model: {best_name} {best_metrics}")
    print(f"Saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
