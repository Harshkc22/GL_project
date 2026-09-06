"""
Model Building with Experimentation Tracking
---------------------------------------------
1. Loads the train/test splits from the Hugging Face dataset space.
2. Builds a preprocessing + XGBoost classification pipeline.
3. Tunes hyperparameters with GridSearchCV and logs every run with MLflow.
4. Evaluates the best model on the held-out test set.
5. Registers the best model on the Hugging Face model hub.
"""
import os
from pathlib import Path

import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from huggingface_hub import hf_hub_download, HfApi, create_repo
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HF_USERNAME = "harshkc"
DATASET_REPO_ID = f"{HF_USERNAME}/tourism-wellness-dataset"
MODEL_REPO_ID = f"{HF_USERNAME}/tourism-wellness-model"
TARGET_COL = "ProdTaken"

BASE_DIR = Path(__file__).resolve().parent.parent  # tourism_project/
token = os.environ.get("HF_TOKEN")

# Use a local MLflow server if one is already running (started by the CI
# pipeline / a notebook cell); fall back to a local sqlite store otherwise.
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")


def load_data():
    train_path = hf_hub_download(repo_id=DATASET_REPO_ID, filename="train.csv",
                                  repo_type="dataset", token=token)
    test_path = hf_hub_download(repo_id=DATASET_REPO_ID, filename="test.csv",
                                 repo_type="dataset", token=token)
    return pd.read_csv(train_path), pd.read_csv(test_path)


def build_pipeline(Xtrain: pd.DataFrame) -> Pipeline:
    num_cols = Xtrain.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in Xtrain.columns if c not in num_cols]

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_cols),
    ])

    model = XGBClassifier(random_state=42, eval_metric="logloss")
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def main():
    train_df, test_df = load_data()
    Xtrain, ytrain = train_df.drop(columns=[TARGET_COL]), train_df[TARGET_COL]
    Xtest, ytest = test_df.drop(columns=[TARGET_COL]), test_df[TARGET_COL]

    pipe = build_pipeline(Xtrain)

    # Hyperparameters to tune
    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [3, 5, 7],
        "model__learning_rate": [0.05, 0.1],
    }

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    except Exception:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("tourism-wellness-package")

    with mlflow.start_run(run_name="xgboost_gridsearch"):
        grid = GridSearchCV(pipe, param_grid, cv=3, scoring="f1", n_jobs=-1)
        grid.fit(Xtrain, ytrain)
        best_model = grid.best_estimator_

        # Log all tuned parameters
        mlflow.log_params(grid.best_params_)

        # Evaluate model performance
        preds = best_model.predict(Xtest)
        metrics = {
            "accuracy": accuracy_score(ytest, preds),
            "precision": precision_score(ytest, preds),
            "recall": recall_score(ytest, preds),
            "f1_score": f1_score(ytest, preds),
        }
        mlflow.log_metrics(metrics)

        print("Best hyperparameters:", grid.best_params_)
        print("Test metrics:", metrics)

    # Save the best model locally
    model_dir = BASE_DIR / "model_building"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "best_model.joblib"
    joblib.dump(best_model, model_path)

    # Register the best model on the Hugging Face model hub
    api = HfApi(token=token)
    create_repo(repo_id=MODEL_REPO_ID, repo_type="model", private=False,
                exist_ok=True, token=token)
    api.upload_file(
        path_or_fileobj=str(model_path),
        path_in_repo="best_model.joblib",
        repo_id=MODEL_REPO_ID,
        repo_type="model",
        token=token,
    )
    print(f"Best model registered at: https://huggingface.co/{MODEL_REPO_ID}")


if __name__ == "__main__":
    main()
