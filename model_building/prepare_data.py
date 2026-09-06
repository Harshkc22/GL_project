"""
Data Preparation
----------------
1. Loads the raw dataset directly from the Hugging Face dataset space.
2. Cleans the data (drops ID columns, fixes inconsistent category labels,
   imputes any missing values).
3. Splits the data into train and test sets.
4. Saves both splits locally.
5. Uploads the train/test splits back to the Hugging Face dataset space.
"""
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import hf_hub_download, HfApi

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HF_USERNAME = "harshkc"
DATASET_REPO_ID = f"{HF_USERNAME}/tourism-wellness-dataset"
TARGET_COL = "ProdTaken"

BASE_DIR = Path(__file__).resolve().parent.parent  # tourism_project/
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

token = os.environ.get("HF_TOKEN")


def load_raw_data() -> pd.DataFrame:
    """Loads the raw dataset directly from the Hugging Face Hub."""
    raw_path = hf_hub_download(
        repo_id=DATASET_REPO_ID,
        filename="tourism.csv",
        repo_type="dataset",
        token=token,
    )
    return pd.read_csv(raw_path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drops unnecessary columns and fixes known data-quality issues."""
    drop_cols = [c for c in ["Unnamed: 0", "CustomerID"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    # Fix inconsistent category labels found in the raw data
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
    df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]

    # Impute any missing values
    for c in num_cols:
        if df[c].isna().sum() > 0:
            df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        if df[c].isna().sum() > 0:
            df[c] = df[c].fillna(df[c].mode()[0])

    return df


def main():
    df = load_raw_data()
    print("Raw data shape:", df.shape)

    df = clean_data(df)
    print("Cleaned data shape:", df.shape)
    print(df[TARGET_COL].value_counts())

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    train_df = pd.concat([Xtrain, ytrain], axis=1)
    test_df = pd.concat([Xtest, ytest], axis=1)

    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    # Upload the train/test splits back to the Hugging Face dataset space
    api = HfApi(token=token)
    api.upload_file(path_or_fileobj=str(train_path), path_in_repo="train.csv",
                     repo_id=DATASET_REPO_ID, repo_type="dataset", token=token)
    api.upload_file(path_or_fileobj=str(test_path), path_in_repo="test.csv",
                     repo_id=DATASET_REPO_ID, repo_type="dataset", token=token)

    print(f"Train/test datasets uploaded to: https://huggingface.co/datasets/{DATASET_REPO_ID}")


if __name__ == "__main__":
    main()
