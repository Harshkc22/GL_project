"""
Data Registration
------------------
Uploads the raw tourism.csv file to a Hugging Face Dataset repository so that
every later stage of the pipeline (data prep, training, CI/CD) can pull the
data directly from the Hugging Face Hub instead of relying on local files.

Requires the HF_TOKEN environment variable (a Hugging Face access token with
write permission) to be set before running.
"""
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

# ---------------------------------------------------------------------------
# Configuration - update HF_USERNAME with your own Hugging Face username
# ---------------------------------------------------------------------------
HF_USERNAME = "harshkc"
DATASET_REPO_ID = f"{HF_USERNAME}/tourism-wellness-dataset"

BASE_DIR = Path(__file__).resolve().parent.parent  # tourism_project/
RAW_DATA_PATH = BASE_DIR / "data" / "tourism.csv"

token = os.environ.get("HF_TOKEN")


def main():
    api = HfApi(token=token)

    # Create the dataset repo if it does not already exist
    create_repo(
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        private=False,
        exist_ok=True,
        token=token,
    )

    # Upload the raw csv file to the dataset repo
    api.upload_file(
        path_or_fileobj=str(RAW_DATA_PATH),
        path_in_repo="tourism.csv",
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        token=token,
    )

    print(f"Raw dataset registered at: https://huggingface.co/datasets/{DATASET_REPO_ID}")


if __name__ == "__main__":
    main()
