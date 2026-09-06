"""
Hosting
-------
Pushes the deployment files (Dockerfile, app.py, requirements.txt) to a
Hugging Face Space so that the Streamlit prediction app is hosted and
publicly reachable.
"""
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

HF_USERNAME = "harshkc"
SPACE_REPO_ID = f"{HF_USERNAME}/tourism-wellness-app"

BASE_DIR = Path(__file__).resolve().parent  # tourism_project/deployment/
token = os.environ.get("HF_TOKEN")


def main():
    api = HfApi(token=token)

    # Create the Space (Docker SDK, since we ship our own Dockerfile)
    create_repo(
        repo_id=SPACE_REPO_ID,
        repo_type="space",
        space_sdk="docker",
        private=False,
        exist_ok=True,
        token=token,
    )

    # Push every file needed to run the app (Dockerfile, app.py, requirements.txt)
    api.upload_folder(
        folder_path=str(BASE_DIR),
        repo_id=SPACE_REPO_ID,
        repo_type="space",
        token=token,
    )

    print(f"App deployed at: https://huggingface.co/spaces/{SPACE_REPO_ID}")


if __name__ == "__main__":
    main()
