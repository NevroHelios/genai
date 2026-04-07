from huggingface_hub import HfApi, create_repo, upload_file
from pathlib import Path
from src.config import CFG 

repo_id = f"{CFG.HF_USERNAME}/{CFG.REPO_NAME}"

api = HfApi()

if __name__ == '__main__':
    create_repo(
        repo_id=repo_id,
        repo_type="model",
        exist_ok=True,
        private=False
    )

    upload_file(
        path_or_fileobj=CFG.MODEL_PATH,
        path_in_repo=Path(CFG.MODEL_PATH).name,
        repo_id=repo_id,
        repo_type="model"
    )
