"""Deploy this app to a Hugging Face Space (Docker SDK).

Usage (locally or from GitHub Actions):
    HF_TOKEN=hf_xxx HF_SPACE=your-username/icu-mortality-xai python deploy/push_to_space.py

Creates the Space if needed and uploads only what the container needs, with a
Space-specific README (Hugging Face reads its settings from the YAML header).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent

SPACE_README = """---
title: ICU Mortality Prediction (Explainable)
emoji: 🩺
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Explainable multimodal ICU mortality prediction (research)
---

# ICU Mortality Prediction - Explainable Clinical NLP

BioBERT clinical-note encoder + vitals/labs network, with attention, SHAP and LIME
explanations. **Research prototype - not for clinical use.**

Source code: https://github.com/{repo}
"""


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    space = os.environ.get("HF_SPACE")
    if not token or not space:
        print("Set HF_TOKEN and HF_SPACE (e.g. username/icu-mortality-xai).", file=sys.stderr)
        return 1

    api = HfApi(token=token)
    api.create_repo(repo_id=space, repo_type="space", space_sdk="docker", exist_ok=True)

    # Optional: forward model-location settings to the Space as variables.
    for name in ("MODEL_HF_REPO", "MODEL_URL", "MODEL_SHA256", "THRESHOLD"):
        value = os.environ.get(name)
        if value:
            api.add_space_variable(repo_id=space, key=name, value=value)

    github_repo = os.environ.get(
        "GITHUB_REPOSITORY", "kalavalashivaharikowshik/mimic-nlp-explainable-ai"
    )
    api.upload_file(
        path_or_fileobj=SPACE_README.format(repo=github_repo).encode(),
        path_in_repo="README.md",
        repo_id=space,
        repo_type="space",
        commit_message="Update Space README",
    )
    api.upload_folder(
        folder_path=str(ROOT),
        repo_id=space,
        repo_type="space",
        allow_patterns=["Dockerfile", ".dockerignore", "backend/**", "frontend/**"],
        ignore_patterns=["**/__pycache__/**", "backend/models/**", "*.pth"],
        delete_patterns=["backend/**", "frontend/**"],
        commit_message=f"Deploy {os.environ.get('GITHUB_SHA', 'local')[:7]}",
    )
    print(f"Deployed: https://huggingface.co/spaces/{space}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
