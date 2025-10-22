"""Utility for exporting SpiralReality artifacts to the Hugging Face Hub."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

from huggingface_hub import HfApi, HfFolder, create_repo, upload_folder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "repository",
        help="Target repository on the Hugging Face Hub (e.g. user/model-name)",
    )
    parser.add_argument(
        "--path",
        default="dist",
        help="Path to the directory containing artifacts to upload",
    )
    parser.add_argument(
        "--repo-type",
        default="model",
        choices=["model", "dataset"],
        help="Repository type to create/update",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repository as private",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token. Defaults to the cached token",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Optional JSON file describing model card metadata",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload SpiralReality artifacts",
        help="Commit message to use for the upload",
    )
    return parser.parse_args()


def load_metadata(path: Optional[Path]) -> Optional[dict]:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_token(token: Optional[str]) -> str:
    if token:
        return token
    cached = HfFolder.get_token()
    if cached:
        return cached
    raise RuntimeError(
        "No Hugging Face token provided. Pass --token or run 'huggingface-cli login'."
    )


def ensure_repo(api: HfApi, repo_id: str, repo_type: str, private: bool) -> None:
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
    except Exception:
        create_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            private=private,
            exist_ok=True,
        )


def upload_artifacts(
    repo_id: str,
    repo_type: str,
    path: str,
    commit_message: str,
    token: str,
    metadata: Optional[dict],
) -> None:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Path {path_obj} does not exist")

    if metadata:
        model_card = path_obj / "README.md"
        card_lines = [metadata.get("model_card", "# SpiralReality AIT")]
        tags = metadata.get("tags")
        if tags:
            card_lines.append("\n**Tags:** " + ", ".join(tags))
        datasets = metadata.get("datasets")
        if datasets:
            card_lines.append("\n**Datasets:** " + ", ".join(datasets))
        license_name = metadata.get("license")
        if license_name:
            card_lines.append(f"\n**License:** {license_name}")
        languages = metadata.get("language")
        if languages:
            card_lines.append("\n**Languages:** " + ", ".join(languages))
        model_card.write_text("\n".join(card_lines), encoding="utf-8")

    upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=str(path_obj),
        commit_message=commit_message,
        token=token,
        path_in_repo="",
    )


def main() -> None:
    args = parse_args()
    token = ensure_token(args.token)
    api = HfApi(token=token)

    ensure_repo(api, args.repository, args.repo_type, args.private)

    metadata = load_metadata(args.metadata)

    upload_artifacts(
        repo_id=args.repository,
        repo_type=args.repo_type,
        path=args.path,
        commit_message=args.commit_message,
        token=token,
        metadata=metadata,
    )

    print(f"Uploaded contents of {args.path} to {args.repo_type} repo {args.repository}")


if __name__ == "__main__":
    main()
