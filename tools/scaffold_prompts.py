"""Scaffold prompt files for a new provider."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

PROMPT_REL_FILES = [
    # system prompts
    Path("system/category/chat-system.txt"),
    Path("system/sub-category/chat-system.txt"),
    # user prompts: label/category with icl-demo variants
    Path("user/label/category/icl-demo/none/classify-questions.txt"),
    Path("user/label/category/icl-demo/category/classify-questions.txt"),
    Path("user/label/category/icl-demo/sub-category/classify-questions.txt"),
    # user prompts: label/sub-category with icl-demo variants
    Path("user/label/sub-category/icl-demo/none/classify-questions.txt"),
    Path("user/label/sub-category/icl-demo/category/classify-questions.txt"),
    Path("user/label/sub-category/icl-demo/sub-category/classify-questions.txt"),
]

STUB_SYSTEM = """# TODO: System prompt for {provider}
# Add taxonomy definitions, constraints, and output requirements here.
"""

STUB_USER = """# TODO: User prompt for {provider}
# Keep placeholders consistent with your prompt formatting pipeline.
"""


# scaffold empty prompt files with stubs
def scaffold_empty(dest_root: Path, provider: str, force: bool) -> None:
    if dest_root.exists() and any(dest_root.iterdir()) and not force:
        raise SystemExit(f"Destination exists and is not empty: {dest_root} (use --force)")

    for rel in PROMPT_REL_FILES:
        path = dest_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not force:
            continue
        if rel.name == "chat-system.txt":
            path.write_text(STUB_SYSTEM.format(provider=provider), encoding="utf-8")
        else:
            path.write_text(STUB_USER.format(provider=provider), encoding="utf-8")


# scaffold prompt files by copying from existing template
def scaffold_from_template(prompts_root: Path, provider: str, template: str, force: bool) -> None:
    src = prompts_root / template
    dest = prompts_root / provider

    if not src.exists():
        raise SystemExit(f"Template provider folder not found: {src}")

    if dest.exists():
        if force:
            shutil.rmtree(dest)
        else:
            raise SystemExit(f"Destination exists: {dest} (use --force to overwrite)")

    shutil.copytree(src, dest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True, help="New provider folder name (e.g., google, bedrock, ollama)")
    ap.add_argument("--prompts-root", default="prompts", help="Prompts root directory (default: prompts)")
    ap.add_argument("--from", dest="template", default=None, help="Copy prompts from existing provider folder")
    ap.add_argument("--empty", action="store_true", help="Create folder structure with stub files")
    ap.add_argument("--force", action="store_true", help="Overwrite if destination exists")
    args = ap.parse_args()

    prompts_root = Path(args.prompts_root)
    dest_root = prompts_root / args.provider

    if args.template:
        scaffold_from_template(prompts_root, args.provider, args.template, args.force)
        print(f"Copied prompts from {args.template!r} -> {args.provider!r}: {dest_root}")
        return
    if args.empty:
        scaffold_empty(dest_root, args.provider, args.force)  # provider used in stubs
        print(f"Scaffolded empty prompt tree at: {dest_root}")
        return

    raise SystemExit("Choose one: --from <provider> or --empty")


if __name__ == "__main__":
    main()
