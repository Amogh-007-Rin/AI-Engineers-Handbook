#!/usr/bin/env python3
"""Offline release-hygiene scan for secrets, caches, and large artifacts."""

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules"}
TEXT_SUFFIXES = {".md", ".py", ".json", ".yml", ".yaml", ".txt", ".toml", ".ini", ".cfg", ".csv"}
SECRET_PATTERNS = {
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "OpenAI-style key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "GitHub token": re.compile(r"\bgh[opusr]_[A-Za-z0-9]{30,}\b"),
}
FORBIDDEN_SUFFIXES = {".pyc", ".pyo", ".pkl", ".pt", ".pth", ".onnx", ".ckpt", ".safetensors"}


def files(root=ROOT):
    for path in root.rglob("*"):
        if path.is_file() and not any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            yield path


def scan(root=ROOT, maximum_bytes=1_000_000):
    errors = []
    for path in files(root):
        relative = path.relative_to(root)
        if path.suffix in FORBIDDEN_SUFFIXES:
            errors.append(f"{relative}: generated/model artifact is forbidden")
        if path.stat().st_size > maximum_bytes and path.suffix.lower() != ".pdf":
            errors.append(f"{relative}: exceeds {maximum_bytes} bytes")
        if path.suffix.lower() in TEXT_SUFFIXES:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                errors.append(f"{relative}: expected UTF-8 text")
                continue
            for label, pattern in SECRET_PATTERNS.items():
                if pattern.search(text):
                    errors.append(f"{relative}: possible {label}")
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-bytes", type=int, default=1_000_000)
    args = parser.parse_args()
    errors = scan(maximum_bytes=args.max_bytes)
    if errors:
        print("Repository hygiene scan failed:")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print("Repository hygiene scan passed: no secrets or forbidden/oversized artifacts found.")
    return 0


if __name__ == "__main__": raise SystemExit(main())
