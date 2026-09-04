"""Dependency-free orientation lab: validate a study plan and local runtime."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import platform
from pathlib import Path
import sys
from typing import Iterable


@dataclass(frozen=True)
class StudyBlock:
    day: str
    minutes: int
    activity: str


def validate_blocks(blocks: Iterable[StudyBlock], weekly_target_minutes: int) -> dict[str, object]:
    """Validate a realistic schedule and return an immutable evidence summary."""
    items = tuple(blocks)
    if weekly_target_minutes <= 0:
        raise ValueError("weekly_target_minutes must be positive")
    if not items:
        raise ValueError("at least one study block is required")
    for block in items:
        if not block.day.strip() or not block.activity.strip():
            raise ValueError("every block needs a day and activity")
        if not 15 <= block.minutes <= 240:
            raise ValueError("each block must last between 15 and 240 minutes")

    scheduled = sum(block.minutes for block in items)
    return {
        "blocks": [asdict(block) for block in items],
        "scheduled_minutes": scheduled,
        "target_minutes": weekly_target_minutes,
        "target_met": scheduled >= weekly_target_minutes,
        "recovery_minutes": max(0, weekly_target_minutes - scheduled),
    }


def environment_diagnostic(repository: Path) -> dict[str, object]:
    """Report observable setup facts without changing the learner's machine."""
    required = ("README.md", "project.md", "scripts/validate_content.py")
    missing = [name for name in required if not (repository / name).is_file()]
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.system(),
        "repository": str(repository.resolve()),
        "missing_required_files": missing,
        "ready": sys.version_info >= (3, 10) and not missing,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate handbook orientation evidence")
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--target-minutes", type=int, default=600)
    parser.add_argument(
        "--block",
        action="append",
        nargs=3,
        metavar=("DAY", "MINUTES", "ACTIVITY"),
        default=[],
        help="repeatable study block, for example: --block Mon 60 practice",
    )
    args = parser.parse_args(argv)
    try:
        blocks = [StudyBlock(day, int(minutes), activity) for day, minutes, activity in args.block]
        result = {
            "environment": environment_diagnostic(args.repository),
            "plan": validate_blocks(blocks, args.target_minutes),
        }
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["environment"]["ready"] and result["plan"]["target_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
