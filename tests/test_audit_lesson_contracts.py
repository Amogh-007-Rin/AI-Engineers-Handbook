import importlib.util
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "lesson_contracts",
    Path(__file__).parents[1] / "scripts" / "audit_lesson_contracts.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class LessonContractAuditTests(unittest.TestCase):
    def test_completed_foundations_lessons_pass(self):
        root = MODULE.ROOT / "curriculum" / "foundations"
        for relative in ("00-orientation/README.md", "01-python-foundations/README.md"):
            with self.subTest(relative=relative):
                self.assertTrue(all(MODULE.evidence(root / relative).values()))

    def test_short_outline_exposes_contract_gaps(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "README.md"
            path.write_text("---\nformats:\n  - lesson\n---\n# Outline\n", encoding="utf-8")
            evidence = MODULE.evidence(path)
            self.assertFalse(any(evidence.values()))
            self.assertTrue(MODULE.is_lesson(path))

    def test_non_lesson_metadata_is_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "README.md"
            path.write_text("---\nformats:\n  - project\n---\n# Project\n", encoding="utf-8")
            self.assertFalse(MODULE.is_lesson(path))


if __name__ == "__main__":
    unittest.main()
