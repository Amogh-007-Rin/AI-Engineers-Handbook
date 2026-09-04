import importlib.util
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "learning_system_audit",
    Path(__file__).parents[1] / "scripts" / "audit_learning_system.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class LearningSystemAuditTests(unittest.TestCase):
    def test_repository_has_every_cross_cutting_surface(self):
        self.assertTrue(all(MODULE.inspect().values()))

    def test_empty_repository_fails_every_surface(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(any(MODULE.inspect(Path(directory)).values()))

    def test_project_ladder_requires_exactly_steps_one_to_fourteen(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "projects").mkdir()
            (root / "projects" / "README.md").write_text(
                "# Projects\n" + "x" * 2100 + "\n| 1 | only one | missing |\n",
                encoding="utf-8",
            )
            self.assertFalse(MODULE.inspect(root)["fourteen project steps"])


if __name__ == "__main__":
    unittest.main()
