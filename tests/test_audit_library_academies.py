import importlib.util
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("audit", Path(__file__).parents[1] / "scripts" / "audit_library_academies.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class LibraryAuditTests(unittest.TestCase):
    def test_complete_academy_requires_every_artifact_class(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "README.md").write_text("[Primary source registry](../SOURCES.md)\n" + "g" * 1000)
            (root / "01-foundations").mkdir()
            (root / "01-foundations" / "README.md").write_text("---\n" + "l" * 800)
            for child in ("exercises", "projects", "environment"): (root / child).mkdir()
            (root / "exercises" / "README.md").write_text("e" * 150)
            (root / "projects" / "model.py").write_text("m" * 100)
            (root / "projects" / "test_model.py").write_text("t" * 100)
            (root / "projects" / "solution.md").write_text("s" * 180)
            (root / "assessment.md").write_text("a" * 120)
            (root / "environment" / "requirements.txt").write_text("")
            self.assertTrue(all(MODULE.evidence(root).values()))

    def test_academy_requires_primary_source_registry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "README.md").write_text("g" * 1200)
            self.assertFalse(MODULE.evidence(root)["sources"])

    def test_project_readme_is_not_executable_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "projects").mkdir()
            (root / "projects" / "README.md").write_text("instructions")
            self.assertFalse(MODULE.evidence(root)["project"])


if __name__ == "__main__": unittest.main()
