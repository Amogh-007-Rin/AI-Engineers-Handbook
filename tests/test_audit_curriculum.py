import importlib.util
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("curriculum_audit", Path(__file__).parents[1] / "scripts" / "audit_curriculum.py")
MODULE = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MODULE)


class CurriculumAuditTests(unittest.TestCase):
    def test_empty_track_has_no_completion_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(any(MODULE.evidence(Path(directory)).values()))

    def test_project_needs_code_test_and_solution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "project").mkdir()
            (root / "project" / "README.md").write_text("r" * 300)
            result = MODULE.evidence(root)
            self.assertFalse(result["project"]); self.assertFalse(result["tests"]); self.assertFalse(result["solution"])


if __name__ == "__main__": unittest.main()
