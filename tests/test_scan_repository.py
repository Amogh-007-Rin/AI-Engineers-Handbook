import importlib.util
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("scan", Path(__file__).parents[1] / "scripts" / "scan_repository.py")
MODULE = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MODULE)


class RepositoryScanTests(unittest.TestCase):
    def test_clean_text_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "README.md").write_text("safe documentation")
            self.assertEqual(MODULE.scan(root), [])

    def test_secret_and_artifact_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.txt").write_text("token=" + "sk-" + "a" * 22)
            (root / "weights.pt").write_bytes(b"model")
            errors = MODULE.scan(root)
            self.assertTrue(any("OpenAI-style key" in error for error in errors))
            self.assertTrue(any("forbidden" in error for error in errors))


if __name__ == "__main__": unittest.main()
