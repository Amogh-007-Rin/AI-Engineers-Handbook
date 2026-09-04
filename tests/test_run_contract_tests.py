import importlib.util
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("contracts", Path(__file__).parents[1] / "scripts" / "run_contract_tests.py")
MODULE = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MODULE)


class ContractRunnerTests(unittest.TestCase):
    def test_manifest_is_unique_and_complete_for_curriculum(self):
        self.assertEqual(len(MODULE.CURRICULUM), len(set(MODULE.CURRICULUM)))
        actual = {path.name for path in (MODULE.ROOT / "curriculum").iterdir() if path.is_dir()}
        self.assertEqual(set(MODULE.CURRICULUM), actual)

    def test_declared_suites_have_tests(self):
        for label, directory in MODULE.suites():
            with self.subTest(label=label):
                self.assertTrue(directory.is_dir())
                self.assertTrue(any(directory.glob("test*.py")))


if __name__ == "__main__": unittest.main()
