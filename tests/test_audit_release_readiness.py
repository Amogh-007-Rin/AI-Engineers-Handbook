import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "audit_release_readiness.py"
SPEC = importlib.util.spec_from_file_location("audit_release_readiness", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ReleaseReadinessTests(unittest.TestCase):
    def test_repository_inventory_contains_all_sixty_academies(self):
        self.assertEqual(len(MODULE.academy_names()), 60)

    def test_empty_manifest_reports_every_external_gate(self):
        manifest = {"schema_version": 1, "release_candidate": None, "commit": None,
                    "workflow_runs": [], "reviews": [], "learner_journeys": []}
        pending = MODULE.audit(manifest, academies=["NumPy"], workflows={"quality.yml"})
        self.assertTrue(any("quality.yml" in item for item in pending))
        self.assertTrue(any("technical review" in item for item in pending))
        self.assertTrue(any("practitioner learner" in item for item in pending))

    def test_complete_fixture_passes(self):
        commit = "abcdef1"
        reviews = [
            {"review_type": kind, "decision": "approved", "independent": True,
             "commit": commit, "academies": ["NumPy"]}
            for kind in {"technical", "pedagogy", *MODULE.REQUIRED_GLOBAL_REVIEWS}
        ]
        manifest = {
            "schema_version": 1, "release_candidate": "v1.0.0", "commit": commit,
            "workflow_runs": [{"workflow": "quality.yml", "conclusion": "success",
                               "commit": commit,
                               "url": "https://github.com/example/project/actions/runs/123"}],
            "reviews": reviews,
            "learner_journeys": [
                {"academy": "NumPy", "level": level, "result": "passed",
                 "blocking_feedback_resolved": True, "commit": commit,
                 "evidence": "reports/journeys/numpy.md"}
                for level in ("foundation", "practitioner")
            ],
        }
        self.assertEqual(MODULE.audit(manifest, academies=["NumPy"],
                                      workflows={"quality.yml"}), [])


if __name__ == "__main__":
    unittest.main()
