import tempfile
import unittest
from pathlib import Path

from study_plan import StudyBlock, environment_diagnostic, validate_blocks


class StudyPlanTests(unittest.TestCase):
    def test_reports_target_and_recovery(self):
        result = validate_blocks(
            [StudyBlock("Monday", 60, "learn"), StudyBlock("Saturday", 120, "build")],
            weekly_target_minutes=240,
        )
        self.assertEqual(result["scheduled_minutes"], 180)
        self.assertEqual(result["recovery_minutes"], 60)
        self.assertFalse(result["target_met"])

    def test_accepts_a_met_target(self):
        result = validate_blocks([StudyBlock("Sunday", 90, "review")], 90)
        self.assertTrue(result["target_met"])

    def test_rejects_empty_or_implausible_blocks(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            validate_blocks([], 60)
        with self.assertRaisesRegex(ValueError, "between 15 and 240"):
            validate_blocks([StudyBlock("Monday", 5, "practice")], 60)
        with self.assertRaisesRegex(ValueError, "day and activity"):
            validate_blocks([StudyBlock("", 60, "practice")], 60)

    def test_diagnostic_names_missing_repository_files(self):
        with tempfile.TemporaryDirectory() as directory:
            result = environment_diagnostic(Path(directory))
        self.assertFalse(result["ready"])
        self.assertEqual(
            result["missing_required_files"],
            ["README.md", "project.md", "scripts/validate_content.py"],
        )


if __name__ == "__main__":
    unittest.main()
