import unittest
from study import run_study


class StudyTests(unittest.TestCase):
    def test_seeded_study_is_reproducible_and_bounded(self) -> None:
        first, second = run_study(20), run_study(20)
        self.assertEqual(first.best_params, second.best_params)
        self.assertEqual(first.best_value, second.best_value)
        self.assertEqual(len(first.trials), 20)
        self.assertEqual(first.best_trial.user_attrs["objective_version"], "v1")

    def test_conditional_parameter(self) -> None:
        study = run_study(30)
        for trial in study.trials:
            self.assertEqual("penalty" in trial.params, trial.params["family"] == "offset")

    def test_invalid_budget_fails(self) -> None:
        with self.assertRaises(ValueError):
            run_study(0)


if __name__ == "__main__":
    unittest.main()
