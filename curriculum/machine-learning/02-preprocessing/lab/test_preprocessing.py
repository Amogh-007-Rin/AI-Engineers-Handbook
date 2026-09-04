import math
import unittest

from preprocessing import PipelineState, dumps, feature_names, fit, loads, transform, transform_one


TRAINING = [{"age": 20, "plan": "basic"}, {"age": 30, "plan": "pro"}, {"age": 40, "plan": "basic"}]


class PreprocessingTests(unittest.TestCase):
    def test_fit_learns_only_declared_state(self):
        state = fit(TRAINING)
        self.assertEqual(state.age_median, 30)
        self.assertEqual(state.age_scale, 10)
        self.assertEqual(state.plans, ("basic", "pro"))

    def test_fit_rejects_empty_and_all_missing_age(self):
        with self.assertRaisesRegex(ValueError, "training records"):
            fit([])
        with self.assertRaisesRegex(ValueError, "all-missing"):
            fit([{"age": None, "plan": "basic"}])

    def test_schema_is_exact(self):
        with self.assertRaisesRegex(ValueError, "schema mismatch"):
            fit([{"age": 20, "plan": "basic", "target": 1}])
        with self.assertRaisesRegex(ValueError, "schema mismatch"):
            transform_one({"age": 20}, fit(TRAINING))

    def test_numeric_boundary_rejects_bool_and_nonfinite(self):
        for age in (True, math.nan, math.inf):
            with self.subTest(age=age), self.assertRaises(ValueError):
                transform_one({"age": age, "plan": "basic"}, fit(TRAINING))

    def test_missing_and_unknown_have_explicit_features(self):
        state = fit(TRAINING)
        names = feature_names(state)
        values = transform_one({"age": None, "plan": "enterprise"}, state)
        self.assertEqual(dict(zip(names, values))["age_missing"], 1)
        self.assertEqual(dict(zip(names, values))["plan=other"], 1)

    def test_output_order_and_values_are_deterministic(self):
        state = fit(list(reversed(TRAINING)))
        self.assertEqual(feature_names(state), ("age_scaled", "age_missing", "plan=basic", "plan=pro", "plan=other"))
        self.assertEqual(transform([{"age": 40, "plan": "pro"}], state), [(1.0, 0.0, 0.0, 1.0, 0.0)])

    def test_reserved_category_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "reserved"):
            fit([{"age": 20, "plan": "__other__"}])

    def test_json_round_trip_and_compatibility(self):
        state = fit(TRAINING)
        self.assertEqual(loads(dumps(state)), state)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            transform_one({"age": 20, "plan": "basic"}, PipelineState(99, 30, 10, ("basic",)))
        with self.assertRaisesRegex(ValueError, "invalid"):
            loads('{"version": 1}')


if __name__ == "__main__":
    unittest.main()
