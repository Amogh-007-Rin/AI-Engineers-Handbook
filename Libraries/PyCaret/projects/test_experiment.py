import unittest
import pandas as pd
from experiment import train, validate_inference


class PyCaretTests(unittest.TestCase):
    def test_real_experiment_predicts(self):
        experiment, model = train()
        result = experiment.predict_model(model, data=pd.DataFrame({"age": [40], "income": [50], "country": ["FR"]}), verbose=False)
        self.assertEqual(len(result), 1)
        self.assertIn("prediction_label", result.columns)

    def test_schema_fails_closed(self):
        with self.assertRaises(ValueError):
            validate_inference(pd.DataFrame({"age": [1], "income": [2], "country": ["GB"], "target": [1]}))


if __name__ == "__main__": unittest.main()
