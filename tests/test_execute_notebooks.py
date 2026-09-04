import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("notebooks", Path(__file__).parents[1] / "scripts" / "execute_notebooks.py")
MODULE = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MODULE)


class NotebookRunnerTests(unittest.TestCase):
    def make(self, outputs=None):
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".ipynb", delete=False)
        json.dump({"nbformat": 4, "cells": [{"cell_type": "code", "outputs": outputs or [], "source": ["x = 4\n", "assert x == 4\n"]}]}, handle)
        handle.close(); self.addCleanup(Path(handle.name).unlink)
        return Path(handle.name)

    def test_executes(self): MODULE.execute(self.make())

    def test_rejects_outputs(self):
        with self.assertRaises(ValueError): MODULE.execute(self.make([{"output_type": "stream"}]))


if __name__ == "__main__": unittest.main()
