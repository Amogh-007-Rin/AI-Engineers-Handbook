import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "validate_content.py"
SPEC = importlib.util.spec_from_file_location("validate_content", MODULE_PATH)
assert SPEC and SPEC.loader
validator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validator
SPEC.loader.exec_module(validator)


class MetadataParserTests(unittest.TestCase):
    def test_parses_supported_front_matter(self) -> None:
        text = """---
title: Arrays
slug: arrays
level: foundation
prerequisites:
  - python-basics
estimated_hours: 2
---
# Arrays
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lesson.md"
            path.write_text(text, encoding="utf-8")
            result = validator.parse_front_matter(path)
        self.assertEqual(result["slug"], "arrays")
        self.assertEqual(result["prerequisites"], ["python-basics"])
        self.assertEqual(result["estimated_hours"], 2)

    def test_rejects_unclosed_front_matter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lesson.md"
            path.write_text("---\ntitle: Broken\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "closing"):
                validator.parse_front_matter(path)


class MetadataValidationTests(unittest.TestCase):
    def test_rejects_invalid_slug(self) -> None:
        doc = validator.Document(Path("lesson.md"), {
            "title": "Lesson", "slug": "Not Valid", "level": "foundation",
            "stage": "foundations", "estimated_hours": 1, "prerequisites": [],
            "learning_objectives": ["Build a useful thing"], "formats": ["lesson"],
            "compute": "cpu", "status": "draft", "last_verified": "2026-09-03",
        })
        self.assertIn("slug must be lowercase kebab-case", validator.validate_metadata(doc))


class LibraryCoverageTests(unittest.TestCase):
    def test_every_academy_has_substantive_navigation(self) -> None:
        libraries = Path(__file__).parents[1] / "Libraries"
        excluded = {"Z-Roadmap"}
        academies = [path for path in libraries.iterdir() if path.is_dir() and path.name not in excluded]
        self.assertEqual(len(academies), 60)
        for academy in academies:
            candidates = [academy / "README.md", academy / "readme.md"]
            readme = next((path for path in candidates if path.exists()), None)
            self.assertIsNotNone(readme, academy.name)
            text = readme.read_text(encoding="utf-8")
            self.assertGreater(len(text), 1_000, academy.name)
            self.assertIn("## Learning path" if academy.name in {"NumPy", "Pandas"} else "## Level 1", text, academy.name)


if __name__ == "__main__":
    unittest.main()
