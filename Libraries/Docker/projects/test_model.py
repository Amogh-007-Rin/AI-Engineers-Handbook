import unittest
from model import validate_service


GOOD = {"image": "registry.example/model@sha256:" + "a" * 64, "command": ["serve"],
        "healthcheck": ["/healthz"], "user": "10001", "resources": {"cpus": 1, "memory": "1Gi"}}


class DockerProjectTests(unittest.TestCase):
    def test_valid_service(self): self.assertTrue(validate_service(GOOD))

    def test_reproducibility_and_security_gates(self):
        for key, value in (("image", "model:latest"), ("user", "root")):
            bad = {**GOOD, key: value}
            with self.assertRaises(ValueError): validate_service(bad)

    def test_required_limits(self):
        with self.assertRaises(ValueError): validate_service({**GOOD, "resources": {"cpus": 1}})


if __name__ == "__main__": unittest.main()
