import unittest
from model import validate_deployment


def deployment():
    c = {"image": "registry.example/model@sha256:" + "b" * 64,
         "resources": {"requests": {"cpu": "100m"}, "limits": {"cpu": "1"}},
         "startupProbe": {}, "readinessProbe": {}, "livenessProbe": {}}
    return {"spec": {"template": {"spec": {"containers": [c]}}}}


class KubernetesProjectTests(unittest.TestCase):
    def test_valid_manifest(self): self.assertTrue(validate_deployment(deployment()))

    def test_probe_and_digest_gates(self):
        for key in ("startupProbe", "readinessProbe", "livenessProbe"):
            bad = deployment(); del bad["spec"]["template"]["spec"]["containers"][0][key]
            with self.assertRaises(ValueError): validate_deployment(bad)
        bad = deployment(); bad["spec"]["template"]["spec"]["containers"][0]["image"] = "model:latest"
        with self.assertRaises(ValueError): validate_deployment(bad)


if __name__ == "__main__": unittest.main()
