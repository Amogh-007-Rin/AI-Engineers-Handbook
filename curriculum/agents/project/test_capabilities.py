import unittest

from capabilities import Budget, Capability, authorize


class CapabilityTests(unittest.TestCase):
    def test_authorized_call_consumes_budget(self) -> None:
        budget = Budget(1)
        authorize(Capability("read", frozenset({"ticket:1"})), "read", "ticket:1", False, budget)
        self.assertEqual(budget.remaining_calls, 0)

    def test_scope_and_write_fail_closed(self) -> None:
        capability = Capability("read", frozenset({"ticket:1"}))
        with self.assertRaises(PermissionError):
            authorize(capability, "read", "ticket:2", False, Budget(1))
        with self.assertRaises(PermissionError):
            authorize(capability, "read", "ticket:1", True, Budget(1))

    def test_exhausted_budget_fails(self) -> None:
        with self.assertRaises(RuntimeError):
            authorize(Capability("read", frozenset({"x"})), "read", "x", False, Budget(0))


if __name__ == "__main__":
    unittest.main()
