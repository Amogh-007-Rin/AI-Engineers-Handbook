import unittest
import httpx

from app import app, repository


class ApiTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        repository.records.clear()
        repository.next_id = 1

    async def request(self, method: str, path: str, **kwargs) -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.request(method, path, **kwargs)

    async def test_health(self) -> None:
        response = await self.request("GET", "/healthz")
        self.assertEqual(response.json(), {"status": "ok"})

    async def test_authentication_fails_closed(self) -> None:
        response = await self.request("POST", "/predictions", json={"features": [1]})
        self.assertEqual(response.status_code, 401)

    async def test_creation_is_idempotent(self) -> None:
        headers = {"X-API-Key": "test-secret", "Idempotency-Key": "request-1"}
        first = await self.request("POST", "/predictions", headers=headers, json={"features": [1, 3]})
        second = await self.request("POST", "/predictions", headers=headers, json={"features": [9]})
        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.json(), second.json())
        self.assertEqual(first.json()["score"], 2)

    async def test_input_and_idempotency_contracts(self) -> None:
        auth = {"X-API-Key": "test-secret"}
        empty = await self.request("POST", "/predictions", headers=auth, json={"features": []})
        no_key = await self.request("POST", "/predictions", headers=auth, json={"features": [1]})
        self.assertEqual(empty.status_code, 422)
        self.assertEqual(no_key.status_code, 400)


if __name__ == "__main__":
    unittest.main()
