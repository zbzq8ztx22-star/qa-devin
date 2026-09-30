"""Executable tests for devin_api_client.py.

Runs the real DevinAPIClient against a controlled local aiohttp server so
no external network calls are made. Requires only stdlib + aiohttp, which
is already a project dependency.

Run: python -m unittest test_devin_api_client -v
"""

import math
import unittest

from aiohttp import web

from devin_api_client import DevinAPIClient, DevinAPIError

API_KEY = "test-secret-key-12345"
NOT_FOUND_BODY = '{"detail": "Session not found"}'


async def start_server(handler):
    """Start a local aiohttp server running `handler` on an ephemeral port.

    Returns (runner, base_url). Callers must `await runner.cleanup()`.
    """
    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    return runner, f"http://127.0.0.1:{port}"


def json_handler(status, body):
    async def handler(request):
        return web.Response(
            status=status, text=body, content_type="application/json"
        )

    return handler


class DevinAPIClientTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._runners = []

    async def asyncTearDown(self):
        for runner in self._runners:
            await runner.cleanup()

    async def client_for(self, handler, **kwargs):
        runner, base_url = await start_server(handler)
        self._runners.append(runner)
        return DevinAPIClient(API_KEY, base_url=base_url, **kwargs)

    # ---- happy paths ----

    async def test_check_auth_returns_auth_response(self):
        client = await self.client_for(
            json_handler(200, '{"status": "ok", "org_id": "org-1"}')
        )
        result = await client.check_auth()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["org_id"], "org-1")

    async def test_start_session_returns_session_response(self):
        client = await self.client_for(
            json_handler(
                200,
                '{"session_id": "sess-1", "url": "https://app.devin.ai/sessions/sess-1", "is_new_ongoing_session": false}',
            )
        )
        result = await client.start_session("prompt")
        self.assertEqual(result["session_id"], "sess-1")
        self.assertEqual(result["url"], "https://app.devin.ai/sessions/sess-1")

    async def test_get_session_status_returns_status_response(self):
        client = await self.client_for(
            json_handler(
                200,
                '{"session_id": "s", "status": "working", "status_enum": "running", "structured_output": null}',
            )
        )
        result = await client.get_session_status("s")
        self.assertEqual(result["status_enum"], "running")

    async def test_bearer_token_sent_in_authorization_header(self):
        seen = {}

        async def handler(request):
            seen["authorization"] = request.headers.get("Authorization")
            return web.Response(
                status=200,
                text='{"status": "ok", "org_id": "o"}',
                content_type="application/json",
            )

        client = await self.client_for(handler)
        await client.check_auth()
        self.assertEqual(seen["authorization"], f"Bearer {API_KEY}")

    # ---- timeout configuration ----

    def test_explicit_timeout_is_configured(self):
        client = DevinAPIClient(API_KEY, timeout_seconds=30)
        self.assertEqual(client._timeout.total, 30)

    def test_default_timeout_is_sixty_seconds(self):
        client = DevinAPIClient(API_KEY)
        self.assertEqual(client._timeout.total, 60)

    def test_invalid_timeout_values_rejected(self):
        for bad in (0, -5, -0.1, None, "30"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                DevinAPIClient(API_KEY, timeout_seconds=bad)

    def test_non_finite_timeout_values_rejected(self):
        for bad in (math.nan, math.inf, -math.inf):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                DevinAPIClient(API_KEY, timeout_seconds=bad)

    async def test_slow_response_raises_devin_api_error(self):
        import asyncio

        async def slow(request):
            await asyncio.sleep(2)
            return web.Response(
                status=200,
                text='{"status": "ok", "org_id": "o"}',
                content_type="application/json",
            )

        client = await self.client_for(slow, timeout_seconds=0.5)
        with self.assertRaises(DevinAPIError):
            await client.check_auth()

    # ---- HTTP error statuses ----

    async def test_http_401_raises_devin_api_error_with_status(self):
        client = await self.client_for(
            json_handler(401, '{"detail": "Invalid authentication credentials"}')
        )
        with self.assertRaises(DevinAPIError) as ctx:
            await client.check_auth()
        self.assertEqual(ctx.exception.status, 401)
        self.assertIn("Invalid authentication credentials", str(ctx.exception))

    async def test_http_500_raises_devin_api_error_with_status(self):
        client = await self.client_for(
            json_handler(500, '{"detail": "Internal Server Error"}')
        )
        with self.assertRaises(DevinAPIError) as ctx:
            await client.start_session("prompt")
        self.assertEqual(ctx.exception.status, 500)

    # ---- non-JSON / malformed responses ----

    async def test_non_json_response_raises_devin_api_error(self):
        async def handler(request):
            return web.Response(
                status=502, text="<html>Bad Gateway</html>", content_type="text/html"
            )

        client = await self.client_for(handler)
        with self.assertRaises(DevinAPIError):
            await client.get_session_status("s")

    # ---- network errors ----

    async def test_connection_refused_raises_devin_api_error(self):
        client = DevinAPIClient(
            API_KEY, base_url="http://127.0.0.1:1", timeout_seconds=5
        )
        with self.assertRaises(DevinAPIError) as ctx:
            await client.check_auth()
        self.assertIsNone(ctx.exception.status)

    # ---- 2xx with error payload / invalid contract ----

    async def test_200_with_error_detail_raises_devin_api_error(self):
        client = await self.client_for(
            json_handler(200, '{"detail": "Unexpected API state"}')
        )
        with self.assertRaises(DevinAPIError):
            await client.get_session_status("s")

    async def test_check_auth_missing_required_key_raises(self):
        client = await self.client_for(json_handler(200, '{"org_id": "o"}'))
        with self.assertRaises(DevinAPIError):
            await client.check_auth()

    async def test_start_session_missing_url_raises(self):
        client = await self.client_for(
            json_handler(200, '{"session_id": "sess-1"}')
        )
        with self.assertRaises(DevinAPIError):
            await client.start_session("prompt")

    async def test_get_session_status_missing_status_enum_raises(self):
        client = await self.client_for(
            json_handler(200, '{"session_id": "s", "structured_output": null}')
        )
        with self.assertRaises(DevinAPIError):
            await client.get_session_status("s")

    async def test_non_dict_json_payload_raises(self):
        client = await self.client_for(json_handler(200, '["not", "a", "dict"]'))
        with self.assertRaises(DevinAPIError):
            await client.get_session_status("s")

    # ---- preserved semantics ----

    async def test_session_not_found_404_returns_none(self):
        client = await self.client_for(
            json_handler(404, NOT_FOUND_BODY)
        )
        result = await client.get_session_status("missing-session")
        self.assertIsNone(result)

    async def test_session_not_found_200_returns_none(self):
        client = await self.client_for(
            json_handler(200, NOT_FOUND_BODY)
        )
        result = await client.get_session_status("missing-session")
        self.assertIsNone(result)

    async def test_404_with_other_detail_raises(self):
        client = await self.client_for(
            json_handler(404, '{"detail": "Not Found"}')
        )
        with self.assertRaises(DevinAPIError):
            await client.get_session_status("s")

    async def test_500_containing_session_not_found_substring_raises(self):
        client = await self.client_for(
            json_handler(
                500, '{"detail": "Unable to check: Session not found in cache"}'
            )
        )
        with self.assertRaises(DevinAPIError) as ctx:
            await client.get_session_status("s")
        self.assertEqual(ctx.exception.status, 500)

    async def test_404_non_json_body_containing_substring_raises(self):
        async def handler(request):
            return web.Response(
                status=404,
                text="Session not found somewhere upstream",
                content_type="text/plain",
            )

        client = await self.client_for(handler)
        with self.assertRaises(DevinAPIError):
            await client.get_session_status("s")

    # ---- malformed response bytes ----

    async def test_invalid_bytes_with_declared_charset_raises(self):
        async def handler(request):
            return web.Response(
                status=200,
                body=b"\xff\xfe\x00not-valid-utf8",
                headers={"Content-Type": "application/json; charset=utf-8"},
            )

        client = await self.client_for(handler)
        with self.assertRaises(DevinAPIError):
            await client.check_auth()

    # ---- redirect credential hygiene ----

    async def test_cross_origin_redirect_does_not_forward_authorization(self):
        captured = {}

        async def handler_b(request):
            captured["authorization"] = request.headers.get("Authorization")
            return web.Response(
                status=200,
                text='{"status": "ok", "org_id": "o"}',
                content_type="application/json",
            )

        runner_b, base_b = await start_server(handler_b)
        self._runners.append(runner_b)

        async def handler_a(request):
            raise web.HTTPFound(f"{base_b}/auth_status")

        client = await self.client_for(handler_a)
        result = await client.check_auth()
        self.assertEqual(result["status"], "ok")
        self.assertIsNone(captured.get("authorization"))

    # ---- secret hygiene ----

    async def test_api_key_never_appears_in_exception_text(self):
        client = await self.client_for(
            json_handler(401, '{"detail": "bad credentials"}')
        )
        try:
            await client.check_auth()
            self.fail("expected DevinAPIError")
        except DevinAPIError as exc:
            self.assertNotIn(API_KEY, str(exc))
            self.assertNotIn(API_KEY, repr(exc))
            self.assertNotIn("Bearer", str(exc))

    async def test_api_key_not_in_network_error_text(self):
        client = DevinAPIClient(
            API_KEY, base_url="http://127.0.0.1:1", timeout_seconds=5
        )
        try:
            await client.check_auth()
            self.fail("expected DevinAPIError")
        except DevinAPIError as exc:
            self.assertNotIn(API_KEY, str(exc))
            self.assertNotIn(API_KEY, repr(exc))


if __name__ == "__main__":
    unittest.main()
