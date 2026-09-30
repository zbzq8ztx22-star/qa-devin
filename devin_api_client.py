import asyncio
import json
import math
import os
from typing import Literal, TypedDict

import aiohttp
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class DevinAPIError(Exception):
    """Unified error for Devin API failures.

    Covers HTTP error statuses, network/connectivity errors, timeouts,
    non-JSON responses, and 2xx payloads missing contract keys.
    `status` is the HTTP status code when one was received (None for
    network errors/timeouts). `body` is the raw response body when one
    was received. Never contains request headers or credentials.
    """

    def __init__(self, message, status=None, body=None):
        super().__init__(message)
        self.status = status
        self.body = body


class DevinAPIAuthResponse(TypedDict):
    status: str
    org_id: str


class DevinAPISessionResponse(TypedDict):
    session_id: str
    url: str
    is_new_session: bool | None


class DevinAPISessionStatusResponse(TypedDict):
    session_id: str
    status: str
    title: str
    created_at: str
    updated_at: str
    snapshot_id: str | None
    playbook_id: str | None
    structured_output: dict
    status_enum: (
        Literal[
            "working",
            "blocked",
            "finished",
            "suspend_requested",
            "resume_requested",
            "resumed",
        ]
        | None
    )


DEFAULT_TIMEOUT_SECONDS = 60


class DevinAPIClient:
    BASE_URL = "https://api.devin.ai/v1"

    def __init__(
        self,
        api_key: str,
        base_url: str = BASE_URL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ):
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("api_key must be a non-empty string")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError(
                f"timeout_seconds must be a positive finite number, "
                f"got {timeout_seconds!r}"
            )
        self.api_key = api_key
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self._timeout = aiohttp.ClientTimeout(total=float(timeout_seconds))
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    async def _request(self, method: str, path: str, **kwargs) -> dict:
        try:
            async with (
                aiohttp.ClientSession(timeout=self._timeout) as session,
                session.request(
                    method, f"{self.base_url}{path}", headers=self.headers, **kwargs
                ) as response,
            ):
                raw = await response.read()
                text = self._sanitize(
                    raw.decode(response.get_encoding(), errors="replace")
                )
                if response.status >= 400:
                    raise DevinAPIError(
                        f"Devin API {method} {path} failed with HTTP "
                        f"{response.status}: {text[:300]}",
                        status=response.status,
                        body=text,
                    )
                try:
                    data = json.loads(text)
                except json.JSONDecodeError:
                    raise DevinAPIError(
                        f"Devin API {method} {path} returned a non-JSON "
                        f"response (HTTP {response.status}): {text[:300]}",
                        status=response.status,
                        body=text,
                    )
                if not isinstance(data, dict):
                    raise DevinAPIError(
                        f"Devin API {method} {path} returned an unexpected "
                        f"payload type: {text[:300]}",
                        status=response.status,
                        body=text,
                    )
                return data
        except DevinAPIError:
            raise
        except asyncio.TimeoutError as exc:
            raise DevinAPIError(
                f"Devin API {method} {path} timed out after "
                f"{self.timeout_seconds} seconds"
            ) from exc
        except aiohttp.ClientError as exc:
            raise DevinAPIError(
                f"Devin API {method} {path} failed: {exc}"
            ) from exc

    def _sanitize(self, text: str) -> str:
        return text.replace(f"Bearer {self.api_key}", "[REDACTED]").replace(
            self.api_key, "[REDACTED]"
        )

    def _require_keys(self, data: dict, keys: tuple[str, ...], label: str) -> None:
        missing = [key for key in keys if key not in data]
        if missing:
            raise DevinAPIError(
                f"Devin API {label} returned HTTP 200 with a payload missing "
                f"required keys {missing}",
                body=self._sanitize(repr(data)),
            )

    async def check_auth(self) -> DevinAPIAuthResponse:
        data = await self._request("GET", "/auth_status")
        self._require_keys(data, ("status", "org_id"), "GET /auth_status")
        return data

    async def start_session(self, prompt: str) -> DevinAPISessionResponse:
        data = await self._request("POST", "/sessions", json={"prompt": prompt})
        self._require_keys(data, ("session_id", "url"), "POST /sessions")
        return data

    async def get_session_status(
        self, session_id: str
    ) -> DevinAPISessionStatusResponse | None:
        try:
            data = await self._request("GET", f"/session/{session_id}")
        except DevinAPIError as exc:
            try:
                body = json.loads(exc.body) if exc.status == 404 else None
            except (TypeError, json.JSONDecodeError):
                body = None
            if isinstance(body, dict) and body.get("detail") == "Session not found":
                return None
            raise
        if data.get("detail") == "Session not found":
            return None
        self._require_keys(
            data,
            ("status_enum", "structured_output"),
            f"GET /session/{session_id}",
        )
        return data


async def main():
    api_key = os.getenv("DEVIN_API_KEY")
    if not api_key:
        raise ValueError("DEVIN_API_KEY environment variable is required")
    client = DevinAPIClient(api_key)
    auth_status = await client.check_auth()
    print("AUTH STATUS: ", auth_status)


if __name__ == "__main__":
    asyncio.run(main())
