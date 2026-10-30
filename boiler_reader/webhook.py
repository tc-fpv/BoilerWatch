"""Send event payloads to a Home Assistant webhook."""

from __future__ import annotations

import json
import logging
import os
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PostJson = Callable[[str, dict[str, object], int], int]


def _post_json(url: str, payload: dict[str, object], timeout: int) -> int:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return response.status


class HomeAssistantWebhookNotifier:
    def __init__(
        self,
        webhook_url: str,
        timeout_seconds: int = 5,
        post_json: PostJson = _post_json,
        logger: Callable[..., None] | None = None,
    ) -> None:
        self.webhook_url = webhook_url
        self.timeout_seconds = timeout_seconds
        self.post_json = post_json
        self.logger = logger or self._log

    @staticmethod
    def _log(event: str, **fields: object) -> None:
        logging.getLogger("boiler_reader.webhook").info(
            json.dumps({"event": event, **fields}, separators=(",", ":"))
        )

    def send(self, payload: dict[str, object]) -> bool:
        try:
            status_code = self.post_json(
                self.webhook_url,
                payload,
                self.timeout_seconds,
            )
            if 200 <= status_code < 300:
                self.logger("ha_webhook_delivered", status_code=status_code)
                return True
            self.logger("ha_webhook_rejected", status_code=status_code)
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as error:
            self.logger("ha_webhook_failed", error_type=type(error).__name__)
        return False


def create_from_env(
    logger: Callable[..., None] | None = None,
) -> HomeAssistantWebhookNotifier | None:
    webhook_url = os.getenv("HA_WEBHOOK_URL", "").strip()
    if not webhook_url:
        return None
    try:
        timeout_seconds = int(os.getenv("HA_WEBHOOK_TIMEOUT", "5"))
    except ValueError:
        raise ValueError("HA_WEBHOOK_TIMEOUT must be a whole number of seconds") from None
    if not 1 <= timeout_seconds <= 60:
        raise ValueError("HA_WEBHOOK_TIMEOUT must be between 1 and 60 seconds")
    return HomeAssistantWebhookNotifier(
        webhook_url,
        timeout_seconds=timeout_seconds,
        logger=logger,
    )