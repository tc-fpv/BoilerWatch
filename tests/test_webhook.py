from urllib.error import URLError

from boiler_reader.webhook import HomeAssistantWebhookNotifier, create_from_env


def test_posts_json_payload_to_configured_ha_webhook() -> None:
    requests = []

    def post_json(url, payload, timeout):
        requests.append((url, payload, timeout))
        return 200

    notifier = HomeAssistantWebhookNotifier(
        "http://ha.local/api/webhook/private-id",
        post_json=post_json,
    )

    assert notifier.send({"status": "invalid", "error_code": "ECS"})
    assert requests == [
        (
            "http://ha.local/api/webhook/private-id",
            {"status": "invalid", "error_code": "ECS"},
            5,
        )
    ]


def test_failed_webhook_returns_false_and_does_not_log_url() -> None:
    log_events = []

    def fail_post(url, payload, timeout):
        raise URLError("connection refused")

    notifier = HomeAssistantWebhookNotifier(
        "http://ha.local/api/webhook/private-id",
        post_json=fail_post,
        logger=lambda event, **fields: log_events.append((event, fields)),
    )

    assert not notifier.send({"status": "low"})
    assert "private-id" not in repr(log_events)


def test_webhook_notifier_is_disabled_without_url(monkeypatch) -> None:
    monkeypatch.delenv("HA_WEBHOOK_URL", raising=False)

    assert create_from_env() is None


def test_create_from_env_accepts_plain_http_scheme(monkeypatch) -> None:
    monkeypatch.setenv(
        "HA_WEBHOOK_URL",
        "http://ha.local/api/webhook/test-id",
    )

    notifier = create_from_env()

    assert notifier.webhook_url == "http://ha.local/api/webhook/test-id"