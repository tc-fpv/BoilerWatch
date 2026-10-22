"""Manually send a simulated boiler error through the HA webhook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from boiler_reader.webhook import create_from_env


def simulated_error_payload() -> dict[str, object]:
    return {
        "status": "invalid",
        "issue": "error",
        "pressure": None,
        "display": "ECS",
        "error_code": "ECS",
        "error": "Manual webhook smoke test",
        "delay_minutes": 0,
        "simulated": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--send",
        action="store_true",
        help="POST the simulated error and trigger the configured HA automation",
    )
    args = parser.parse_args()

    load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)
    payload = simulated_error_payload()
    if not args.send:
        print("Dry run only. Payload:")
        print(json.dumps(payload, indent=2))
        print("Add --send to trigger the Home Assistant automation.")
        return 0

    notifier = create_from_env()
    if notifier is None:
        parser.error("HA_WEBHOOK_URL is not configured")
    if not notifier.send(payload):
        print("Webhook delivery failed; check the viewer logs and HA webhook automation.")
        return 1
    print("Simulated error webhook delivered successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())