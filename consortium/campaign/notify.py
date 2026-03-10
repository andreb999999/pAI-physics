"""
Campaign notifications — dispatch status updates to external channels.

Supports:
  - Slack incoming webhooks (SLACK_WEBHOOK_URL)
  - Telegram bot API (TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID)
  - SMS via Twilio (TWILIO_ACCOUNT_SID + TWILIO_AUTH_TOKEN + TWILIO_FROM_NUMBER + SMS_TO_NUMBER)
  - Push notifications via ntfy.sh (NTFY_TOPIC)
  - stdout fallback (always)

All notification failures are silently swallowed — notifications must never
interrupt the campaign.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import requests

from .spec import NotificationConfig


def notify(
    message: str,
    config: NotificationConfig,
    level: str = "info",
) -> None:
    """
    Dispatch a notification message.

    Args:
        message: Plain-text message to send.
        config:  NotificationConfig (from campaign spec).
        level:   "info" | "success" | "warning" | "error" (used for formatting).
    """
    prefix = {
        "info":    "[consortium]",
        "success": "[consortium] ✓",
        "warning": "[consortium] ⚠",
        "error":   "[consortium] ✗",
    }.get(level, "[consortium]")

    full_message = f"{prefix} {message}"
    print(full_message)

    if config.slack_webhook:
        _slack(full_message, config.slack_webhook)

    if config.telegram_bot_token and config.telegram_chat_id:
        _telegram(full_message, config.telegram_bot_token, config.telegram_chat_id)

    if config.ntfy_topic:
        _ntfy(full_message, config.ntfy_topic, config.ntfy_server)

    if config.twilio_account_sid and config.sms_to_number:
        _sms(full_message, config.twilio_account_sid, config.twilio_auth_token,
             config.twilio_from_number, config.sms_to_number)


def notify_stage_complete(stage_id: str, workspace: str, config: NotificationConfig) -> None:
    notify(
        f"Stage '{stage_id}' completed. Workspace: {workspace}",
        config,
        level="success",
    )


def notify_stage_failed(stage_id: str, reason: str, config: NotificationConfig) -> None:
    notify(
        f"Stage '{stage_id}' FAILED: {reason}",
        config,
        level="error",
    )


def notify_stage_launched(stage_id: str, pid: int, workspace: str, config: NotificationConfig) -> None:
    notify(
        f"Stage '{stage_id}' launched (PID {pid}). Workspace: {workspace}",
        config,
        level="info",
    )


def notify_campaign_complete(campaign_name: str, config: NotificationConfig) -> None:
    notify(
        f"Campaign '{campaign_name}' is fully complete.",
        config,
        level="success",
    )


def notify_heartbeat(summary: str, config: NotificationConfig) -> None:
    if config.on_heartbeat:
        notify(summary, config, level="info")


# ------------------------------------------------------------------
# Transport implementations
# ------------------------------------------------------------------

def _slack(message: str, webhook_url: str) -> None:
    try:
        resp = requests.post(
            webhook_url,
            json={"text": message},
            timeout=10,
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"[campaign:notify] Slack delivery failed: {e}")


def _telegram(message: str, bot_token: str, chat_id: str) -> None:
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        resp = requests.post(
            url,
            json={"chat_id": chat_id, "text": message},
            timeout=10,
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"[campaign:notify] Telegram delivery failed: {e}")


def _ntfy(message: str, topic: str, server: Optional[str] = None) -> None:
    try:
        base = (server or "https://ntfy.sh").rstrip("/")
        resp = requests.post(
            f"{base}/{topic}",
            data=message.encode("utf-8"),
            headers={"Title": "OpenClaw Campaign"},
            timeout=10,
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"[campaign:notify] ntfy delivery failed: {e}")


def _sms(message: str, account_sid: str, auth_token: str,
         from_number: str, to_number: str) -> None:
    try:
        resp = requests.post(
            f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json",
            auth=(account_sid, auth_token),
            data={"From": from_number, "To": to_number, "Body": message[:1600]},
            timeout=10,
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"[campaign:notify] SMS delivery failed: {e}")
