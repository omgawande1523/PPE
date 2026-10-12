"""Alert and deduplication settings, read from environment variables.

Values come from the process environment or from a .env file in the
repository root (see .env.example). Nothing secret is stored in code.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# Twilio account and phone numbers (E.164, e.g. +14155550100)
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "")
ALERT_RECIPIENT_PHONE = os.getenv("ALERT_RECIPIENT_PHONE", "")

# Alert switches
ENABLE_TWILIO_ALERTS = os.getenv("ENABLE_TWILIO_ALERTS", "false").lower() == "true"
ALERT_SMS_ENABLED = os.getenv("ALERT_SMS_ENABLED", "true").lower() == "true"
ALERT_CALL_ENABLED = os.getenv("ALERT_CALL_ENABLED", "false").lower() == "true"

# Alert frequency control
SMS_COOLDOWN_SECONDS = int(os.getenv("SMS_COOLDOWN_SECONDS", "60"))
DAILY_SMS_LIMIT = int(os.getenv("DAILY_SMS_LIMIT", "100"))

# Critical alert settings
CALL_ONLY_CRITICAL = os.getenv("CALL_ONLY_CRITICAL", "true").lower() == "true"
CRITICAL_VIOLATION_THRESHOLD = int(os.getenv("CRITICAL_VIOLATION_THRESHOLD", "3"))

# Alert level thresholds (number of violated items)
ALERT_LEVEL_INFO = 1
ALERT_LEVEL_WARNING = 2
ALERT_LEVEL_CRITICAL = 3

ALERT_LOCATION = os.getenv("ALERT_LOCATION", "Workplace")

# Deduplication
ALERT_COOLDOWN_SECONDS = int(os.getenv("ALERT_COOLDOWN_SECONDS", "30"))
ENABLE_DEDUPLICATION = os.getenv("ENABLE_DEDUPLICATION", "true").lower() == "true"
MAX_VIOLATION_HISTORY = int(os.getenv("MAX_VIOLATION_HISTORY", "100"))


def validate_config():
    """Return (is_valid, message) for the Twilio settings."""
    if ENABLE_TWILIO_ALERTS:
        if not TWILIO_ACCOUNT_SID:
            return False, "TWILIO_ACCOUNT_SID is not set"
        if not TWILIO_AUTH_TOKEN:
            return False, "TWILIO_AUTH_TOKEN is not set"
        if not TWILIO_PHONE_NUMBER:
            return False, "TWILIO_PHONE_NUMBER is not set"
        if ALERT_SMS_ENABLED and not ALERT_RECIPIENT_PHONE:
            return False, "ALERT_RECIPIENT_PHONE is not set (required for SMS alerts)"
        if not TWILIO_PHONE_NUMBER.startswith("+"):
            return False, "TWILIO_PHONE_NUMBER must be in E.164 format (start with +)"
        if ALERT_RECIPIENT_PHONE and not ALERT_RECIPIENT_PHONE.startswith("+"):
            return False, "ALERT_RECIPIENT_PHONE must be in E.164 format (start with +)"
    return True, "Configuration valid"
