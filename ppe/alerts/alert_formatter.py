"""
Alert Formatter
Formats violation information into short, Twilio-safe alert messages
"""

from datetime import datetime
from typing import List
from ppe.alerts import config


class AlertFormatter:
    """
    Formats violation information into alert messages for SMS and phone calls.
    Optimized for Twilio Free / Trial plan (single SMS, short TTS).
    """

    @staticmethod
    def get_alert_level(missing_items: List[str]) -> str:
        """
        Determine alert level based on number of missing items.
        """
        count = len(missing_items)
        if count >= config.ALERT_LEVEL_CRITICAL:
            return "CRITICAL"
        elif count >= config.ALERT_LEVEL_WARNING:
            return "WARNING"
        return "INFO"

    @staticmethod
    def format_missing_items(items: List[str]) -> str:
        """
        Format PPE items into short readable text.
        """
        if not items:
            return "none"

        # Compact naming to reduce SMS length
        short_map = {
            "safety_helmet": "helmet",
            "face_mask": "mask",
            "reflective_vest": "vest",
            "safety_gloves": "gloves",
            "safety_shoes": "shoes"
        }

        formatted = [
            short_map.get(item, item.replace("_", " "))
            for item in items
        ]

        return ", ".join(formatted)

    @staticmethod
    def format_sms_alert(violation_info: dict) -> str:
        """
        Format violation info into a single, Twilio-safe SMS.
        """

        missing = violation_info.get("missing", [])
        timestamp = violation_info.get(
            "timestamp",
            datetime.now().strftime("%H:%M")
        )
        location = violation_info.get(
            "location",
            config.ALERT_LOCATION
        )

        level = AlertFormatter.get_alert_level(missing)
        missing_text = AlertFormatter.format_missing_items(missing)

        # Minimal emoji usage to avoid UTF-16 expansion
        emoji = "🚨 " if level == "CRITICAL" else "⚠️ " if level == "WARNING" else ""

        if level == "CRITICAL":
            return (
                f"{emoji}CRITICAL PPE VIOLATION\n"
                f"Missing: {missing_text}\n"
                f"{location} | {timestamp}\n"
                f"TAKE ACTION NOW"
            )

        return (
            f"{emoji}PPE Violation\n"
            f"Missing: {missing_text}\n"
            f"{location} | {timestamp}"
        )

    @staticmethod
    def format_call_message(violation_info: dict) -> str:
        """
        Format ultra-short TTS message for phone calls.
        """

        missing = violation_info.get("missing", [])
        missing_text = AlertFormatter.format_missing_items(missing)
        level = AlertFormatter.get_alert_level(missing)

        if level == "CRITICAL":
            return (
                f"Critical PPE violation. "
                f"Missing {missing_text}. "
                f"Immediate action required."
            )

        return (
            f"PPE violation detected. "
            f"Missing {missing_text}. "
            f"Please correct immediately."
        )
