"""
Twilio Configuration for Alert System
Configure Twilio credentials and alert settings via environment variables
"""

import os

# Twilio Account Configuration (set these in your .env file)
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "")
ALERT_RECIPIENT_PHONE = os.getenv("ALERT_RECIPIENT_PHONE", "")
# Twilio Account Configuration


# Alert Recipient

# Alert Settings
ENABLE_TWILIO_ALERTS = os.getenv('ENABLE_TWILIO_ALERTS', 'false').lower() == 'true'
ALERT_SMS_ENABLED = os.getenv('ALERT_SMS_ENABLED', 'true').lower() == 'true'
ALERT_CALL_ENABLED = os.getenv('ALERT_CALL_ENABLED', 'false').lower() == 'true'

# Alert Frequency Control
SMS_COOLDOWN_SECONDS = int(os.getenv('SMS_COOLDOWN_SECONDS', '60'))  # Don't send SMS more than once per minute per violation
DAILY_SMS_LIMIT = int(os.getenv('DAILY_SMS_LIMIT', '100'))  # Maximum SMS per day

# Critical Alert Settings
CALL_ONLY_CRITICAL = os.getenv('CALL_ONLY_CRITICAL', 'true').lower() == 'true'
CRITICAL_VIOLATION_THRESHOLD = int(os.getenv('CRITICAL_VIOLATION_THRESHOLD', '3'))  # Missing 3+ items = critical

# Alert Level Thresholds
ALERT_LEVEL_INFO = 1
ALERT_LEVEL_WARNING = 2
ALERT_LEVEL_CRITICAL = 3

# Location/Zone (optional - can be customized)
ALERT_LOCATION = os.getenv('ALERT_LOCATION', 'Workplace')

def validate_config():
    """
    Validate Twilio configuration.
    Returns (is_valid, error_message)
    """
    if ENABLE_TWILIO_ALERTS:
        if not TWILIO_ACCOUNT_SID:
            return False, "TWILIO_ACCOUNT_SID is not set"
        if not TWILIO_AUTH_TOKEN:
            return False, "TWILIO_AUTH_TOKEN is not set"
        if not TWILIO_PHONE_NUMBER:
            return False, "TWILIO_PHONE_NUMBER is not set"
        if ALERT_SMS_ENABLED and not ALERT_RECIPIENT_PHONE:
            return False, "ALERT_RECIPIENT_PHONE is not set (required for SMS alerts)"
        if not TWILIO_PHONE_NUMBER.startswith('+'):
            return False, "TWILIO_PHONE_NUMBER must be in E.164 format (start with +)"
        if ALERT_RECIPIENT_PHONE and not ALERT_RECIPIENT_PHONE.startswith('+'):
            return False, "ALERT_RECIPIENT_PHONE must be in E.164 format (start with +)"
    
    return True, "Configuration valid"
