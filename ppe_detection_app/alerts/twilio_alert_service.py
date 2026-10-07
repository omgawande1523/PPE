"""
Twilio Alert Service
Handles sending SMS and phone call alerts via Twilio
"""

import logging
from typing import Optional, Dict
from datetime import datetime

try:
    from twilio.rest import Client
    from twilio.base.exceptions import TwilioRestException
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    logging.warning("Twilio SDK not installed. Install with: pip install twilio")

import twilio_config as config
from .rate_limiter import RateLimiter
from .alert_formatter import AlertFormatter

logger = logging.getLogger(__name__)


class TwilioAlertService:
    """
    Service for sending alerts via Twilio SMS and phone calls.
    Integrates with deduplication system to prevent alert spam.
    """
    
    def __init__(self):
        """Initialize Twilio Alert Service"""
        if not TWILIO_AVAILABLE:
            raise ImportError("Twilio SDK not installed. Install with: pip install twilio")
        
        # Validate configuration
        is_valid, error_msg = config.validate_config()
        if not is_valid:
            raise ValueError(f"Invalid Twilio configuration: {error_msg}")
        
        # Initialize Twilio client
        try:
            self.client = Client(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)
            logger.info("Twilio client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Twilio client: {e}")
            raise
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(
            sms_cooldown=config.SMS_COOLDOWN_SECONDS,
            daily_limit=config.DAILY_SMS_LIMIT
        )
        
        # Initialize formatter
        self.formatter = AlertFormatter()
        
        # Track last sent alerts for logging
        self.sent_alerts_count = 0
        self.failed_alerts_count = 0
    
    def _create_violation_signature(self, missing_items: list) -> str:
        """
        Create a signature for the violation (same format as deduplicator).
        
        Args:
            missing_items: List of missing PPE items
            
        Returns:
            Normalized signature string
        """
        if not missing_items:
            return ""
        
        # Normalize and sort (same logic as deduplicator)
        normalized = [
            item.lower().replace('_', ' ').replace('-', ' ').strip()
            for item in missing_items
        ]
        normalized.sort()
        return "|".join(normalized)
    
    def send_violation_alert(self, violation_info: dict, retry_count: int = 0) -> Dict[str, any]:
        """
        Send SMS alert for a violation.
        Respects rate limiting and deduplication (should only be called for new violations).
        
        Args:
            violation_info: Dictionary with violation details:
                - missing: List of missing PPE items
                - timestamp: Violation timestamp
                - detected: List of detected items (optional)
                - person_count: Number of people detected (optional)
                - location: Location/zone (optional)
            retry_count: Internal retry counter
            
        Returns:
            Dictionary with result:
                - success: bool
                - message_sid: str (if successful)
                - error: str (if failed)
                - skipped: bool (if rate limited)
        """
        if not config.ENABLE_TWILIO_ALERTS:
            return {'success': False, 'skipped': True, 'reason': 'Twilio alerts disabled'}
        
        if not config.ALERT_SMS_ENABLED:
            return {'success': False, 'skipped': True, 'reason': 'SMS alerts disabled'}
        
        missing_items = violation_info.get('missing', [])
        if not missing_items:
            return {'success': False, 'skipped': True, 'reason': 'No missing items'}
        
        # Create signature for rate limiting
        signature = self._create_violation_signature(missing_items)
        
        # Check rate limiting
        can_send, reason = self.rate_limiter.can_send_sms(signature)
        if not can_send:
            logger.debug(f"SMS alert skipped for violation {signature}: {reason}")
            return {'success': False, 'skipped': True, 'reason': reason}
        
        # Format message
        try:
            message_body = self.formatter.format_sms_alert(violation_info)
        except Exception as e:
            logger.error(f"Failed to format alert message: {e}")
            return {'success': False, 'error': f'Format error: {str(e)}'}
        
        # Send SMS
        try:
            message = self.client.messages.create(
                body=message_body,
                from_=config.TWILIO_PHONE_NUMBER,
                to=config.ALERT_RECIPIENT_PHONE
            )
            
            # Record successful send
            self.rate_limiter.record_sms_sent(signature)
            self.sent_alerts_count += 1
            
            # Log detailed information
            logger.info(
                f"SMS alert sent successfully. "
                f"Violation: {signature}, "
                f"Message SID: {message.sid}, "
                f"Status: {message.status}, "
                f"To: {message.to}, "
                f"From: {message.from_}, "
                f"Total sent: {self.sent_alerts_count}"
            )
            
            # Check message status - might be queued or have issues
            status_warnings = []
            if message.status in ['queued', 'sending']:
                status_warnings.append(f"Message is {message.status} (may take a moment)")
            elif message.status in ['failed', 'undelivered']:
                status_warnings.append(f"WARNING: Message status is {message.status}")
                if hasattr(message, 'error_message') and message.error_message:
                    status_warnings.append(f"Error: {message.error_message}")
            
            return {
                'success': True,
                'message_sid': message.sid,
                'message_status': message.status,
                'violation_signature': signature,
                'warnings': status_warnings if status_warnings else None
            }
        
        except TwilioRestException as e:
            self.failed_alerts_count += 1
            error_code = e.code
            error_msg = e.msg
            
            # Handle specific errors
            if error_code == 21614:  # Invalid phone number
                logger.error(f"Invalid phone number: {error_msg}")
                return {'success': False, 'error': 'invalid_phone', 'details': error_msg}
            
            elif error_code == 21608:  # Rate limit exceeded
                logger.warning(f"Twilio rate limit exceeded: {error_msg}")
                # Still record in rate limiter to prevent more attempts
                self.rate_limiter.record_sms_sent(signature)
                return {'success': False, 'error': 'rate_limit', 'details': error_msg}
            
            elif error_code == 21610:  # Unsubscribed recipient
                logger.error(f"Recipient unsubscribed: {error_msg}")
                return {'success': False, 'error': 'unsubscribed', 'details': error_msg}
            
            elif retry_count < 2:  # Retry up to 2 times
                import time
                wait_time = 2 ** retry_count  # Exponential backoff
                logger.warning(f"Twilio API error, retrying in {wait_time}s: {error_msg}")
                time.sleep(wait_time)
                return self.send_violation_alert(violation_info, retry_count + 1)
            
            else:
                logger.error(f"Failed to send SMS alert after retries: {error_msg} (code: {error_code})")
                return {'success': False, 'error': f'twilio_error_{error_code}', 'details': error_msg}
        
        except Exception as e:
            self.failed_alerts_count += 1
            logger.error(f"Unexpected error sending SMS alert: {e}")
            return {'success': False, 'error': 'unexpected_error', 'details': str(e)}
    
    def get_stats(self) -> dict:
        """
        Get alert service statistics
        
        Returns:
            Dictionary with statistics
        """
        rate_limiter_stats = self.rate_limiter.get_stats()
        
        return {
            'enabled': config.ENABLE_TWILIO_ALERTS,
            'sms_enabled': config.ALERT_SMS_ENABLED,
            'call_enabled': config.ALERT_CALL_ENABLED,
            'total_sent': self.sent_alerts_count,
            'total_failed': self.failed_alerts_count,
            'rate_limiter': rate_limiter_stats
        }
    
    def test_connection(self) -> dict:
        """
        Test Twilio connection and configuration.
        
        Returns:
            Dictionary with test result
        """
        try:
            # Try to fetch account info
            account = self.client.api.accounts(config.TWILIO_ACCOUNT_SID).fetch()
            
            return {
                'success': True,
                'account_sid': account.sid,
                'account_name': account.friendly_name,
                'status': account.status
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
