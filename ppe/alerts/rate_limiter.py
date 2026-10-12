"""
Rate Limiter for Alert System
Prevents spam and respects Twilio rate limits
"""

from datetime import datetime, timedelta
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Rate limiter for SMS alerts to prevent spam and respect Twilio limits.
    """
    
    def __init__(self, sms_cooldown: int = 60, daily_limit: int = 100):
        """
        Initialize the RateLimiter
        
        Args:
            sms_cooldown: Minimum seconds between SMS for same violation (default: 60)
            daily_limit: Maximum SMS per day (default: 100)
        """
        self.sms_cooldown = sms_cooldown
        self.daily_limit = daily_limit
        self.last_sms_time = {}  # violation_signature -> datetime
        self.daily_sms_count = 0
        self.daily_reset_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    
    def _reset_daily_counter_if_needed(self):
        """Reset daily counter if a new day has started"""
        current_time = datetime.now()
        if current_time >= self.daily_reset_time:
            old_count = self.daily_sms_count
            self.daily_sms_count = 0
            self.daily_reset_time = current_time.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            logger.info(f"Daily SMS counter reset (previous count: {old_count})")
    
    def can_send_sms(self, violation_signature: str) -> tuple:
        """
        Check if SMS can be sent for a violation.
        
        Args:
            violation_signature: Unique signature for the violation
            
        Returns:
            (can_send: bool, reason: str or None)
        """
        self._reset_daily_counter_if_needed()
        
        # Check daily limit
        if self.daily_sms_count >= self.daily_limit:
            remaining_time = self.daily_reset_time - datetime.now()
            hours_remaining = remaining_time.total_seconds() / 3600
            return False, f"Daily SMS limit reached ({self.daily_limit}). Resets in {hours_remaining:.1f} hours."
        
        # Check cooldown for this specific violation
        if violation_signature in self.last_sms_time:
            last_sms_time = self.last_sms_time[violation_signature]
            time_since_last = (datetime.now() - last_sms_time).total_seconds()
            
            if time_since_last < self.sms_cooldown:
                remaining_cooldown = self.sms_cooldown - time_since_last
                return False, f"SMS cooldown active. Wait {remaining_cooldown:.0f} more seconds."
        
        return True, None
    
    def record_sms_sent(self, violation_signature: str):
        """
        Record that an SMS was sent for a violation.
        
        Args:
            violation_signature: Unique signature for the violation
        """
        current_time = datetime.now()
        self.last_sms_time[violation_signature] = current_time
        self.daily_sms_count += 1
        
        logger.debug(
            f"SMS sent recorded for violation: {violation_signature} "
            f"(Daily count: {self.daily_sms_count}/{self.daily_limit})"
        )
    
    def get_stats(self) -> dict:
        """
        Get rate limiter statistics
        
        Returns:
            Dictionary with rate limiter stats
        """
        self._reset_daily_counter_if_needed()
        
        return {
            'daily_sms_count': self.daily_sms_count,
            'daily_limit': self.daily_limit,
            'sms_remaining_today': max(0, self.daily_limit - self.daily_sms_count),
            'cooldown_seconds': self.sms_cooldown,
            'tracked_violations': len(self.last_sms_time),
            'next_reset_time': self.daily_reset_time.isoformat()
        }
