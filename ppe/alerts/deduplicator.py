"""
Violation Deduplicator - Prevents duplicate alerts for the same violation patterns
Phase 2: Enhanced with state machine and pattern-based deduplication
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ViolationState(Enum):
    """Violation state enumeration"""
    NEW = "new"           # Just detected, first alert
    ACTIVE = "active"     # Currently active, within cooldown
    EXPIRED = "expired"   # Cooldown expired, can alert again
    RESOLVED = "resolved" # No longer detected in frames


class ViolationDeduplicator:
    """
    Deduplicates violations by tracking active violations with timestamps and states.
    Prevents duplicate alerts for the same violation pattern within a cooldown period.
    Phase 2: Enhanced with state machine for better lifecycle management.
    """
    
    def __init__(self, cooldown_seconds: int = 30, max_active_violations: int = 100):
        """
        Initialize the ViolationDeduplicator
        
        Args:
            cooldown_seconds: Time window (in seconds) to suppress duplicate alerts (default: 30)
            max_active_violations: Maximum number of active violations to track (default: 100)
        """
        self.cooldown_seconds = cooldown_seconds
        self.max_active_violations = max_active_violations
        # Enhanced violation tracking with state machine
        # signature -> {
        #   'timestamp': datetime,      # Last alert time
        #   'first_seen': datetime,     # First detection time
        #   'last_seen': datetime,      # Last detection time (even if suppressed)
        #   'state': ViolationState,    # Current state
        #   'alert_count': int,         # Number of alerts generated
        #   'detection_count': int      # Total detections (including suppressed)
        # }
        self.active_violations: Dict[str, dict] = {}
        self.stats = {
            'total_violations': 0,
            'duplicates_suppressed': 0,
            'new_violations': 0,
            'state_new': 0,
            'state_active': 0,
            'state_expired': 0,
            'state_resolved': 0,
            'total_alerts_generated': 0,
            'total_detections': 0
        }
    
    def create_signature(self, missing_ppe: List[str]) -> str:
        """
        Create a unique signature from missing PPE items.
        Normalizes and sorts items for consistent comparison.
        
        Args:
            missing_ppe: List of missing PPE item names
            
        Returns:
            A normalized, sorted string signature
        """
        if not missing_ppe:
            return ""
        
        # Normalize: lowercase, remove underscores/dashes, strip whitespace
        normalized = [
            item.lower().replace('_', ' ').replace('-', ' ').strip()
            for item in missing_ppe
        ]
        
        # Sort to ensure consistent signature regardless of input order
        normalized.sort()
        
        # Create signature by joining with delimiter
        signature = "|".join(normalized)
        
        return signature
    
    def _update_violation_state(self, signature: str, current_time: datetime):
        """
        Update violation state based on time since last alert.
        
        Args:
            signature: Violation signature
            current_time: Current timestamp
        """
        if signature not in self.active_violations:
            return
        
        violation = self.active_violations[signature]
        time_since_last_alert = (current_time - violation['timestamp']).total_seconds()
        
        # State transitions
        if violation['state'] == ViolationState.RESOLVED:
            # Resolved violations become NEW if detected again
            violation['state'] = ViolationState.NEW
            violation['first_seen'] = current_time
        elif time_since_last_alert >= self.cooldown_seconds:
            if violation['state'] != ViolationState.EXPIRED:
                violation['state'] = ViolationState.EXPIRED
        elif violation['state'] == ViolationState.NEW:
            violation['state'] = ViolationState.ACTIVE
        elif violation['state'] == ViolationState.EXPIRED:
            # Will transition back to NEW when registered again
            pass
    
    def is_duplicate(self, missing_ppe: List[str]) -> bool:
        """
        Check if a violation is a duplicate within the cooldown period.
        Enhanced with state machine awareness.
        
        Args:
            missing_ppe: List of missing PPE item names
            
        Returns:
            True if duplicate (should be suppressed), False if new violation
        """
        if not missing_ppe:
            return False
        
        signature = self.create_signature(missing_ppe)
        current_time = datetime.now()
        
        if signature in self.active_violations:
            violation_data = self.active_violations[signature]
            last_alert_time = violation_data['timestamp']
            
            # Update state before checking
            self._update_violation_state(signature, current_time)
            
            # Update last_seen timestamp
            violation_data['last_seen'] = current_time
            violation_data['detection_count'] += 1
            self.stats['total_detections'] += 1
            
            time_diff = (current_time - last_alert_time).total_seconds()
            
            # Check if within cooldown period
            if time_diff < self.cooldown_seconds and violation_data['state'] != ViolationState.EXPIRED:
                # Duplicate within cooldown period
                self.stats['duplicates_suppressed'] += 1
                logger.debug(
                    f"Duplicate violation suppressed: {signature} "
                    f"(state: {violation_data['state'].value}, "
                    f"last alert {time_diff:.1f}s ago)"
                )
                return True
        
        return False
    
    def register_violation(self, missing_ppe: List[str]) -> bool:
        """
        Register a new violation and update tracking with state machine.
        
        Args:
            missing_ppe: List of missing PPE item names
            
        Returns:
            True if violation was registered (new or expired), False if duplicate
        """
        if not missing_ppe:
            return False
        
        signature = self.create_signature(missing_ppe)
        current_time = datetime.now()
        
        # Check if duplicate before registering
        if self.is_duplicate(missing_ppe):
            # Update timestamp for existing violation (keep it active)
            self.active_violations[signature]['timestamp'] = current_time
            return False
        
        # Register new violation or update expired one
        if signature in self.active_violations:
            violation = self.active_violations[signature]
            old_state = violation['state']
            
            # Update existing violation (expired or resolved)
            violation['timestamp'] = current_time
            violation['last_seen'] = current_time
            violation['detection_count'] += 1
            
            # State transition: EXPIRED -> NEW (re-alert after cooldown)
            if violation['state'] == ViolationState.EXPIRED:
                violation['state'] = ViolationState.NEW
                violation['first_seen'] = current_time  # Reset first_seen for new cycle
                self.stats['state_expired'] -= 1
                self.stats['state_new'] += 1
            elif violation['state'] == ViolationState.RESOLVED:
                violation['state'] = ViolationState.NEW
                violation['first_seen'] = current_time
                self.stats['state_resolved'] -= 1
                self.stats['state_new'] += 1
            
            violation['alert_count'] += 1
            self.stats['total_alerts_generated'] += 1
            
            logger.info(
                f"Violation re-registered: {signature} "
                f"(state: {old_state.value} -> {violation['state'].value}, "
                f"alerts: {violation['alert_count']})"
            )
        else:
            # New violation
            self.active_violations[signature] = {
                'timestamp': current_time,
                'first_seen': current_time,
                'last_seen': current_time,
                'state': ViolationState.NEW,
                'alert_count': 1,
                'detection_count': 1
            }
            self.stats['new_violations'] += 1
            self.stats['state_new'] += 1
            self.stats['total_alerts_generated'] += 1
            self.stats['total_detections'] += 1
            logger.info(f"New violation registered: {signature} (state: NEW)")
        
        self.stats['total_violations'] += 1
        
        # Cleanup if too many active violations
        if len(self.active_violations) > self.max_active_violations:
            self._cleanup_expired()
        
        return True
    
    def update_violation_timestamp(self, missing_ppe: List[str]):
        """
        Update the timestamp of an existing violation without creating a new alert.
        Useful for updating violation state without triggering duplicate suppression.
        
        Args:
            missing_ppe: List of missing PPE item names
        """
        if not missing_ppe:
            return
        
        signature = self.create_signature(missing_ppe)
        current_time = datetime.now()
        
        if signature in self.active_violations:
            violation = self.active_violations[signature]
            violation['last_seen'] = current_time
            violation['detection_count'] += 1
            self.stats['total_detections'] += 1
            # Update state based on time
            self._update_violation_state(signature, current_time)
    
    def mark_violation_resolved(self, missing_ppe: List[str]):
        """
        Mark a violation as resolved (no longer detected).
        
        Args:
            missing_ppe: List of missing PPE item names
        """
        if not missing_ppe:
            return
        
        signature = self.create_signature(missing_ppe)
        if signature in self.active_violations:
            violation = self.active_violations[signature]
            old_state = violation['state']
            
            violation['state'] = ViolationState.RESOLVED
            
            # Update stats
            if old_state == ViolationState.NEW:
                self.stats['state_new'] -= 1
            elif old_state == ViolationState.ACTIVE:
                self.stats['state_active'] -= 1
            elif old_state == ViolationState.EXPIRED:
                self.stats['state_expired'] -= 1
            
            self.stats['state_resolved'] += 1
            
            logger.info(f"Violation marked as resolved: {signature}")
    
    def _cleanup_expired(self):
        """
        Remove expired violations from tracking.
        Expired = older than 2x cooldown period (safe cleanup threshold)
        Enhanced to update states before cleanup.
        """
        current_time = datetime.now()
        cleanup_threshold = self.cooldown_seconds * 2
        
        # Update all violation states first
        for sig in list(self.active_violations.keys()):
            self._update_violation_state(sig, current_time)
        
        # Find violations to remove (RESOLVED or very old EXPIRED)
        expired_signatures = []
        for sig, data in self.active_violations.items():
            time_since_last = (current_time - data['last_seen']).total_seconds()
            
            # Remove if:
            # 1. RESOLVED and old enough (2x cooldown)
            # 2. EXPIRED and very old (4x cooldown) - likely won't be seen again
            if (data['state'] == ViolationState.RESOLVED and time_since_last > cleanup_threshold) or \
               (data['state'] == ViolationState.EXPIRED and time_since_last > cleanup_threshold * 2):
                expired_signatures.append(sig)
        
        # Update stats before removal
        for sig in expired_signatures:
            state = self.active_violations[sig]['state']
            if state == ViolationState.RESOLVED:
                self.stats['state_resolved'] -= 1
            elif state == ViolationState.EXPIRED:
                self.stats['state_expired'] -= 1
        
        # Remove from tracking
        for sig in expired_signatures:
            del self.active_violations[sig]
        
        if expired_signatures:
            logger.debug(f"Cleaned up {len(expired_signatures)} expired violations")
    
    def get_violations_by_state(self, state: ViolationState) -> Dict[str, dict]:
        """
        Get all violations in a specific state.
        
        Args:
            state: The violation state to filter by
            
        Returns:
            Dictionary of violations in the specified state
        """
        current_time = datetime.now()
        
        # Update all states before filtering
        for sig in self.active_violations:
            self._update_violation_state(sig, current_time)
        
        return {
            sig: data.copy()
            for sig, data in self.active_violations.items()
            if data['state'] == state
        }
    
    def get_violation_info(self, missing_ppe: List[str]) -> Optional[dict]:
        """
        Get detailed information about a specific violation.
        
        Args:
            missing_ppe: List of missing PPE item names
            
        Returns:
            Violation information dict or None if not found
        """
        if not missing_ppe:
            return None
        
        signature = self.create_signature(missing_ppe)
        current_time = datetime.now()
        
        if signature in self.active_violations:
            # Update state before returning
            self._update_violation_state(signature, current_time)
            
            violation = self.active_violations[signature].copy()
            violation['state'] = violation['state'].value  # Convert enum to string
            violation['signature'] = signature
            violation['missing_ppe'] = signature.split('|')
            
            # Calculate durations
            violation['duration_active'] = (current_time - violation['first_seen']).total_seconds()
            violation['time_since_last_alert'] = (current_time - violation['timestamp']).total_seconds()
            violation['time_since_last_seen'] = (current_time - violation['last_seen']).total_seconds()
            
            return violation
        
        return None
    
    def clear_all(self):
        """Clear all tracked violations (useful for testing or reset)"""
        count = len(self.active_violations)
        self.active_violations.clear()
        logger.info(f"Cleared {count} tracked violations")
    
    def get_stats(self) -> dict:
        """
        Get deduplication statistics with state information
        
        Returns:
            Dictionary with enhanced statistics including state breakdown
        """
        current_time = datetime.now()
        
        # Update all violation states before generating stats
        for sig in self.active_violations:
            self._update_violation_state(sig, current_time)
        
        # Count states for accuracy
        state_counts = {
            ViolationState.NEW: 0,
            ViolationState.ACTIVE: 0,
            ViolationState.EXPIRED: 0,
            ViolationState.RESOLVED: 0
        }
        
        for violation in self.active_violations.values():
            state_counts[violation['state']] += 1
        
        # Calculate suppression rate
        suppression_rate = 0.0
        if self.stats['total_detections'] > 0:
            suppression_rate = (self.stats['duplicates_suppressed'] / self.stats['total_detections']) * 100
        
        return {
            **self.stats,
            'active_violations_count': len(self.active_violations),
            'cooldown_seconds': self.cooldown_seconds,
            'state_breakdown': {
                'new': state_counts[ViolationState.NEW],
                'active': state_counts[ViolationState.ACTIVE],
                'expired': state_counts[ViolationState.EXPIRED],
                'resolved': state_counts[ViolationState.RESOLVED]
            },
            'suppression_rate_percent': round(suppression_rate, 2),
            'avg_alerts_per_violation': round(
                self.stats['total_alerts_generated'] / max(self.stats['new_violations'], 1),
                2
            )
        }
    
    def get_active_violations(self) -> dict:
        """
        Get all active violations with their timestamps
        
        Returns:
            Dictionary mapping signatures to violation data
        """
        return self.active_violations.copy()
