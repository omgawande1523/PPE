"""
Deduplication module for PPE Detection System
Prevents duplicate alerts for the same violation patterns
Phase 2: Enhanced with state machine and pattern-based deduplication
"""

from .violation_deduplicator import ViolationDeduplicator, ViolationState

__all__ = ['ViolationDeduplicator', 'ViolationState']
