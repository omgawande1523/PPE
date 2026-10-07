"""
Configuration for deduplication module
"""

import os

# Deduplication Configuration
ALERT_COOLDOWN_SECONDS = int(os.getenv('ALERT_COOLDOWN_SECONDS', '30'))  # Default 30 seconds
ENABLE_DEDUPLICATION = os.getenv('ENABLE_DEDUPLICATION', 'true').lower() == 'true'  # Default enabled
MAX_VIOLATION_HISTORY = int(os.getenv('MAX_VIOLATION_HISTORY', '100'))  # Max violations to track

# Logging level for deduplication
DEDUPLICATION_LOG_LEVEL = os.getenv('DEDUPLICATION_LOG_LEVEL', 'INFO')  # DEBUG, INFO, WARNING, ERROR
