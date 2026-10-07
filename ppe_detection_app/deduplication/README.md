# Deduplication Module

This module prevents duplicate alerts for the same PPE violation patterns within a configurable time window.

## Features

- **Time-based Alert Suppression**: Prevents duplicate alerts within a cooldown period (default: 30 seconds)
- **Pattern Recognition**: Creates unique signatures from missing PPE items for accurate duplicate detection
- **Automatic Cleanup**: Removes expired violations from memory automatically
- **Statistics Tracking**: Monitors total violations, duplicates suppressed, and active violations

## Configuration

Configure via environment variables or edit `config.py`:

```bash
# Set cooldown period in seconds (default: 30)
export ALERT_COOLDOWN_SECONDS=30

# Enable/disable deduplication (default: true)
export ENABLE_DEDUPLICATION=true

# Maximum violations to track (default: 100)
export MAX_VIOLATION_HISTORY=100
```

Or edit `ppe_detection_app/deduplication/config.py` directly.

## Usage

The deduplicator is automatically initialized and used in `app.py`. No manual integration needed.

### API Endpoints

- **`GET /deduplication_stats`**: Get deduplication statistics
  ```json
  {
    "enabled": true,
    "config": {
      "cooldown_seconds": 30,
      "max_violations": 100
    },
    "stats": {
      "total_violations": 50,
      "duplicates_suppressed": 150,
      "new_violations": 50,
      "active_violations_count": 5
    },
    "active_violations": 5
  }
  ```

- **`GET /detection_status`**: Includes deduplication stats in response

## How It Works

1. **Violation Detection**: When a violation is detected (missing PPE items), a signature is created
2. **Duplicate Check**: The signature is checked against active violations
3. **Alert Suppression**: If the same violation was reported within the cooldown period, the alert is suppressed
4. **Registration**: New violations are registered with a timestamp
5. **Expiration**: Violations older than 2x cooldown period are automatically cleaned up

## Example

```
Time 0s:  Missing helmet detected → Alert sent ✅
Time 5s:  Missing helmet detected → Duplicate suppressed ❌
Time 10s: Missing helmet detected → Duplicate suppressed ❌
Time 35s: Missing helmet detected → Alert sent ✅ (cooldown expired)
```

## Testing

To test deduplication:

1. Start the application
2. Trigger a violation (e.g., remove helmet)
3. Check `/deduplication_stats` endpoint
4. Verify `duplicates_suppressed` counter increases
5. Wait for cooldown period to expire
6. Trigger same violation again - should create new alert

## Performance

- **Memory**: ~100-200 bytes per active violation
- **CPU**: O(1) duplicate check (dictionary lookup)
- **Cleanup**: Automatic, runs when needed

## Troubleshooting

### Deduplication not working
- Check if `ENABLE_DEDUPLICATION=true` in config
- Verify cooldown period is reasonable (not too short)
- Check logs for deduplication messages

### Too many duplicates still showing
- Increase `ALERT_COOLDOWN_SECONDS` value
- Check if violations have consistent signatures (same missing items)

### Memory concerns
- Reduce `MAX_VIOLATION_HISTORY` value
- Automatic cleanup should handle most cases
