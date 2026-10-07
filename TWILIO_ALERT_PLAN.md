# Twilio Alert Integration Plan

## Overview
Integrate Twilio SMS/Phone call functionality to send real-time alerts to your phone when PPE violations are detected. The system will respect the deduplication logic to prevent alert spam.

---

## Objectives

1. **Send SMS alerts** when new violations are detected (respecting deduplication)
2. **Optional phone calls** for critical violations
3. **Configurable alert frequency** to prevent notification fatigue
4. **Alert formatting** with violation details and timestamps
5. **Error handling** and retry logic for failed deliveries
6. **Rate limiting** to comply with Twilio limits

---

## Phase 1: Basic SMS Alert Integration

### 1.1 Setup & Configuration

#### Requirements:
- Twilio account (sign up at https://www.twilio.com)
- Twilio phone number (can get trial number)
- Phone number to receive alerts (your phone)

#### Dependencies:
```bash
pip install twilio
```

#### Configuration File (`ppe_detection_app/twilio_config.py`):
```python
import os

# Twilio Account Configuration
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID', '')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN', '')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER', '')  # Your Twilio number

# Alert Recipient
ALERT_RECIPIENT_PHONE = os.getenv('ALERT_RECIPIENT_PHONE', '')  # Your phone number (E.164 format)

# Alert Settings
ENABLE_TWILIO_ALERTS = os.getenv('ENABLE_TWILIO_ALERTS', 'true').lower() == 'true'
ALERT_SMS_ENABLED = os.getenv('ALERT_SMS_ENABLED', 'true').lower() == 'true'
ALERT_CALL_ENABLED = os.getenv('ALERT_CALL_ENABLED', 'false').lower() == 'true'

# Alert Frequency Control
SMS_COOLDOWN_SECONDS = int(os.getenv('SMS_COOLDOWN_SECONDS', '60'))  # Don't send SMS more than once per minute
CALL_ONLY_CRITICAL = os.getenv('CALL_ONLY_CRITICAL', 'true').lower() == 'true'
CRITICAL_VIOLATION_THRESHOLD = int(os.getenv('CRITICAL_VIOLATION_THRESHOLD', '3'))  # Missing 3+ items = critical
```

### 1.2 Alert Service Module

#### File: `ppe_detection_app/alerts/twilio_alert_service.py`

**Features:**
- Send SMS alerts with violation details
- Optional phone calls for critical violations
- Rate limiting and cooldown management
- Error handling and retry logic
- Integration with deduplication system

**Core Methods:**
```python
class TwilioAlertService:
    def __init__(self):
        # Initialize Twilio client
        # Load configuration
        # Initialize rate limiter
        
    def send_violation_alert(self, violation_info):
        # Send SMS alert for new violation
        # Respect deduplication (only send for new violations)
        
    def send_critical_alert(self, violation_info):
        # Send phone call for critical violations
        
    def format_alert_message(self, violation_info):
        # Format alert message with violation details
        
    def can_send_alert(self, violation_signature):
        # Check if alert can be sent (rate limiting)
```

### 1.3 Integration Points

#### Integration with app.py:
1. **Initialize Twilio service** at module level (like deduplicator)
2. **Send alert in gen_frames()** when new violation is registered
3. **Respect deduplication**: Only send when `register_violation()` returns True

#### Integration Flow:
```
1. Violation detected → deduplicator.is_duplicate() → False
2. deduplicator.register_violation() → True (new violation)
3. twilio_service.send_violation_alert() → Send SMS
4. Duplicate detected → deduplicator.is_duplicate() → True
5. No alert sent (deduplication prevents spam)
```

---

## Phase 2: Enhanced Alert Features

### 2.1 Alert Message Formatting

#### SMS Format Examples:

**Basic Alert:**
```
🚨 PPE VIOLATION DETECTED 🚨

Missing: Helmet, Safety Vest
Time: 2025-01-15 14:30:25
Location: Camera 1

Action Required: Ensure worker wears all required PPE.
```

**Critical Alert (Multiple Items Missing):**
```
🚨 CRITICAL PPE VIOLATION 🚨

Missing: Helmet, Safety Vest, Gloves, Boots
Time: 2025-01-15 14:30:25
Location: Camera 1
Severity: CRITICAL

IMMEDIATE ACTION REQUIRED!
```

**Resolved Alert (Optional):**
```
✅ PPE VIOLATION RESOLVED ✅

Previous Missing: Helmet, Safety Vest
Resolved At: 2025-01-15 14:35:10
Duration: 4 minutes 45 seconds
```

### 2.2 Alert Prioritization

#### Alert Levels:
1. **INFO**: Single missing item (e.g., just gloves)
2. **WARNING**: 2 missing items
3. **CRITICAL**: 3+ missing items → Phone call option
4. **RESOLVED**: Violation cleared (optional notification)

#### Configuration:
```python
ALERT_LEVEL_INFO = 1
ALERT_LEVEL_WARNING = 2
ALERT_LEVEL_CRITICAL = 3

def get_alert_level(missing_items):
    count = len(missing_items)
    if count >= ALERT_LEVEL_CRITICAL:
        return "CRITICAL"
    elif count >= ALERT_LEVEL_WARNING:
        return "WARNING"
    else:
        return "INFO"
```

### 2.3 Rate Limiting

#### SMS Rate Limiting:
- **Cooldown per violation**: Don't send SMS for same violation within X seconds
- **Daily limit**: Maximum SMS per day (Twilio free tier: limited)
- **Per-violation tracking**: Track last SMS time per violation signature

#### Implementation:
```python
class RateLimiter:
    def __init__(self, sms_cooldown=60, daily_limit=100):
        self.sms_cooldown = sms_cooldown
        self.daily_limit = daily_limit
        self.last_sms_time = {}  # signature -> datetime
        self.daily_sms_count = 0
        self.daily_reset_time = None
        
    def can_send_sms(self, violation_signature):
        # Check cooldown
        # Check daily limit
        # Return True if can send
        
    def record_sms_sent(self, violation_signature):
        # Record SMS sent time
        # Increment daily counter
```

---

## Phase 3: Advanced Features

### 3.1 Phone Call Alerts

#### Use Cases:
- Critical violations (3+ missing items)
- Escalation after multiple SMS alerts
- Confirmation required alerts

#### Implementation:
```python
def send_phone_call_alert(self, violation_info):
    # Use Twilio Voice API
    # Call recipient
    # Play recorded message or use TTS
    # Log call status
```

#### TwiML for Voice:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="alice" language="en-US">
        Alert: Critical PPE violation detected. Missing items: Helmet, Safety Vest, and Gloves.
        Please check the workplace immediately.
    </Say>
</Response>
```

### 3.2 Alert Templates & Customization

#### Template System:
- Configurable message templates
- Customizable emojis/icons
- Localization support
- Include/exclude specific details

#### Template Example:
```python
ALERT_TEMPLATE = """
🚨 {severity} PPE VIOLATION 🚨

Missing Items: {missing_items}
Time: {timestamp}
Location: {location}

{action_message}
"""

def format_alert(violation_info, template=ALERT_TEMPLATE):
    return template.format(
        severity=violation_info['severity'],
        missing_items=', '.join(violation_info['missing']),
        timestamp=violation_info['timestamp'],
        location=violation_info.get('location', 'Unknown'),
        action_message=get_action_message(violation_info['severity'])
    )
```

### 3.3 Multi-Recipient Support

#### Features:
- Multiple phone numbers
- Role-based routing (manager, supervisor, safety officer)
- Escalation chains

#### Configuration:
```python
ALERT_RECIPIENTS = {
    'primary': '+1234567890',  # Your phone
    'manager': '+1234567891',
    'safety_officer': '+1234567892'
}

ALERT_ROUTING = {
    'INFO': ['primary'],
    'WARNING': ['primary', 'manager'],
    'CRITICAL': ['primary', 'manager', 'safety_officer']
}
```

### 3.4 Alert History & Logging

#### Features:
- Log all sent alerts (database or file)
- Track delivery status
- Alert analytics dashboard
- Resend failed alerts

#### Database Schema (Optional):
```sql
CREATE TABLE alert_history (
    id INTEGER PRIMARY KEY,
    violation_signature TEXT,
    alert_type TEXT,  -- 'SMS' or 'CALL'
    recipient TEXT,
    message TEXT,
    status TEXT,  -- 'sent', 'failed', 'delivered'
    sent_at TIMESTAMP,
    delivered_at TIMESTAMP,
    error_message TEXT
);
```

---

## Implementation Details

### File Structure
```
ppe_detection_app/
├── alerts/
│   ├── __init__.py
│   ├── twilio_alert_service.py
│   ├── rate_limiter.py
│   ├── alert_formatter.py
│   └── templates.py
├── twilio_config.py (or integrate into existing config)
└── app.py (modified)
```

### Integration with Existing Code

#### Modified app.py:
```python
from alerts import TwilioAlertService
from twilio_config import ENABLE_TWILIO_ALERTS

# Initialize Twilio service
if ENABLE_TWILIO_ALERTS:
    twilio_service = TwilioAlertService()
    print("Twilio alerts enabled")
else:
    twilio_service = None
    print("Twilio alerts disabled")

# In gen_frames():
if not is_duplicate:
    violation_entry = {...}
    
    # Register violation
    if deduplicator:
        deduplicator.register_violation(missing_ppe)
    
    # Send Twilio alert (only for new violations)
    if twilio_service:
        try:
            twilio_service.send_violation_alert({
                'missing': missing_ppe,
                'timestamp': violation_entry['timestamp'],
                'detected': list(detected_classes),
                'person_count': violation_entry['person_count']
            })
        except Exception as e:
            logger.error(f"Failed to send Twilio alert: {e}")
```

---

## Configuration & Setup

### Environment Variables

Create `.env` file or set environment variables:

```bash
# Twilio Credentials
TWILIO_ACCOUNT_SID=your_account_sid_here
TWILIO_AUTH_TOKEN=your_auth_token_here
TWILIO_PHONE_NUMBER=+1234567890  # Your Twilio number

# Alert Recipient (Your phone number in E.164 format)
ALERT_RECIPIENT_PHONE=+1234567890  # +1 for US, include country code

# Alert Settings
ENABLE_TWILIO_ALERTS=true
ALERT_SMS_ENABLED=true
ALERT_CALL_ENABLED=false
SMS_COOLDOWN_SECONDS=60
CALL_ONLY_CRITICAL=true
CRITICAL_VIOLATION_THRESHOLD=3
```

### Phone Number Format

**Important**: Use E.164 format for phone numbers
- US: `+1234567890` (country code + number)
- Include `+` prefix
- No spaces or dashes

### Getting Twilio Credentials

1. Sign up at https://www.twilio.com
2. Get Account SID and Auth Token from dashboard
3. Get a phone number (free trial available)
4. Verify your recipient phone number (trial accounts)

---

## Testing Strategy

### 1. Unit Tests
- Test alert formatting
- Test rate limiting logic
- Test error handling

### 2. Integration Tests
- Test SMS sending with test credentials
- Test phone call functionality
- Test deduplication integration

### 3. Manual Testing
- Trigger violations and verify SMS received
- Test rate limiting (rapid violations)
- Test critical alerts (phone calls)

### Test Script:
```python
# test_twilio_alerts.py
from alerts import TwilioAlertService

service = TwilioAlertService()

# Test basic alert
service.send_violation_alert({
    'missing': ['helmet', 'vest'],
    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'detected': [],
    'person_count': 1
})
```

---

## Error Handling

### Common Errors:
1. **Invalid phone number**: Validate E.164 format
2. **Twilio API errors**: Retry with exponential backoff
3. **Rate limit exceeded**: Queue alerts or skip
4. **Network errors**: Log and continue processing

### Error Handling Strategy:
```python
def send_violation_alert(self, violation_info, retry_count=0):
    try:
        # Send alert
        message = self.client.messages.create(...)
        return {'success': True, 'message_sid': message.sid}
    except TwilioRestException as e:
        if e.code == 21614:  # Invalid phone number
            logger.error(f"Invalid phone number: {e}")
            return {'success': False, 'error': 'invalid_phone'}
        elif e.code == 21608:  # Rate limit
            logger.warning(f"Rate limit exceeded, queuing alert")
            return {'success': False, 'error': 'rate_limit'}
        elif retry_count < 3:
            time.sleep(2 ** retry_count)  # Exponential backoff
            return self.send_violation_alert(violation_info, retry_count + 1)
        else:
            logger.error(f"Failed to send alert after retries: {e}")
            return {'success': False, 'error': str(e)}
```

---

## Cost Considerations

### Twilio Pricing (Approximate):
- **SMS (US)**: ~$0.0075 per message
- **Voice Call (US)**: ~$0.013 per minute
- **Free Trial**: $15.50 credit (enough for testing)

### Cost Optimization:
1. Use SMS for most alerts (cheaper)
2. Use calls only for critical violations
3. Respect rate limiting to prevent spam
4. Use deduplication to reduce alert count

### Estimated Costs:
- **100 alerts/day**: ~$22.50/month (SMS only)
- **10 critical calls/day**: ~$4/month (calls)
- **Total**: ~$26.50/month for active monitoring

---

## Security Best Practices

1. **Never commit credentials**: Use environment variables
2. **Validate phone numbers**: Prevent abuse
3. **Rate limiting**: Prevent spam attacks
4. **Logging**: Track all alerts (without sensitive data)
5. **Error handling**: Don't expose API keys in errors

---

## Rollout Plan

### Step 1: Setup Twilio Account (30 minutes)
1. Sign up for Twilio account
2. Get credentials and phone number
3. Verify recipient phone number

### Step 2: Basic Implementation (2-3 hours)
1. Install Twilio SDK
2. Create alert service module
3. Integrate with app.py
4. Test SMS sending

### Step 3: Testing & Refinement (1-2 hours)
1. Test with real violations
2. Verify deduplication integration
3. Test rate limiting
4. Adjust message formatting

### Step 4: Advanced Features (Optional, 2-4 hours)
1. Phone call alerts
2. Multi-recipient support
3. Alert history logging
4. Dashboard integration

---

## Success Criteria

✅ **SMS alerts sent** when new violations detected
✅ **Deduplication respected** - no duplicate alerts
✅ **Rate limiting works** - prevents spam
✅ **Error handling robust** - doesn't crash on failures
✅ **Message formatting clear** - easy to understand alerts
✅ **Configurable** - easy to enable/disable and customize

---

## Future Enhancements

1. **WhatsApp integration** (Twilio supports WhatsApp)
2. **Email alerts** as backup
3. **Mobile app push notifications**
4. **Alert acknowledgment** (reply to confirm)
5. **Alert dashboard** with history
6. **Scheduled reports** (daily/weekly summaries)

---

## Documentation Updates Needed

1. Update README with Twilio setup instructions
2. Add configuration guide
3. Document environment variables
4. Add troubleshooting guide

---

## Questions to Consider

1. **Alert frequency preference**: How often do you want alerts? (Current: once per cooldown period)
2. **Phone calls**: Only for critical violations or all?
3. **Multi-recipient**: Do you want to notify multiple people?
4. **Alert format**: Preferred message format/structure?
5. **Business hours**: Only send alerts during work hours?

---

## Quick Start Checklist

- [ ] Create Twilio account
- [ ] Get Twilio credentials (SID, Auth Token)
- [ ] Get Twilio phone number
- [ ] Verify recipient phone number
- [ ] Install Twilio SDK: `pip install twilio`
- [ ] Create alert service module
- [ ] Configure environment variables
- [ ] Integrate with app.py
- [ ] Test SMS sending
- [ ] Verify deduplication integration
- [ ] Monitor for first real alert!

---

**Ready to implement?** Start with Phase 1 for basic SMS alerts, then add advanced features as needed.
