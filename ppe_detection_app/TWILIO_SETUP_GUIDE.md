# Twilio Alert Setup Guide

## Quick Start

Follow these steps to set up Twilio SMS alerts for your PPE detection system.

---

## Step 1: Install Twilio SDK

```bash
pip install twilio
```

---

## Step 2: Get Twilio Account

1. **Sign up for Twilio**:
   - Go to https://www.twilio.com
   - Create a free account (includes $15.50 trial credit)

2. **Get your credentials**:
   - Account SID: Found in dashboard
   - Auth Token: Found in dashboard (click to reveal)

3. **Get a phone number**:
   - Go to Phone Numbers → Buy a Number
   - Select a number (free trial numbers available)
   - Note the number in E.164 format (e.g., `+1234567890`)

4. **Verify your phone** (for trial accounts):
   - Go to Phone Numbers → Verified Caller IDs
   - Add and verify your phone number (where you want to receive alerts)

---

## Step 3: Configure Environment Variables

Create a `.env` file in `ppe_detection_app/` directory or set environment variables:

### Windows (PowerShell):
```powershell
$env:TWILIO_ACCOUNT_SID="your_account_sid_here"
$env:TWILIO_AUTH_TOKEN="your_auth_token_here"
$env:TWILIO_PHONE_NUMBER="+1234567890"
$env:ALERT_RECIPIENT_PHONE="+1234567890"
$env:ENABLE_TWILIO_ALERTS="true"
$env:ALERT_SMS_ENABLED="true"
```

### Windows (Command Prompt):
```cmd
set TWILIO_ACCOUNT_SID=your_account_sid_here
set TWILIO_AUTH_TOKEN=your_auth_token_here
set TWILIO_PHONE_NUMBER=+1234567890
set ALERT_RECIPIENT_PHONE=+1234567890
set ENABLE_TWILIO_ALERTS=true
set ALERT_SMS_ENABLED=true
```

### Linux/Mac:
```bash
export TWILIO_ACCOUNT_SID="your_account_sid_here"
export TWILIO_AUTH_TOKEN="your_auth_token_here"
export TWILIO_PHONE_NUMBER="+1234567890"
export ALERT_RECIPIENT_PHONE="+1234567890"
export ENABLE_TWILIO_ALERTS="true"
export ALERT_SMS_ENABLED="true"
```

**Important**: 
- Phone numbers must be in E.164 format: `+` followed by country code and number
- Example: US number `(123) 456-7890` becomes `+11234567890`
- No spaces, dashes, or parentheses

---

## Step 4: Test Configuration

Run the test script:

```bash
cd ppe_detection_app
python test_twilio_alerts.py
```

This will:
- ✅ Validate configuration
- ✅ Test alert formatting
- ✅ Test rate limiting
- ✅ Test Twilio connection (if credentials are set)

---

## Step 5: Start Your Application

```bash
python app.py
```

You should see:
```
Twilio alerts enabled
Twilio connection verified - Account: Your Account Name
```

---

## Step 6: Test with Real Alert

To test sending an actual SMS, you can:

1. **Trigger a violation** through your camera feed
2. **Use the test endpoint** (if you add it):
   ```bash
   curl http://localhost:5000/twilio_test
   ```

---

## Configuration Options

### Alert Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_TWILIO_ALERTS` | `false` | Enable/disable Twilio alerts |
| `ALERT_SMS_ENABLED` | `true` | Enable SMS alerts |
| `ALERT_CALL_ENABLED` | `false` | Enable phone call alerts (not in Phase 1) |
| `SMS_COOLDOWN_SECONDS` | `60` | Minimum seconds between SMS for same violation |
| `DAILY_SMS_LIMIT` | `100` | Maximum SMS per day |
| `CRITICAL_VIOLATION_THRESHOLD` | `3` | Missing items count for critical alerts |
| `ALERT_LOCATION` | `Workplace` | Location name in alerts |

### Rate Limiting

- **Per-violation cooldown**: Same violation won't send SMS more than once per `SMS_COOLDOWN_SECONDS`
- **Daily limit**: Maximum `DAILY_SMS_LIMIT` SMS per day
- **Automatic reset**: Daily counter resets at midnight

---

## API Endpoints

### Get Twilio Statistics
```bash
GET /twilio_stats
```

Response:
```json
{
  "enabled": true,
  "sms_enabled": true,
  "call_enabled": false,
  "total_sent": 5,
  "total_failed": 0,
  "rate_limiter": {
    "daily_sms_count": 5,
    "daily_limit": 100,
    "sms_remaining_today": 95,
    "cooldown_seconds": 60,
    "tracked_violations": 3
  }
}
```

### Test Twilio Connection
```bash
GET /twilio_test
```

Response:
```json
{
  "success": true,
  "account_sid": "ACxxxxx",
  "account_name": "My Account",
  "status": "active"
}
```

### Get Detection Status (includes Twilio info)
```bash
GET /detection_status
```

---

## Troubleshooting

### "Twilio alerts disabled: TWILIO_ACCOUNT_SID is not set"
- Set the `TWILIO_ACCOUNT_SID` environment variable
- Or edit `twilio_config.py` directly (not recommended for production)

### "Invalid phone number"
- Ensure phone numbers are in E.164 format: `+1234567890`
- Include country code (e.g., `+1` for US)
- No spaces or special characters

### "Rate limit exceeded"
- You've hit Twilio's rate limit
- Free tier has limits
- Wait a few minutes or upgrade account

### "Recipient unsubscribed"
- Your phone number unsubscribed from Twilio messages
- Re-verify in Twilio dashboard

### SMS not sending
1. Check console logs for errors
2. Verify credentials are correct
3. Verify recipient number is verified (trial accounts)
4. Check `/twilio_stats` endpoint for errors

### Module not found errors
```bash
pip install twilio
```

---

## Cost Estimation

### Twilio Pricing (Approximate):
- **SMS (US)**: ~$0.0075 per message
- **Free Trial**: $15.50 credit (enough for ~2,000 SMS)

### Example Monthly Costs:
- **100 alerts/day**: ~$22.50/month (SMS only)
- **500 alerts/day**: ~$112.50/month
- **1000 alerts/day**: ~$225/month

**Note**: With deduplication, actual alerts will be much fewer than detections!

---

## Security Notes

⚠️ **Never commit credentials to git!**

- Use environment variables
- Add `.env` to `.gitignore`
- Use separate accounts for development/production
- Rotate credentials regularly

---

## Next Steps

Once Phase 1 is working:

1. **Phase 2**: Add phone call alerts for critical violations
2. **Multi-recipient**: Notify multiple people
3. **Alert templates**: Customize message format
4. **Alert history**: Log all sent alerts

---

## Support

- Twilio Documentation: https://www.twilio.com/docs
- Twilio Console: https://console.twilio.com
- Check logs: Look for `✅ Twilio alert sent` in console

---

**Happy Alerting! 🚨📱**
