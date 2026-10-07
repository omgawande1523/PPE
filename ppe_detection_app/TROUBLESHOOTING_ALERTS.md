# Troubleshooting: Not Receiving Alerts

## Quick Fixes

### 1. Check Message Status
Run the troubleshooting script:
```bash
cd ppe_detection_app
python check_twilio_status.py
```

This will show:
- ✅ Configuration status
- ✅ Connection test
- ✅ Recent messages and their status
- ✅ Phone number verification status
- ⚠️ Any errors

### 2. Verify Your Phone Number (TRIAL ACCOUNTS)

**If you have a Twilio trial account**, you MUST verify your recipient phone number:

1. Go to: https://console.twilio.com/us1/develop/phone-numbers/manage/verified
2. Click "Add a new number"
3. Enter your phone number (where you want to receive alerts)
4. Twilio will send a verification code
5. Enter the code to verify

**Trial accounts CANNOT send SMS to unverified numbers!**

### 3. Check Twilio Console

1. Go to: https://console.twilio.com/us1/monitor/logs/sms
2. Check recent messages
3. Look at message status:
   - ✅ **delivered** = Message sent successfully
   - ⚠️ **sent** = Message sent but may not be delivered yet
   - ❌ **failed** = Message failed (check error message)
   - ❌ **undelivered** = Message couldn't be delivered

### 4. SMS vs WhatsApp

**Important**: The current implementation sends **SMS**, not WhatsApp!

- ✅ SMS goes to your phone's regular messages
- ❌ WhatsApp messages require different setup (WhatsApp Business API)

To check if you received SMS:
- Check your phone's regular Messages/SMS app
- Check spam/junk folder
- Wait 1-2 minutes (delivery can take time)

### 5. Check Console Output

When a violation is detected, you should see:
```
✅ Twilio SMS sent - Violation: helmet, vest
   📱 To: +1234567890
   📞 From: +1234567890
   🆔 Message SID: SMxxxxxxxxxxxxx
   📊 Status: queued
```

If you see errors:
- ❌ **Invalid phone number**: Check phone format (must be E.164: +1234567890)
- ❌ **Unverified number**: Verify your number in Twilio console
- ❌ **Rate limit**: Wait a few minutes

---

## Step-by-Step Debugging

### Step 1: Verify Configuration
```bash
python check_twilio_status.py
```

### Step 2: Check Recent Messages
```bash
# Check Twilio console online:
# https://console.twilio.com/us1/monitor/logs/sms
```

### Step 3: Test with Simple Script
```python
from alerts import TwilioAlertService
import twilio_config

service = TwilioAlertService()
result = service.send_violation_alert({
    'missing': ['helmet'],
    'timestamp': '2025-01-19 13:00:00',
    'person_count': 1
})

print(f"Result: {result}")
```

### Step 4: Check Phone Number Format

Phone numbers must be in **E.164 format**:
- ✅ Correct: `+11234567890` (US)
- ✅ Correct: `+918262963069` (India)
- ❌ Wrong: `11234567890` (missing +)
- ❌ Wrong: `(123) 456-7890` (wrong format)
- ❌ Wrong: `1234567890` (no country code)

### Step 5: Verify Trial Account Limitations

If you see "TRIAL ACCOUNT DETECTED" in the status checker:
1. You MUST verify recipient number
2. Go to: https://console.twilio.com/us1/develop/phone-numbers/manage/verified
3. Add and verify your number
4. Restart your application

---

## Common Issues & Solutions

### Issue: "Message sent" but not received

**Possible causes:**
1. ✅ **Trial account + unverified number** → Verify number
2. ✅ **Wrong phone number** → Check E.164 format
3. ✅ **Spam folder** → Check spam/junk
4. ✅ **Delivery delay** → Wait 2-3 minutes
5. ✅ **Message failed** → Check Twilio console for errors

### Issue: "Invalid phone number" error

**Solution:**
- Use E.164 format: `+` + country code + number
- Example: US number `(123) 456-7890` → `+11234567890`
- Example: India number `8262963069` → `+918262963069`

### Issue: "Unverified number" error

**Solution:**
- Verify your number at: https://console.twilio.com/us1/develop/phone-numbers/manage/verified
- This is REQUIRED for trial accounts
- Paid accounts don't need verification

### Issue: Looking for WhatsApp but receiving SMS

**Solution:**
- Current implementation sends SMS only
- WhatsApp requires:
  1. WhatsApp Business API setup
  2. Different Twilio configuration
  3. Sandbox approval or Business verification

---

## Check Your Environment Variables

Make sure these are set correctly:

```bash
# Windows PowerShell
$env:TWILIO_ACCOUNT_SID
$env:TWILIO_AUTH_TOKEN
$env:TWILIO_PHONE_NUMBER
$env:ALERT_RECIPIENT_PHONE
$env:ENABLE_TWILIO_ALERTS="true"
```

To check current values:
```powershell
echo $env:TWILIO_PHONE_NUMBER
echo $env:ALERT_RECIPIENT_PHONE
```

---

## Still Not Working?

1. ✅ Run: `python check_twilio_status.py`
2. ✅ Check Twilio console for message logs
3. ✅ Verify phone number is verified (trial accounts)
4. ✅ Check phone number format (E.164)
5. ✅ Check spam/junk folder
6. ✅ Wait 2-3 minutes for delivery
7. ✅ Check Twilio account balance/limits

---

## Need Help?

1. Check Twilio console logs: https://console.twilio.com/us1/monitor/logs/sms
2. Run troubleshooting script: `python check_twilio_status.py`
3. Check application console output for errors
4. Verify all environment variables are set correctly

---

**Remember**: 
- 📱 SMS goes to regular Messages app (not WhatsApp)
- ✅ Verify recipient number for trial accounts
- 📞 Check phone number format (E.164: +1234567890)
