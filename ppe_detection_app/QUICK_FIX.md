# Quick Fix: Not Receiving Alerts

## Immediate Steps

### 1. Check if messages are actually being sent

Visit this URL in your browser while your app is running:
```
http://localhost:5000/twilio_send_test
```

This will:
- Send a test SMS
- Show you the result
- Display phone numbers being used

### 2. Most Common Issue: Phone Number Verification

**If you have a Twilio trial account**, you MUST verify your recipient phone number:

1. Go to: https://console.twilio.com/us1/develop/phone-numbers/manage/verified
2. Click "Add a new number" 
3. Enter your phone number (where you want alerts)
4. Twilio sends a code via SMS
5. Enter the code to verify

**Trial accounts CANNOT send SMS to unverified numbers!**

### 3. Check Phone Number Format

Your recipient phone number must be in **E.164 format**:

✅ **Correct Examples:**
- US: `+11234567890` 
- India: `+918262963069` (your number format)

❌ **Wrong:**
- `8262963069` (missing + and country code)
- `+82 6296 3069` (spaces not allowed)
- `(826) 296-3069` (wrong format)

### 4. Check Your Phone

- ✅ Check your **regular SMS/Messages app** (not WhatsApp)
- ✅ Check **spam/junk folder**
- ✅ Wait **1-2 minutes** (delivery can take time)
- ✅ Check if **airplane mode** is off
- ✅ Check if **Do Not Disturb** is blocking messages

### 5. Check Twilio Console

1. Go to: https://console.twilio.com/us1/monitor/logs/sms
2. Look for recent messages
3. Check status:
   - ✅ **delivered** = Message received
   - ⚠️ **sent** = Sent but may be pending
   - ❌ **failed** = Check error message
   - ❌ **undelivered** = Couldn't deliver (wrong number?)

### 6. Verify Environment Variables

In the PowerShell session where your app is running, check:

```powershell
echo $env:TWILIO_PHONE_NUMBER
echo $env:ALERT_RECIPIENT_PHONE
```

Make sure:
- Both are set
- Both start with `+`
- Both include country code

---

## Quick Test

**Option 1: Use the test endpoint**
```
http://localhost:5000/twilio_send_test
```

**Option 2: Check Twilio stats**
```
http://localhost:5000/twilio_stats
```

---

## What to Check in Console Output

When a violation is detected, you should see:
```
✅ Twilio SMS sent - Violation: helmet, vest
   📱 To: +918262963069
   📞 From: +1234567890
   🆔 Message SID: SMxxxxxxxxxxxxx
   📊 Status: queued
```

If status shows:
- ✅ **queued/sent/delivered** = Message sent (check your phone)
- ❌ **failed** = There's an error (check message details)

---

## Still Not Working?

1. ✅ Run test: `http://localhost:5000/twilio_send_test`
2. ✅ Verify number: https://console.twilio.com/us1/develop/phone-numbers/manage/verified
3. ✅ Check Twilio console: https://console.twilio.com/us1/monitor/logs/sms
4. ✅ Check phone number format (E.164: +918262963069)
5. ✅ Check your SMS app (not WhatsApp)
6. ✅ Wait 1-2 minutes for delivery

---

**Most likely issue**: Phone number not verified (if trial account) or wrong format!
