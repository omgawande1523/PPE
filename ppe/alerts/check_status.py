"""
Check Twilio Message Status and Troubleshooting Tool
Use this to debug why messages aren't being received.
Run: python -m ppe.alerts.check_status
"""

import sys
from datetime import datetime, timedelta

# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def check_twilio_setup():
    """Check Twilio configuration and setup"""
    print("=" * 70)
    print("Twilio Setup & Status Checker")
    print("=" * 70)
    
    try:
        from ppe.alerts import config
        from twilio.rest import Client
        from twilio.base.exceptions import TwilioRestException
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("   Install Twilio SDK: pip install twilio")
        return False
    
    print("\n1. Configuration Check:")
    print("-" * 70)
    
    # Check configuration
    is_valid, error_msg = config.validate_config()
    if not is_valid:
        print(f"❌ Configuration Invalid: {error_msg}")
        return False
    
    print("✅ Configuration valid")
    print(f"   Twilio Account SID: {config.TWILIO_ACCOUNT_SID[:10]}...")
    print(f"   Twilio Phone: {config.TWILIO_PHONE_NUMBER}")
    print(f"   Recipient Phone: {config.ALERT_RECIPIENT_PHONE}")
    print(f"   SMS Enabled: {config.ALERT_SMS_ENABLED}")
    
    print("\n2. Twilio Connection Test:")
    print("-" * 70)
    
    try:
        client = Client(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)
        account = client.api.accounts(config.TWILIO_ACCOUNT_SID).fetch()
        
        print(f"✅ Connected to Twilio")
        print(f"   Account Name: {account.friendly_name}")
        print(f"   Account Status: {account.status}")
        
        # Check if trial account
        if account.type == 'Trial':
            print(f"\n⚠️  TRIAL ACCOUNT DETECTED")
            print(f"   Trial accounts can only send to VERIFIED phone numbers!")
            print(f"   Verify your number at: https://console.twilio.com/us1/develop/phone-numbers/manage/verified")
            print(f"   Your recipient number must be verified: {config.ALERT_RECIPIENT_PHONE}")
        
    except TwilioRestException as e:
        print(f"❌ Connection Failed: {e}")
        print(f"   Error Code: {e.code}")
        print(f"   Error Message: {e.msg}")
        return False
    
    print("\n3. Recent Messages Check:")
    print("-" * 70)
    
    try:
        # Get messages from last 24 hours
        yesterday = datetime.now() - timedelta(days=1)
        messages = client.messages.list(
            date_sent_after=yesterday,
            to=config.ALERT_RECIPIENT_PHONE,
            limit=10
        )
        
        if messages:
            print(f"✅ Found {len(messages)} messages in last 24 hours:")
            for msg in messages:
                status_icon = "✅" if msg.status == "delivered" else "⚠️" if msg.status == "sent" else "❌"
                print(f"\n{status_icon} Message SID: {msg.sid}")
                print(f"   Status: {msg.status}")
                print(f"   From: {msg.from_}")
                print(f"   To: {msg.to}")
                print(f"   Sent: {msg.date_sent}")
                print(f"   Body: {msg.body[:50]}..." if len(msg.body) > 50 else f"   Body: {msg.body}")
                
                if msg.status in ['failed', 'undelivered']:
                    if hasattr(msg, 'error_code') and msg.error_code:
                        print(f"   ⚠️  Error Code: {msg.error_code}")
                    if hasattr(msg, 'error_message') and msg.error_message:
                        print(f"   ⚠️  Error: {msg.error_message}")
        else:
            print("⚠️  No messages found in last 24 hours")
            print("   This might mean:")
            print("   1. No messages were sent")
            print("   2. Messages failed before being created")
            print("   3. Messages are older than 24 hours")
    
    except Exception as e:
        print(f"❌ Error checking messages: {e}")
    
    print("\n4. Phone Number Verification Check:")
    print("-" * 70)
    
    try:
        # Check if recipient number is verified (for trial accounts)
        verified_numbers = client.outgoing_caller_ids.list()
        verified_phones = [v.phone_number for v in verified_numbers]
        
        if config.ALERT_RECIPIENT_PHONE in verified_phones:
            print(f"✅ Recipient number is VERIFIED: {config.ALERT_RECIPIENT_PHONE}")
        else:
            print(f"⚠️  Recipient number NOT VERIFIED: {config.ALERT_RECIPIENT_PHONE}")
            if account.type == 'Trial':
                print(f"   ❌ TRIAL ACCOUNTS CAN ONLY SEND TO VERIFIED NUMBERS!")
                print(f"   📝 Verify at: https://console.twilio.com/us1/develop/phone-numbers/manage/verified")
            else:
                print(f"   ✅ Not required for paid accounts")
        
        if verified_phones:
            print(f"\n   Verified numbers in your account:")
            for phone in verified_phones:
                print(f"   - {phone}")
    
    except Exception as e:
        print(f"⚠️  Could not check verification: {e}")
    
    print("\n5. Test Message Send:")
    print("-" * 70)
    print("To send a test message, run:")
    print("  python -c \"from alerts import TwilioAlertService; s=TwilioAlertService(); s.send_violation_alert({'missing':['helmet'],'timestamp':'test','person_count':1})\"")
    
    print("\n" + "=" * 70)
    print("Troubleshooting Tips:")
    print("=" * 70)
    print("1. ✅ Check phone number format: Must be E.164 (e.g., +1234567890)")
    print("2. ✅ Verify recipient number is VERIFIED (required for trial accounts)")
    print("3. ✅ Check Twilio console for message logs: https://console.twilio.com")
    print("4. ✅ Check spam/junk folder in your phone")
    print("5. ✅ Wait a few minutes - messages can take time to deliver")
    print("6. ✅ Check message status in Twilio console")
    print("\n📱 Note: This is SMS, not WhatsApp. WhatsApp requires different setup.")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    check_twilio_setup()
