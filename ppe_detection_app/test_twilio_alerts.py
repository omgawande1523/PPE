"""
Test script for Twilio Alert System (Phase 1)
Tests SMS alert functionality with mock data
"""

import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

def test_twilio_config():
    """Test Twilio configuration"""
    print("=" * 70)
    print("Testing Twilio Configuration")
    print("=" * 70)
    
    try:
        import twilio_config as config
        from twilio_config import validate_config
        
        print(f"\nConfiguration Status:")
        print(f"  ENABLE_TWILIO_ALERTS: {config.ENABLE_TWILIO_ALERTS}")
        print(f"  ALERT_SMS_ENABLED: {config.ALERT_SMS_ENABLED}")
        print(f"  TWILIO_ACCOUNT_SID: {'Set' if config.TWILIO_ACCOUNT_SID else 'NOT SET'}")
        print(f"  TWILIO_AUTH_TOKEN: {'Set' if config.TWILIO_AUTH_TOKEN else 'NOT SET'}")
        print(f"  TWILIO_PHONE_NUMBER: {config.TWILIO_PHONE_NUMBER if config.TWILIO_PHONE_NUMBER else 'NOT SET'}")
        print(f"  ALERT_RECIPIENT_PHONE: {config.ALERT_RECIPIENT_PHONE if config.ALERT_RECIPIENT_PHONE else 'NOT SET'}")
        print(f"  SMS_COOLDOWN_SECONDS: {config.SMS_COOLDOWN_SECONDS}")
        print(f"  DAILY_SMS_LIMIT: {config.DAILY_SMS_LIMIT}")
        
        is_valid, error_msg = validate_config()
        print(f"\nValidation Result: {'✅ Valid' if is_valid else '❌ Invalid'}")
        if not is_valid:
            print(f"  Error: {error_msg}")
        
        return is_valid
    
    except Exception as e:
        print(f"\n❌ Error loading configuration: {e}")
        return False


def test_alert_formatter():
    """Test alert formatter"""
    print("\n" + "=" * 70)
    print("Testing Alert Formatter")
    print("=" * 70)
    
    try:
        from alerts.alert_formatter import AlertFormatter
        
        formatter = AlertFormatter()
        
        # Test 1: Basic alert
        print("\n[Test 1] Basic Alert Formatting")
        violation_info = {
            'missing': ['helmet', 'vest'],
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'detected': ['gloves'],
            'person_count': 1
        }
        message = formatter.format_sms_alert(violation_info)
        print("Formatted Message:")
        print(message)
        print(f"  Level: {formatter.get_alert_level(violation_info['missing'])}")
        
        # Test 2: Critical alert
        print("\n[Test 2] Critical Alert Formatting")
        violation_info_critical = {
            'missing': ['helmet', 'vest', 'gloves', 'boots'],
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'detected': [],
            'person_count': 1
        }
        message_critical = formatter.format_sms_alert(violation_info_critical)
        print("Formatted Message:")
        print(message_critical)
        print(f"  Level: {formatter.get_alert_level(violation_info_critical['missing'])}")
        
        # Test 3: Single item
        print("\n[Test 3] Single Item Alert")
        violation_info_single = {
            'missing': ['gloves'],
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'detected': ['helmet', 'vest'],
            'person_count': 1
        }
        message_single = formatter.format_sms_alert(violation_info_single)
        print("Formatted Message:")
        print(message_single)
        print(f"  Level: {formatter.get_alert_level(violation_info_single['missing'])}")
        
        return True
    
    except Exception as e:
        print(f"\n❌ Error testing formatter: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rate_limiter():
    """Test rate limiter"""
    print("\n" + "=" * 70)
    print("Testing Rate Limiter")
    print("=" * 70)
    
    try:
        from alerts.rate_limiter import RateLimiter
        
        limiter = RateLimiter(sms_cooldown=5, daily_limit=10)  # Short cooldown for testing
        
        # Test 1: First SMS
        print("\n[Test 1] First SMS - Should be allowed")
        signature1 = "helmet|vest"
        can_send, reason = limiter.can_send_sms(signature1)
        print(f"  Can send: {can_send}")
        if can_send:
            limiter.record_sms_sent(signature1)
            print("  ✅ SMS recorded")
        else:
            print(f"  ❌ Blocked: {reason}")
        
        # Test 2: Immediate duplicate
        print("\n[Test 2] Immediate duplicate - Should be blocked")
        can_send2, reason2 = limiter.can_send_sms(signature1)
        print(f"  Can send: {can_send2}")
        if not can_send2:
            print(f"  ✅ Correctly blocked: {reason2}")
        else:
            print("  ❌ Should have been blocked!")
        
        # Test 3: Different violation
        print("\n[Test 3] Different violation - Should be allowed")
        signature2 = "gloves"
        can_send3, reason3 = limiter.can_send_sms(signature2)
        print(f"  Can send: {can_send3}")
        if can_send3:
            limiter.record_sms_sent(signature2)
            print("  ✅ SMS recorded")
        else:
            print(f"  ❌ Blocked: {reason3}")
        
        # Test 4: Stats
        print("\n[Test 4] Rate Limiter Statistics")
        stats = limiter.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        return True
    
    except Exception as e:
        print(f"\n❌ Error testing rate limiter: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_twilio_service():
    """Test Twilio service (requires valid credentials)"""
    print("\n" + "=" * 70)
    print("Testing Twilio Service")
    print("=" * 70)
    
    try:
        from alerts import TwilioAlertService
        
        print("\n[Test 1] Initialize Twilio Service")
        try:
            service = TwilioAlertService()
            print("  ✅ Service initialized")
        except Exception as e:
            print(f"  ❌ Failed to initialize: {e}")
            print("  Note: This is expected if Twilio credentials are not configured")
            return False
        
        # Test 2: Connection test
        print("\n[Test 2] Test Twilio Connection")
        test_result = service.test_connection()
        if test_result['success']:
            print(f"  ✅ Connection successful")
            print(f"  Account: {test_result.get('account_name', 'N/A')}")
            print(f"  Status: {test_result.get('status', 'N/A')}")
        else:
            print(f"  ❌ Connection failed: {test_result.get('error', 'Unknown error')}")
            return False
        
        # Test 3: Send test alert (uncomment to actually send)
        print("\n[Test 3] Send Test Alert")
        print("  ⚠️  Uncomment the code below to actually send an SMS")
        """
        violation_info = {
            'missing': ['helmet', 'vest'],
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'detected': ['gloves'],
            'person_count': 1,
            'location': 'Test Location'
        }
        
        result = service.send_violation_alert(violation_info)
        if result['success']:
            print(f"  ✅ Alert sent successfully!")
            print(f"  Message SID: {result.get('message_sid', 'N/A')}")
        elif result.get('skipped'):
            print(f"  ⚠️  Alert skipped: {result.get('reason', 'Unknown')}")
        else:
            print(f"  ❌ Alert failed: {result.get('error', 'Unknown error')}")
        """
        
        # Test 4: Stats
        print("\n[Test 4] Service Statistics")
        stats = service.get_stats()
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")
        
        return True
    
    except ImportError as e:
        print(f"\n⚠️  Twilio SDK not installed: {e}")
        print("  Install with: pip install twilio")
        return False
    except ValueError as e:
        print(f"\n⚠️  Configuration error: {e}")
        print("  Set environment variables or configure twilio_config.py")
        return False
    except Exception as e:
        print(f"\n❌ Error testing Twilio service: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("Twilio Alert System - Phase 1 Tests")
    print("=" * 70)
    
    results = []
    
    # Test 1: Configuration
    results.append(("Configuration", test_twilio_config()))
    
    # Test 2: Alert Formatter
    results.append(("Alert Formatter", test_alert_formatter()))
    
    # Test 3: Rate Limiter
    results.append(("Rate Limiter", test_rate_limiter()))
    
    # Test 4: Twilio Service (requires credentials)
    results.append(("Twilio Service", test_twilio_service()))
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    print(f"\nOverall: {'✅ ALL TESTS PASSED' if all_passed else '⚠️  SOME TESTS FAILED'}")
    print("\nNote: Twilio Service test requires valid credentials.")
    print("Set environment variables or configure twilio_config.py to test actual SMS sending.")


if __name__ == "__main__":
    main()
