"""
Simple test script for deduplication functionality
Run this to verify deduplication is working correctly
"""

from deduplication import ViolationDeduplicator
from datetime import datetime
import time

def test_deduplication():
    print("=" * 60)
    print("Testing Violation Deduplicator")
    print("=" * 60)
    
    # Create deduplicator with 5 second cooldown for testing
    deduplicator = ViolationDeduplicator(cooldown_seconds=5)
    
    # Test 1: First violation should be registered
    print("\n[Test 1] First violation (should be NEW)")
    missing1 = ['helmet', 'vest']
    result1 = deduplicator.register_violation(missing1)
    print(f"  Missing PPE: {missing1}")
    print(f"  Registered: {result1}")
    print(f"  Is duplicate: {deduplicator.is_duplicate(missing1)}")
    print(f"  Stats: {deduplicator.get_stats()}")
    
    # Test 2: Same violation immediately should be duplicate
    print("\n[Test 2] Same violation immediately (should be DUPLICATE)")
    missing2 = ['helmet', 'vest']  # Same as missing1
    result2 = deduplicator.register_violation(missing2)
    print(f"  Missing PPE: {missing2}")
    print(f"  Registered: {result2}")
    print(f"  Is duplicate: {deduplicator.is_duplicate(missing2)}")
    print(f"  Stats: {deduplicator.get_stats()}")
    
    # Test 3: Different violation should be new
    print("\n[Test 3] Different violation (should be NEW)")
    missing3 = ['gloves', 'boots']
    result3 = deduplicator.register_violation(missing3)
    print(f"  Missing PPE: {missing3}")
    print(f"  Registered: {result3}")
    print(f"  Is duplicate: {deduplicator.is_duplicate(missing3)}")
    print(f"  Stats: {deduplicator.get_stats()}")
    
    # Test 4: Same violation after cooldown
    print("\n[Test 4] Waiting for cooldown period (5 seconds)...")
    for i in range(5, 0, -1):
        print(f"  {i}...", end=' ', flush=True)
        time.sleep(1)
    print("\n")
    
    print("[Test 4] Same violation after cooldown (should be NEW)")
    missing4 = ['helmet', 'vest']  # Same as missing1, but after cooldown
    result4 = deduplicator.register_violation(missing4)
    print(f"  Missing PPE: {missing4}")
    print(f"  Registered: {result4}")
    print(f"  Is duplicate: {deduplicator.is_duplicate(missing4)}")
    print(f"  Stats: {deduplicator.get_stats()}")
    
    # Test 5: Normalization (order shouldn't matter)
    print("\n[Test 5] Same items, different order (should be DUPLICATE)")
    missing5 = ['vest', 'helmet']  # Same items, different order
    result5 = deduplicator.register_violation(missing5)
    print(f"  Missing PPE: {missing5}")
    print(f"  Registered: {result5}")
    print(f"  Is duplicate: {deduplicator.is_duplicate(missing5)}")
    print(f"  Stats: {deduplicator.get_stats()}")
    
    # Final stats
    print("\n" + "=" * 60)
    print("Final Statistics:")
    print("=" * 60)
    stats = deduplicator.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)
    print(f"  ✓ Test 1 (First violation): {'PASS' if result1 else 'FAIL'}")
    print(f"  ✓ Test 2 (Immediate duplicate): {'PASS' if not result2 else 'FAIL'}")
    print(f"  ✓ Test 3 (Different violation): {'PASS' if result3 else 'FAIL'}")
    print(f"  ✓ Test 4 (After cooldown): {'PASS' if result4 else 'FAIL'}")
    print(f"  ✓ Test 5 (Normalization): {'PASS' if not result5 else 'FAIL'}")
    print("\nAll tests completed!")

if __name__ == "__main__":
    test_deduplication()
