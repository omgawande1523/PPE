"""
Test script for Phase 2 deduplication features
Tests state machine, violation info, and state-based queries
"""

from deduplication import ViolationDeduplicator, ViolationState
from datetime import datetime
import time

def test_phase2_features():
    print("=" * 70)
    print("Testing Phase 2: State Machine & Pattern-Based Deduplication")
    print("=" * 70)
    
    # Create deduplicator with 5 second cooldown for testing
    deduplicator = ViolationDeduplicator(cooldown_seconds=5)
    
    # Test 1: New violation state
    print("\n[Test 1] New Violation - State should be NEW")
    missing1 = ['helmet', 'vest']
    deduplicator.register_violation(missing1)
    info1 = deduplicator.get_violation_info(missing1)
    print(f"  Missing PPE: {missing1}")
    print(f"  State: {info1['state'] if info1 else 'Not found'}")
    print(f"  Alert Count: {info1['alert_count'] if info1 else 0}")
    assert info1 and info1['state'] == 'new', "Violation should be in NEW state"
    
    # Test 2: Active state transition
    print("\n[Test 2] Immediate duplicate - State should transition to ACTIVE")
    missing2 = ['helmet', 'vest']
    is_dup = deduplicator.is_duplicate(missing2)
    info2 = deduplicator.get_violation_info(missing2)
    print(f"  Is duplicate: {is_dup}")
    print(f"  State: {info2['state'] if info2 else 'Not found'}")
    assert is_dup == True, "Should be duplicate"
    assert info2 and info2['state'] == 'active', "Violation should be ACTIVE"
    
    # Test 3: Expired state
    print("\n[Test 3] Waiting for cooldown to expire...")
    for i in range(5, 0, -1):
        print(f"  {i}...", end=' ', flush=True)
        time.sleep(1)
    print("\n")
    
    info3 = deduplicator.get_violation_info(missing1)
    print(f"  State after cooldown: {info3['state'] if info3 else 'Not found'}")
    assert info3 and info3['state'] == 'expired', "Violation should be EXPIRED"
    
    # Test 4: Re-register expired violation
    print("\n[Test 4] Re-register expired violation - Should become NEW again")
    deduplicator.register_violation(missing1)
    info4 = deduplicator.get_violation_info(missing1)
    print(f"  State: {info4['state'] if info4 else 'Not found'}")
    print(f"  Alert Count: {info4['alert_count'] if info4 else 0}")
    assert info4 and info4['state'] == 'new', "Should transition back to NEW"
    assert info4['alert_count'] == 2, "Should have 2 alerts now"
    
    # Test 5: State-based queries
    print("\n[Test 5] Query violations by state")
    new_violations = deduplicator.get_violations_by_state(ViolationState.NEW)
    active_violations = deduplicator.get_violations_by_state(ViolationState.ACTIVE)
    expired_violations = deduplicator.get_violations_by_state(ViolationState.EXPIRED)
    
    print(f"  NEW violations: {len(new_violations)}")
    print(f"  ACTIVE violations: {len(active_violations)}")
    print(f"  EXPIRED violations: {len(expired_violations)}")
    
    # Test 6: Mark as resolved
    print("\n[Test 6] Mark violation as resolved")
    deduplicator.mark_violation_resolved(missing1)
    info6 = deduplicator.get_violation_info(missing1)
    resolved_violations = deduplicator.get_violations_by_state(ViolationState.RESOLVED)
    print(f"  State: {info6['state'] if info6 else 'Not found'}")
    print(f"  RESOLVED violations: {len(resolved_violations)}")
    assert info6 and info6['state'] == 'resolved', "Should be RESOLVED"
    
    # Test 7: Enhanced statistics
    print("\n[Test 7] Enhanced statistics with state breakdown")
    stats = deduplicator.get_stats()
    print(f"  Total violations: {stats['total_violations']}")
    print(f"  Duplicates suppressed: {stats['duplicates_suppressed']}")
    print(f"  New violations: {stats['new_violations']}")
    print(f"  Suppression rate: {stats['suppression_rate_percent']}%")
    print(f"  State breakdown:")
    for state, count in stats['state_breakdown'].items():
        print(f"    {state.upper()}: {count}")
    print(f"  Avg alerts per violation: {stats['avg_alerts_per_violation']}")
    
    # Test 8: Multiple violations
    print("\n[Test 8] Multiple different violations")
    missing3 = ['gloves']
    missing4 = ['boots', 'safety_glasses']
    deduplicator.register_violation(missing3)
    deduplicator.register_violation(missing4)
    
    stats_final = deduplicator.get_stats()
    print(f"  Total active violations: {stats_final['active_violations_count']}")
    print(f"  Total new violations registered: {stats_final['new_violations']}")
    
    print("\n" + "=" * 70)
    print("All Phase 2 tests completed successfully!")
    print("=" * 70)
    
    # Final summary
    print("\nFinal Statistics Summary:")
    print("-" * 70)
    final_stats = deduplicator.get_stats()
    for key, value in final_stats.items():
        if key != 'state_breakdown':
            print(f"  {key}: {value}")
        else:
            print(f"  {key}:")
            for state, count in value.items():
                print(f"    {state}: {count}")

if __name__ == "__main__":
    test_phase2_features()
