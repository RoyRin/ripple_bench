#!/usr/bin/env python3
"""
Quick script to check Anthropic API spending.
Can be run directly or imported.
"""

from ripple_bench.anthropic_utils import print_spending_summary, get_total_spending, SPENDING_LOG_FILE

if __name__ == "__main__":
    # Print the summary
    print_spending_summary()
    
    # Also show the total as a simple number
    spending = get_total_spending()
    print(f"\n💸 Quick Total: ${spending['total']:.2f}")