#!/usr/bin/env python3
"""
Check Anthropic API spending from the log file.
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import csv

from ripple_bench.anthropic_utils import get_total_spending, print_spending_summary, SPENDING_LOG_FILE


def check_spending(log_file: Path = SPENDING_LOG_FILE, days: int = None, show_details: bool = True):
    """Check and display API spending."""
    
    if not log_file.exists():
        print(f"No spending log found at {log_file}")
        return
    
    # Use the new summary function for main report
    print_spending_summary(log_file)
    
    # If days specified, show recent spending
    if days:
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_total = 0.0
        recent_by_model = {}
        
        with open(log_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                timestamp = datetime.fromisoformat(row['timestamp'])
                if timestamp >= cutoff_date:
                    cost = float(row['total_cost'])
                    recent_total += cost
                    
                    model = row['model']
                    if model not in recent_by_model:
                        recent_by_model[model] = 0.0
                    recent_by_model[model] += cost
        
        print(f"\n--- Last {days} days ---")
        print(f"Total: ${recent_total:.2f}")
        
        if recent_by_model:
            print("\nBy model:")
            for model, cost in sorted(recent_by_model.items(), key=lambda x: x[1], reverse=True):
                print(f"  {model}: ${cost:.2f}")
    
    # Show recent entries if requested
    if show_details:
        print("\n--- Recent API calls ---")
        with open(log_file, 'r') as f:
            lines = list(csv.DictReader(f))
            
        for row in lines[-10:]:  # Last 10 entries
            timestamp = datetime.fromisoformat(row['timestamp'])
            print(f"\n{timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  Model: {row['model']}")
            print(f"  Tokens: {row['input_tokens']} in, {row['output_tokens']} out")
            print(f"  Cost: ${float(row['total_cost']):.4f}")
            print(f"  Prompt: {row['prompt_preview'][:50]}...")


def main():
    parser = argparse.ArgumentParser(description="Check Anthropic API spending")
    parser.add_argument(
        "--log-file",
        type=Path,
        default=SPENDING_LOG_FILE,
        help="Path to spending log file"
    )
    parser.add_argument(
        "--days",
        type=int,
        help="Show spending for last N days"
    )
    parser.add_argument(
        "--no-details",
        action="store_true",
        help="Don't show recent API calls"
    )
    
    args = parser.parse_args()
    check_spending(args.log_file, args.days, show_details=not args.no_details)


if __name__ == "__main__":
    main()