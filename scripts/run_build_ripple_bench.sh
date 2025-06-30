#!/bin/bash
# Wrapper script to run build_ripple_bench_from_wmdp.py with correct Python path

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the parent directory (project root)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Add project root to PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Run the Python script with all arguments passed through
python "$SCRIPT_DIR/build_ripple_bench_from_wmdp.py" "$@"