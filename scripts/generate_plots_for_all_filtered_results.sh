#!/bin/bash
# Generate plots for all filtered result directories
# This script runs headline_figure_generation_sept_23.py on each filtered results directory
# Generates 3 plot types: combined, distance, and comparison

set -e  # Exit on error

# Base directory containing all results
RESULTS_BASE="/Users/roy/data/ripple_bench/9_25_2025/results"

# Plot types to generate
PLOT_TYPES=("combined" "distance" "comparison" "progression")

# Find all directories matching the pattern
RESULT_DIRS=$(find "$RESULTS_BASE" -maxdepth 1 -type d -name "all_models__*" | sort)

if [ -z "$RESULT_DIRS" ]; then
    echo "No matching directories found in $RESULTS_BASE"
    exit 1
fi

# Count directories
DIR_COUNT=$(echo "$RESULT_DIRS" | wc -l | xargs)
TOTAL_PLOTS=$((DIR_COUNT * ${#PLOT_TYPES[@]}))

echo "════════════════════════════════════════════════"
echo "  Ripple Bench Multi-Plot Generator"
echo "════════════════════════════════════════════════"
echo ""
echo "Found $DIR_COUNT result directories:"
echo "$RESULT_DIRS" | sed 's|^|  - |'
echo ""
echo "Will generate ${#PLOT_TYPES[@]} plot types per directory:"
for plot_type in "${PLOT_TYPES[@]}"; do
    echo "  - $plot_type"
done
echo ""
echo "Total plots to generate: $TOTAL_PLOTS"
echo "════════════════════════════════════════════════"
echo ""

# Process each directory
counter=1
for dir in $RESULT_DIRS; do
    dir_name=$(basename "$dir")

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[$counter/$DIR_COUNT] Processing: $dir_name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    # Generate each plot type
    for plot_type in "${PLOT_TYPES[@]}"; do
        echo "  Generating $plot_type plot..."

        python3 notebooks/headline_figure_generation_sept_23.py \
            --dataset bio \
            --plot "$plot_type" \
            --checkpoint ckpt8 \
            --results-fn legacy \
            --dir "$dir"

        echo "  ✓ $plot_type plot completed"
        echo ""
    done

    echo "✓ All plots completed for: $dir_name"
    echo ""
    ((counter++))
done

echo "════════════════════════════════════════════════"
echo "✅ All $TOTAL_PLOTS plots generated successfully!"
echo "════════════════════════════════════════════════"
echo ""
echo "Plot types generated:"
echo "  - Combined plots: bio_ckpt8__<dir_name>_<timestamp>/"
echo "  - Distance progression: distance_progression_bio_<dir_name>_<timestamp>/"
echo "  - Method comparison: method_comparison_bio_<dir_name>_<timestamp>/"
echo "  - Checkpoint progression: checkpoint_progression_bio_<dir_name>_<timestamp>/"
echo ""
echo "Plots saved to: notebooks/plots/"
echo ""
echo "To view the plots, run:"
echo "  ls -lt notebooks/plots/ | head -20"
echo ""
echo "To find plots by type:"
echo "  ls -d notebooks/plots/bio_ckpt8__* | tail -5                      # Combined plots"
echo "  ls -d notebooks/plots/distance_progression_bio_* | tail -5         # Distance plots"
echo "  ls -d notebooks/plots/method_comparison_bio_* | tail -5            # Comparison plots"
echo "  ls -d notebooks/plots/checkpoint_progression_bio_* | tail -5       # Progression plots"
