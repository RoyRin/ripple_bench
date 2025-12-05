#!/usr/bin/env python3
"""
Example script to run the WMDP Ripple Effect Pipeline
"""

from wmdp_ripple_effect_pipeline import WMDPRippleEffectPipeline

# Initialize pipeline
pipeline = WMDPRippleEffectPipeline(output_dir="ripple_results_example")

# Run pipeline with a small sample for testing
results = pipeline.run_pipeline(
    wmdp_path="notebooks/wmdp/wmdp-bio.json",  # Path to WMDP questions
    num_samples=10,  # Process only 10 questions for testing
    k_neighbors=3,   # Get 3 related topics for each main topic
    questions_per_topic=3,  # Generate 3 questions per topic
    base_model_path=None,  # Add path to base model if available
    unlearned_model_path=None  # Add path to unlearned model if available
)

print("Pipeline execution complete!")
print(f"Results saved to: ripple_results_example/")

# If you have model paths, you can run with evaluation:
# results = pipeline.run_pipeline(
#     wmdp_path="notebooks/wmdp/wmdp-bio.json",
#     num_samples=50,
#     k_neighbors=5,
#     questions_per_topic=5,
#     base_model_path="path/to/base/model",
#     unlearned_model_path="path/to/unlearned/model"
# )