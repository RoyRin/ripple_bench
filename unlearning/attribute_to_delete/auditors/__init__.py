from pathlib import Path 
ULIRA_BASE_DIR = Path(
    "/n/home04/rrinberg/data_dir__holylabs/unlearning/precomputed_models/ULIRA_clean/"
)
if not ULIRA_BASE_DIR.exists():
    ULIRA_BASE_DIR = Path("/mnt/xfs/projects/untrak/ULIRA/ULIRA_clean/")