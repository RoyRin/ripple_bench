from pathlib import Path
import numpy as np
import sys
from train_living import wrapper_for_train_living17_on_subset_submitit

BASE_SAVE_PATH = Path("/mnt/xfs/projects/untrak/MATCHING/oracles/LIVING17")
FORGET_SETS_PATH = Path("/mnt/xfs/projects/untrak/MATCHING/forget_set_inds/LIVING17")

N_models_per_job = 1
N_models_per_forget_set = 250
N_jobs_per_forget_set = N_models_per_forget_set // N_models_per_job

MASK_PATHS = []

recreate_masks = False

for SET_PATH in sorted(list(FORGET_SETS_PATH.iterdir())):
    key = SET_PATH.stem
    if key == 'forget_set_1':
        continue
    print(f"key: {key}")
    print(f"forget_set_path: {SET_PATH}")
    if recreate_masks:
        forget_set = np.load(SET_PATH)
        print(len(forget_set))
        mask = np.ones(44_200)
        mask[forget_set] = 0
        mask = mask.astype(bool)
        mask = np.stack([mask] * N_models_per_job, axis=0)
        print(mask.shape)

    MASK_DIR = BASE_SAVE_PATH / key
    MASK_PATH = MASK_DIR / "mask.npy"
    print(MASK_PATH)
    MASK_PATHS.append(MASK_PATH)
    if recreate_masks:
        MASK_DIR.mkdir(exist_ok=True, parents=True)
        np.save(MASK_PATH, mask)


batch_args = []

for MASK_PATH in MASK_PATHS:
    CKPT_PATH = MASK_PATH.parent
    for i in range(N_jobs_per_forget_set):
        idx_start = i * N_models_per_job
        model_id_offset = idx_start
        should_save_train_logits = True
        should_save_val_logits = True
        batch_args.append([MASK_PATH,
                           0,
                           N_models_per_job,
                           CKPT_PATH,
                           should_save_train_logits,
                           should_save_val_logits,
                           model_id_offset,
                           [24]
                           ])

ID = int(sys.argv[1])
wrapper_for_train_living17_on_subset_submitit(*batch_args[ID])