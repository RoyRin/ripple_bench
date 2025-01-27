import time

out_dir = "/mnt/xfs/projects/untrak/MATCHING/full_models/SHAKESPEARE/"
eval_interval = 99
eval_iters = 200
num_models = 20

wandb_log = False  # feel free to turn on
wandb_project = "shakespeare"
wandb_run_name = "ft-" + str(time.time())

dataset = "shakespeare"
init_from = "gpt2-medium"

# only save checkpoints if the validation loss improves
always_save_checkpoint = True

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
batch_size = 1
gradient_accumulation_steps = 32
max_iters = 100

# finetune at constant LR
learning_rate = 3e-4
decay_lr = False
