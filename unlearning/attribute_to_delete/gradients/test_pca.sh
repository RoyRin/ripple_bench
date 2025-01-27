
#!/bin/bash


module load python/3.10.12-fasrc01
module load intelpython/3.9.16-fasrc01
module load cuda cudnn

mamba activate unlearning_3.10

CODE_BASE=/n/home04/rrinberg/code/unlearning-with-trak/

set -x 
pushd $CODE_BASE

which python
export CUDA_LAUNCH_BLOCKING=1

python $CODE_BASE/unlearning/gradients/pca_test.py 