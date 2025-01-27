#!/bin/bash

# load modules
module load python/3.10.12-fasrc01
module load intelpython/3.9.16-fasrc01
module load cuda cudnn
mamba activate unlearning_3.10


# go to the right place
pushd /n/home04/rrinberg/data_dir/unlearning

CODE_DIR=/n/home04/rrinberg/code/unlearning-through-influence/unlearning

# call python
python $CODE_DIR/trak_cifar10.py

popd

