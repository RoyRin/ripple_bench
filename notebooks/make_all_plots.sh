  #!/bin/bash
  # Generate all plots for both datasets

  echo "Generating BIO plots..."
python headline_figure_generation_sept_11.py --dataset bio --plot combined --checkpoint ckpt8 --wmdp
python headline_figure_generation_sept_11.py --dataset bio --plot combined --checkpoint ckpt8
python headline_figure_generation_sept_11.py --dataset bio --plot progression
python headline_figure_generation_sept_11.py --dataset bio --plot distance

python headline_figure_generation_sept_11.py --dataset bio --plot comparison

  echo "Generating CHEM plots..."

python headline_figure_generation_sept_11.py --dataset chem --plot combined --checkpoint ckpt8
#python headline_figure_generation_sept_11.py --dataset chem --plot combined --checkpoint ckpt8 --wmdp
python headline_figure_generation_sept_11.py --dataset chem --plot progression


