#!/bin/bash
set -x 
sbatch baselines_eval.slrm
sbatch scrub_eval.slrm
sbatch ga_gd_eval.slrm

