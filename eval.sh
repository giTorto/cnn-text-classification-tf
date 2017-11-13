#!/bin/bash

#PBS -l walltime=12:0:0

#PBS -q gpuq

#PBS -e evaluation.log

sh /home/giuliano.tortoreto/slu/cnn-text-classification-tf/eval.actual.sh
