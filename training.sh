#!/bin/bash

#PBS -l walltime=1:0:0

#PBS -q gpuq

#PBS -e training.log

sh /home/giuliano.tortoreto/slu/cnn-text-classification-tf/training.actual.sh
