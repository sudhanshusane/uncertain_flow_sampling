#!/bin/bash

INPUT="/home/sci/ssane/data/DoubleGyre/FlowMaps/Uniform/Uniform_1_64.vtk"
ITERATION="10"
SEEDS="seeds.txt"
NUM_SEEDS="10000"
GT="/home/sci/ssane/projects/uncertainty_vis/DoubleGyre/Lagrangian/generate_gt_pathlines/DoubleGyre_Pathlines_GT.vtk"
THRESHOLD="1.0"
TEST="Uniform_R3"


./Reconstruct_Pathlines $INPUT $ITERATION $SEEDS $NUM_SEEDS $GT $THRESHOLD $TEST
