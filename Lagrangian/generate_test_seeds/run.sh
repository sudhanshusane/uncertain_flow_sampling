#!/bin/bash

NUM_SEEDS=10000
XBOUND=200
YBOUND=100

g++ generate_test_seeds.cxx -o GenSeeds
./GenSeeds $NUM_SEEDS $XBOUND $YBOUND
