#!/bin/bash

cmake -DCMAKE_C_COMPILER=/usr/local/bin/gcc -DCGAL_DIR=/home/sci/ssane/packages/CGAL-5.2.1 -DCMAKE_BUILD_TYPE=Release -DCGAL_HEADER_ONLY="ON" -DBOOST_ROOT=/home/sci/ssane/packages/boost_1_57_0 .
