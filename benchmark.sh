#!/bin/bash

# CUDA_CONFIG NUM_WARPS
function build {
	make -C src TARGET=RELEASE CUDA_CONFIG=$1 NUM_WARPS=$2	
}

# find all cfg files and use them to execute all chosen builds with them
CFGs=$(find benchmark/*.cfg)

# CUDA_CONFIG NUM_WARPS
function benchmark {
	build $1 $2

	for cfg in $CFGs; do
		MarDyn.$1.$2 $cfg 10 $(basename $cfg)
	done
}

benchmark NO_CUDA 0

for numWarps in {1..6}; do
	benchmark CUDA_DOUBLE_SORTED $numWarps
done
