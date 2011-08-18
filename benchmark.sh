#!/bin/bash

# CUDA_CONFIG NUM_WARPS
function build {
	make -C src TARGET=RELEASE CUDA_CONFIG=$1 NUM_WARPS=$2

	if [ $? != "0" ]; then
		exit 1
	fi	
}

# find all cfg files and use them to execute all chosen builds with them
CFGs=$(find benchmark/*.cfg)

# CUDA_CONFIG NUM_WARPS
function benchmark {
	build $1 $2

	for cfg in $CFGs; do
		outputPrefix=$( echo $1_$(basename $cfg .cfg) | tr '[:upper:]' '[:lower:]' )
		if [ ! -f benchmark/$outputPrefix.results.csv ]; then
			./src/MarDyn.$1.$2 $cfg 10 $outputPrefix
			#echo $outputPrefix doesn\'t exist
		else
			echo $outputPrefix has been benchmarked already 
		fi
	done
}

benchmark NO_CUDA 1

exit

for numWarps in {1..6}; do
	benchmark CUDA_DOUBLE_SORTED $numWarps
	benchmark CUDA_DOUBLE_UNSORTED $numWarps
	benchmark CUDA_FLOAT_UNSORTED $numWarps
done
