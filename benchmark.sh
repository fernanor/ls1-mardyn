#!/bin/bash

NUM_WARPS=
NUM_LOCAL_STORAGE_WARPS=
MAX_REGISTER_COUNT=

LOGFILE="benchmark.$(date +%Y.%m.%d.%H.%M.%S).log"

# find all cfg files
ALL_CFGs=$(find benchmark/*.cfg)

function log {
	echo -e $1
	echo -e $1 >> $LOGFILE
}

function fatal_error {
	log $1
	
	# dump log
	cat $LOGFILE
	exit 1
}

# CUDA_CONFIG
function build {
	echo "Building $1 with $NUM_WARPS warps${NUM_LOCAL_STORAGE_WARPS:+, $NUM_LOCAL_STORAGE_WARPS local storage warps}${MAX_REGISTER_COUNT:+, $MAX_REGISTER_COUNT max registers count}:\n"
	
	BUILD_NUM_WARPS=$NUM_WARPS
	BUILD_NUM_LOCAL_STORAGE_WARPS=${NUM_LOCAL_STORAGE_WARPS:-$NUM_WARPS}
	BUILD_MAX_REGISTER_COUNT=${MAX_REGISTER_COUNT:-63}
	
	make -C src clean
	make -C src TARGET=RELEASE CUDA_CONFIG=$1 NUM_WARPS=$BUILD_NUM_WARPS NUM_LOCAL_STORAGE_WARPS=$BUILD_NUM_LOCAL_STORAGE_WARPS MAX_REGISTER_COUNT=$BUILD_MAX_REGISTER_COUNT 

	if [ $? != "0" ]; then
		fatal_error "TARGET=RELEASE CUDA_CONFIG=$1 NUM_WARPS=$BUILD_NUM_WARPS NUM_LOCAL_STORAGE_WARPS=$BUILD_NUM_LOCAL_STORAGE_WARPS MAX_REGISTER_COUNT=$BUILD_MAX_REGISTER_COUNT failed"
	fi
}

# CUDA_CONFIG NUM_WARPS CFGS
function benchmark {
	build $1

	for cfg in $2; do
		outputPrefix=$( echo $1_${BUILD_NUM_WARPS}_${BUILD_NUM_LOCAL_STORAGE_WARPS}_${BUILD_MAX_REGISTER_COUNT}_$(basename $cfg .cfg) | tr '[:upper:]' '[:lower:]' )
		if [ ! -f benchmark/$outputPrefix.results.csv ]; then
			echo -e "\nBenchmarking $outputPrefix:\n"
			./src/MarDyn.$1.${BUILD_NUM_WARPS}.${BUILD_NUM_LOCAL_STORAGE_WARPS}.${BUILD_MAX_REGISTER_COUNT} $cfg 10 $outputPrefix
			if [ $? != "0" ]; then
				log "$outputPrefix benchmark failed!"
				echo "continuing"
			else
				log "$outputPrefix benchmarked successfully"
			fi
			
		else
			log "$outputPrefix has been benchmarked already" 
		fi
	done
}

case $1 in
	"no_constant_memory" )
		log "Benchmarking no constant memory vs constant memory:"
		CFGs=$(echo benchmark/lj_{1,2,4,8,16,32}0000.cfg benchmark/lj3d1_{1,5,10}0000.cfg benchmark/lj3d1_lj2d1_{1,5,10}0000.cfg)
		
		NUM_WARPS=1
		
		benchmark CUDA_DOUBLE_SORTED_NO_CONSTANT_MEMORY "$CFGs"
		benchmark CUDA_DOUBLE_SORTED "$CFGs"
		;;
	"warp_count_with_cache" )
		log "Benchmarking different warp counts with cache:"
		CFGs=$(echo benchmark/lj_{8,16,32,64,128}0000.cfg benchmark/lj3d1_{1,5,10}0000.cfg benchmark/lj3d1_lj2d1_{1,5,10}0000.cfg)
		for NUM_WARPS in {1,2,4,8}; do
			benchmark CUDA_DOUBLE_SORTED "$CFGs"
		done
		;;
	* )
		log "Benchmarking everything without cuda:"
		
		benchmark NO_CUDA "$ALL_CFGs"
		;;
esac

echo -e "\n\n"
# dump log
cat $LOGFILE
exit 0