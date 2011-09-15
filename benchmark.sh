#!/bin/bash

NUM_WARPS=
MAX_REGISTER_COUNT=

MAX_NUM_COMPONENTS=
MAX_NUM_LJCENTERS=
MAX_NUM_CHARGES=
MAX_NUM_DIPOLES=

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
	echo "Building $1 with $NUM_WARPS warps${MAX_REGISTER_COUNT:+, $MAX_REGISTER_COUNT max registers count}:\n"
	
	BUILD_NUM_WARPS=$NUM_WARPS
	BUILD_MAX_REGISTER_COUNT=${MAX_REGISTER_COUNT:-63}
	BUILD_MAX_NUM_COMPONENTS=${MAX_NUM_COMPONENTS:-2}
	BUILD_MAX_NUM_LJCENTERS=${MAX_NUM_LJCENTERS:-1}
	BUILD_MAX_NUM_CHARGES=${MAX_NUM_CHARGES:-0}
	BUILD_MAX_NUM_DIPOLES=${MAX_NUM_DIPOLES:-0}
	
	make -C src TARGET=RELEASE CUDA_CONFIG=$1 NUM_WARPS=$BUILD_NUM_WARPS MAX_REGISTER_COUNT=$BUILD_MAX_REGISTER_COUNT  MAX_NUM_COMPONENTS=$BUILD_MAX_NUM_COMPONENTS MAX_NUM_LJCENTERS=$BUILD_MAX_NUM_LJCENTERS MAX_NUM_CHARGES=$BUILD_MAX_NUM_CHARGES MAX_NUM_DIPOLES=$BUILD_MAX_NUM_DIPOLES

	if [ $? != "0" ]; then
		fatal_error "TARGET=RELEASE CUDA_CONFIG=$1 NUM_WARPS=$BUILD_NUM_WARPS MAX_REGISTER_COUNT=$BUILD_MAX_REGISTER_COUNT  MAX_NUM_COMPONENTS=$BUILD_MAX_NUM_COMPONENTS MAX_NUM_LJCENTERS=$BUILD_MAX_NUM_LJCENTERS MAX_NUM_CHARGES=$BUILD_MAX_NUM_CHARGES MAX_NUM_DIPOLES=$BUILD_MAX_NUM_DIPOLES failed"
	fi
}

BENCHMARK_IN_ORDER_INDEX=0

# CUDA_CONFIG CFGS OPT_SUFFIX
function benchmark {
	BENCHMARK_BUILT=0

	BENCHMARK_CFG_INDEX=0
	for cfg in $2; do
		outputPrefix=$( echo $1_$3$(basename $cfg .cfg) | tr '[:upper:]' '[:lower:]' )
		if [ ! -f benchmark/$outputPrefix.results.csv ]; then
			if (( $BENCHMARK_BUILT == 0 )); then
				build $1
				cp src/MarDyn MarDyn_$1
				BENCHMARK_BUILT=1
			fi

			echo -e "\nBenchmarking $outputPrefix:\n"
			./src/MarDyn $cfg 4 $outputPrefix
			if [ $? != "0" ]; then
				log "$outputPrefix benchmark failed!"
				echo "continuing"
			else
				log "$outputPrefix benchmarked successfully"
			fi
		else
			log "$outputPrefix has been benchmarked already" 
		fi
		BENCHMARK_CFG_INDEX=$((BENCHMARK_CFG_INDEX+1))
	done
	BENCHMARK_IN_ORDER_INDEX=$((BENCHMARK_IN_ORDER_INDEX+1))
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
	"cell_density" )
		log "Benchmarking with different densities:"
		MAX_NUM_COMPONENTS=1
		MAX_NUM_LJCENTERS=1
		MAX_NUM_CHARGES=0
		MAX_NUM_DIPOLES=0
		
		CFGs_NORMAL=$(echo benchmark/lj_80000.cfg benchmark/lj_80000_{10,15,20,30,40}.cfg)
		for NUM_WARPS in {1,2,4,8}; do
			benchmark CUDA_DOUBLE_UNSORTED "$CFGs_NORMAL" "${NUM_WARPS}_"
		done
		
		CFGs_WBDP=$(echo benchmark/lj_80000.cfg benchmark/lj_80000_{10,15,20,30,40}.cfg)
		for NUM_WARPS in {1,2,4,8,16}; do
			benchmark CUDA_DOUBLE_UNSORTED_WBDP "$CFGs_WBDP" "${NUM_WARPS}_"
		done
		benchmark CUDA_DOUBLE_UNSORTED "benchmark/lj_80000_50.cfg" "${NUM_WARPS}_"

		NUM_WARPS=1
		CFGs_NO_CUDA=$(echo benchmark/lj_80000.cfg benchmark/lj_80000_{10,15,20,30,40}.cfg)
		benchmark NO_CUDA "$CFGs_NO_CUDA" 
		;;
	"cell_density_2" )
		log "Benchmarking with different densities (part 2):"
		MAX_NUM_COMPONENTS=2
		MAX_NUM_LJCENTERS=3
		MAX_NUM_CHARGES=0
		MAX_NUM_DIPOLES=1
		
		CFGs=$(echo benchmark/lj3d1_50000.cfg benchmark/lj3d1_50000_{10,15,20,30,40}.cfg)
		NUM_WARPS=1
		benchmark CUDA_DOUBLE_UNSORTED "$CFGs" "${NUM_WARPS}_"
				
		NUM_WARPS=8
		benchmark CUDA_DOUBLE_UNSORTED_WBDP "$CFGs" "${NUM_WARPS}_"

		NUM_WARPS=1
		CFGs_NO_CUDA=$(echo benchmark/lj3d1_50000.cfg benchmark/lj3d1_50000_{10,15,20}.cfg)
		benchmark NO_CUDA "$CFGs_NO_CUDA"
		;;
	"cell_density_3" )
		log "Benchmarking with different densities (part 3):"
		MAX_NUM_COMPONENTS=2
		MAX_NUM_LJCENTERS=3
		MAX_NUM_CHARGES=0
		MAX_NUM_DIPOLES=1
		
		CFGs=$(echo benchmark/lj3d1_lj2d1_50000.cfg benchmark/lj3d1_lj2d1_50000_{10,15,20,30,40}.cfg)
		NUM_WARPS=1
		benchmark CUDA_DOUBLE_UNSORTED "$CFGs" "${NUM_WARPS}_"
				
		NUM_WARPS=8
		benchmark CUDA_DOUBLE_UNSORTED_WBDP "$CFGs" "${NUM_WARPS}_"

		NUM_WARPS=1
		CFGs=$(echo benchmark/lj3d1_lj2d1_50000.cfg benchmark/lj3d1_lj2d1_50000_{10,15,20}.cfg)
		benchmark NO_CUDA "$CFGs_NO_CUDA"
		;;
	"sorted_vs_unsorted" )
		log "Benchmarking sorted vs unsorted on mixed molecule domains:"
		#CFGs=$(echo benchmark/lj3d1_lj2d1_50000_{40,50,60}.cfg benchmark/lj3d1_lj2d1_100000_30.cfg)
		CFGs=$(echo benchmark/lj3d1_lj2d1_50000{,_10,_15,_20}.cfg)
		
		MAX_NUM_COMPONENTS=2
		MAX_NUM_LJCENTERS=3
		MAX_NUM_CHARGES=0
		MAX_NUM_DIPOLES=1
		
		#NUM_WARPS=1
		#benchmark CUDA_DOUBLE_UNSORTED "$CFGs"
		NUM_WARPS=1
		benchmark CUDA_DOUBLE_UNSORTED "$CFGs"
		benchmark CUDA_DOUBLE_SORTED "$CFGs"
		;;
	"debug" )
		CFGs=$(echo benchmark/lj_80000.cfg benchmark/lj_80000_{10,15,20,40,50,60}.cfg)
		
		MAX_NUM_COMPONENTS=1
		MAX_NUM_LJCENTERS=1
		MAX_NUM_CHARGES=0
		MAX_NUM_DIPOLES=0
		
		NUM_WARPS=1
		benchmark CUDA_DOUBLE_UNSORTED "$CFGs"
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
