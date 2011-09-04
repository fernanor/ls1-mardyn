/*
 * config.h
 *
 *  Created on: Jun 14, 2011
 *      Author: andreas
 */

#ifndef CONFIG_H_
#define CONFIG_H_

#include "cutil_double_math.h"

#define MAX_BLOCK_SIZE 1024
#define QUATERNION_BLOCK_SIZE MAX_BLOCK_SIZE

// created by the makefile to avoid having to call make clean all the time
#include "make_config.h"

//#define BENCHMARKING

//#define CONFIG_CUDA_DOUBLE_SORTED

#ifdef CONFIG_NO_CUDA
#	define CONFIG_NAME "no_cuda"
#	define NO_CUDA
#endif

#ifdef CONFIG_CUDA_FLOAT_UNSORTED
#	define CONFIG_NAME "cuda_float_unsorted"
#endif

#ifdef CONFIG_CUDA_DOUBLE_UNSORTED
#	define CONFIG_NAME "cuda_double_unsorted"
#	define CUDA_DOUBLE_MODE
#endif

#ifdef CONFIG_CUDA_DOUBLE_SORTED
#	define CONFIG_NAME "cuda_double_sorted"
#	define CUDA_DOUBLE_MODE
#	define CUDA_SORT_CELLS_BY_COMPONENTTYPE
#endif

#ifdef CONFIG_CUDA_DOUBLE_SORTED_NO_CONSTANT_MEMORY
#	define CONFIG_NAME "cuda_double_sorted_no_constant_memory"

#	define CUDA_DOUBLE_MODE
#	define CUDA_SORT_CELLS_BY_COMPONENTTYPE
#	define NO_CONSTANT_MEMORY
#endif

#ifdef CONFIG_CUDA_DOUBLE_SORTED_HWCACHEONLY
#	define CONFIG_NAME "cuda_double_sorted_hwcacheonly"

#	define CUDA_DOUBLE_MODE
#	define CUDA_SORT_CELLS_BY_COMPONENTTYPE

#	define CUDA_HW_CACHE_ONLY
#endif

#ifndef CONFIG_NAME
#	define CONFIG_NAME "special_cuda_config"
#endif

#define CUDA_HW_CACHE_ONLY

//#define REFERENCE_IMPLEMENTATION
//#define TEST_QUATERNION_MATRIX_CONVERSION
#define COMPARE_TO_CPU
//#define USE_BEHAVIOR_PROBE

//#define DEBUG_COMPONENT_DESCRIPTORS

// TODO: either use warpSize or WARP_SIZE!
//#define MAX_NUM_WARPS 6

#define MAX_NUM_COMPONENTS 2

#define MAX_NUM_LJCENTERS 3
#define MAX_NUM_DIPOLES 3
#define MAX_NUM_CHARGES 0

#ifndef REFERENCE_IMPLEMENTATION
#	define WARP_SIZE 32
#	define NUM_WARPS MAX_NUM_WARPS
#else
#	define WARP_SIZE 1
#	define NUM_WARPS 1
#endif

#ifndef MAX_REGISTER_COUNT
#	define MAX_REGISTER_COUNT 63
#endif

#ifndef NUM_LOCAL_STORAGE_WARPS
# 	warning set NUM_LOCAL_STORAGE_WARPS = NUM_WARPS
#	define NUM_LOCAL_STORAGE_WARPS NUM_WARPS
#endif

#if NUM_LOCAL_STORAGE_WARPS > NUM_WARPS
#	warning NUM_LOCAL_STORAGE_WARPS > NUM_WARPS => set NUM_LOCAL_STORAGE_WARPS = NUM_WARPS
#	define NUM_LOCAL_STORAGE_WARPS NUM_WARPS
#endif

#define BLOCK_SIZE (WARP_SIZE*NUM_WARPS)
#define LOCAL_STORAGE_BLOCK_SIZE (WARP_SIZE*NUM_LOCAL_STORAGE_WARPS)

#ifdef CUDA_DOUBLE_MODE
	typedef double floatType;
	typedef double3 floatType3;

#	define make_floatType3 make_double3
#else
	typedef float floatType;
	typedef float3 floatType3;

#	define make_floatType3 make_float3
#endif

#ifdef NO_CONSTANT_MEMORY
#	define __constant__
#endif

//#define __restrict__

// TODO: move this include into the referencing header files
#include "sharedDecls.h"

#endif /* CONFIG_H_ */
