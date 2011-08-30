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

#define BENCHMARKING

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


//#define NO_CUDA

//#define USE_CONSTANT_MEMORY

//#define REFERENCE_IMPLEMENTATION
//#define TEST_QUATERNION_MATRIX_CONVERSION
//#define COMPARE_TO_CPU
//#define USE_BEHAVIOR_PROBE

//#define DEBUG_COMPONENT_DESCRIPTORS

// TODO: either use warpSize or WARP_SIZE!
//#define MAX_NUM_WARPS 6

#define MAX_NUM_COMPONENTS 4

#define MAX_NUM_LJCENTERS 4
#define MAX_NUM_DIPOLES 2
#define MAX_NUM_CHARGES 2


#ifndef REFERENCE_IMPLEMENTATION
#	define WARP_SIZE 32
#	define NUM_WARPS MAX_NUM_WARPS
#else
#	define WARP_SIZE 1
#	define NUM_WARPS 1
#endif

#define BLOCK_SIZE (WARP_SIZE*NUM_WARPS)

#ifndef CUDA_DOUBLE_MODE
	typedef float floatType;
	typedef float3 floatType3;

#	define make_floatType3 make_float3
#else
	typedef double floatType;
	typedef double3 floatType3;

#	define make_floatType3 make_double3
#endif

#ifndef USE_CONSTANT_MEMORY
#	define __constant__
#endif

// TODO: move this include into the referencing header files
#include "sharedDecls.h"

#endif /* CONFIG_H_ */
