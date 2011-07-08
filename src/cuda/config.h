/*
 * config.h
 *
 *  Created on: Jun 14, 2011
 *      Author: andreas
 */

#ifndef CONFIG_H_
#define CONFIG_H_

#include "cutil_double_math.h"

#define MAX_BLOCK_SIZE 512

#define CUDA_DOUBLE_MODE

//#define REFERENCE_IMPLEMENTATION
//#define TEST_QUATERNION_MATRIX_CONVERSION
#define COMPARE_TO_CPU

#define DEBUG_COMPONENT_DESCRIPTORS

// TODO: either use warpSize or WARP_SIZE!
#ifndef REFERENCE_IMPLEMENTATION
#define WARP_SIZE 32
#define NUM_WARPS 4
#else
#define WARP_SIZE 1
#define NUM_WARPS 1
#endif

#define BLOCK_SIZE (WARP_SIZE*NUM_WARPS)

#define MAX_NUM_LJCENTERS 4
#define MAX_NUM_DIPOLES 2
#define MAX_NUM_CHARGES 2

#ifndef CUDA_DOUBLE_MODE
typedef float floatType;
typedef float3 floatType3;

#define make_floatType3 make_float3
#else
typedef double floatType;
typedef double3 floatType3;

#define make_floatType3 make_double3
#endif

// TODO: move this include into the referencing header files
#include "sharedDecls.h"

#endif /* CONFIG_H_ */
