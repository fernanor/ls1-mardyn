#include <host_defines.h>

#include <stdio.h>

#include "cutil_math.h"

#include "moleculeStorage.cum"

#include "componentDescriptor.cum"

#include "pairTraverser.cum"

#include "cellProcessor.cum"

#include "globalStats.cum"

#include "cellInfo.cum"

#include "molecule.cum"

#include "potForce.cum"

#include "moleculePairHandler.cum"

#include "config.h"

#ifndef REFERENCE_IMPLEMENTATION
#warning using fast cell processor
#else
#warning using reference cell processor
#endif

#ifdef CUDA_DOUBLE_MODE
#	warning using double precision
#else
#	warning using float precision
#endif

#ifdef CUDA_SORT_CELLS_BY_COMPONENTTYPE
#	warning sorting cells by component type
#else
#	warning cells are *not* sorted by component type
#endif

#ifdef CUDA_HW_CACHE_ONLY
#	warning no shared local storage cache
#else
#	warning shared local storage active
#endif

#ifdef NO_CONSTANT_MEMORY
#	warning no constant memory
#else
#	warning constant memory used
#endif

extern "C" {
/* TODO: possible refactoring
 * create a prepare method in MoleculeStorage and make rawQuaternions a global pointer
 * and forward the kernel call to it
 */
// TODO: interesting to benchmark idea: unrolled loop in this kernel vs the way it is now---overhead?
__global__ void convertQuaternionsToRotations( const QuaternionStorage *rawQuaternions, int numMolecules ) {
	const Quaternion *quaternions = (Quaternion*) rawQuaternions;

	int moleculeIndex = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if( moleculeIndex >= numMolecules ) {
		return;
	}

#ifndef TEST_QUATERNION_MATRIX_CONVERSION
	moleculeRotations[ moleculeIndex ] = quaternions[ moleculeIndex ].toRotMatrix3x3();
#else
#	warning CUDA: testing quaternion matrix conversion
	const Matrix3x3 convertedQuaternion = quaternions[ moleculeIndex ].toRotMatrix3x3();
	const Matrix3x3 &correctRotation = moleculeRotations[ moleculeIndex ];

	const floatType error = length( convertedQuaternion.rows[0] - correctRotation.rows[0] ) +
			length( convertedQuaternion.rows[1] - correctRotation.rows[1] ) +
			length( convertedQuaternion.rows[2] - correctRotation.rows[2] );

	if( error > 1e-9 ) {
		printf( "bad quaternion conversion (molecule %i)\n", moleculeIndex );
	}
#endif
}

__global__ void processCellPair( int numCellPairs, int startIndex, int2 dimension, int3 gridOffsets, int neighborOffset ) {
	const int threadIndex = threadIdx.y * WARP_SIZE + threadIdx.x;

	const int cellPairIndex = blockIdx.y * gridDim.x + blockIdx.x;
	if( cellPairIndex >= numCellPairs ) {
		return;
	}

	int cellIndex = getCellIndex( cellPairIndex, startIndex, dimension, gridOffsets );
	int neighborIndex = cellIndex + neighborOffset;

	// TODO: move the swapping bit into the cell processor!
	/*int cellLength = cellInfos[ cellIndex + 1 ] - cellInfos[ cellIndex ];
	int neighborLength = cellInfos[ neighborIndex + 1 ] - cellInfos[ neighborIndex ];

	// ensure that cellA_length <= cellB_length (which will use fewer data transfers)
	// (numTransfersA + numTransfersA * numTransfersB) * BLOCK_SIZE
	if( cellLength > neighborLength ) {
		// swap cellIndex and neighborIndex
		cellIndex = neighborIndex;
		neighborIndex -= neighborOffset;
	}*/

	__shared__ CellStatsCollector<BLOCK_SIZE> globalStatsCollector;
	globalStatsCollector.initThreadLocal( threadIndex );

	ComponentDescriptorAccessor componentDescriptorAccessor;

	MoleculeStorage moleculeStorage;

	MoleculePairHandler<typeof(globalStatsCollector), typeof(componentDescriptorAccessor)> moleculePairHandler( globalStatsCollector, componentDescriptorAccessor );

#ifndef REFERENCE_IMPLEMENTATION
#	ifndef CUDA_HW_CACHE_ONLY
	__shared__ SharedMoleculeLocalStorage<LOCAL_STORAGE_BLOCK_SIZE> moleculeLocalStorage;
#	else
	WriteThroughMoleculeLocalStorage<LOCAL_STORAGE_BLOCK_SIZE> moleculeLocalStorage( moleculeStorage );
#	endif
	HighDensityCellProcessor<BLOCK_SIZE, LOCAL_STORAGE_BLOCK_SIZE, Molecule, typeof(moleculeStorage), typeof(moleculeLocalStorage), typeof(moleculePairHandler)> cellProcessor(moleculeStorage, moleculeLocalStorage, moleculePairHandler);
#else
	ReferenceCellProcessor<Molecule, typeof(moleculeStorage), typeof(moleculePairHandler)> cellProcessor(moleculeStorage, moleculePairHandler);
#endif

	cellProcessor.processCellPair( threadIndex, cellInfoFromCellIndex( cellIndex ), cellInfoFromCellIndex( neighborIndex ) );

	globalStatsCollector.reduceAndSave( threadIndex, cellIndex, neighborIndex );
}

//__launch_bounds__(BLOCK_SIZE, 2)
__global__ void processCell(int numCells) {
	const int threadIndex = threadIdx.y * WARP_SIZE + threadIdx.x;

	int cellIndex = blockIdx.y * gridDim.x + blockIdx.x;
	if( cellIndex >= numCells ) {
		return;
	}

	__shared__ CellStatsCollector<BLOCK_SIZE> globalStatsCollector;
	globalStatsCollector.initThreadLocal( threadIndex );

	ComponentDescriptorAccessor componentDescriptorAccessor;

	MoleculeStorage moleculeStorage;

	MoleculePairHandler<typeof(globalStatsCollector), typeof(componentDescriptorAccessor)> moleculePairHandler( globalStatsCollector, componentDescriptorAccessor );

#ifndef REFERENCE_IMPLEMENTATION
#	ifndef CUDA_HW_CACHE_ONLY
	__shared__ SharedMoleculeLocalStorage<BLOCK_SIZE> moleculeLocalStorage;
#	else
	WriteThroughMoleculeLocalStorage<LOCAL_STORAGE_BLOCK_SIZE> moleculeLocalStorage( moleculeStorage );
#	endif
	HighDensityCellProcessor<BLOCK_SIZE, BLOCK_SIZE, Molecule, typeof(moleculeStorage), typeof(moleculeLocalStorage), typeof(moleculePairHandler)> cellProcessor(moleculeStorage, moleculeLocalStorage, moleculePairHandler);
#else
	ReferenceCellProcessor<Molecule, typeof(moleculeStorage), typeof(moleculePairHandler)> cellProcessor(moleculeStorage, moleculePairHandler);
#endif

	cellProcessor.processCell( threadIndex, cellInfoFromCellIndex( cellIndex ) );

	globalStatsCollector.reduceAndSave( threadIndex, cellIndex, cellIndex );
}
}
