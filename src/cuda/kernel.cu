#include <host_defines.h>

__device__ int getSM() {
	uint smID;

	asm volatile( "mov.u32 %0, %smid;" : "=r"(smID) );

	return smID;
}

#if 0
#	define WARP_PRINTF(format, ...) if( threadIdx.x == 0 ) printf( "(%i W%i {%i}) " format, blockIdx.x + blockIdx.y * gridDim.x, threadIdx.y, getSM(), ##__VA_ARGS__ )
#	define BLOCK_PRINTF(format, ...) if( threadIdx.y == 0 ) printf( "(%i {%i}) " format, blockIdx.x + blockIdx.y * gridDim.x, getSM(), ##__VA_ARGS__ )
#	define GRID_PRINTF(format, ...) if( blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0 ) printf( "{%i} " format, getSM(), ##__VA_ARGS__ )
#else
#	define WARP_PRINTF(format, ...)
#	define BLOCK_PRINTF(format, ...)
#	define GRID_PRINTF(format, ...)
#endif

#include <stdio.h>

#include "cutil_math.h"

#include "moleculeStorage.cum"

#include "componentDescriptor.cum"

#include "domainTraverser.cum"

#include "cellProcessor.cum"

#include "globalStats.cum"

#include "cellInfo.cum"

#include "molecule.cum"

#include "potForce.cum"

#include "moleculePairHandler.cum"

#include "warpBlockCellProcessor.cum"

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
__global__ void convertQuaternionsToRotations( int numMolecules ) {
	int moleculeIndex = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if( moleculeIndex >= numMolecules ) {
		return;
	}

#ifndef CUDA_UNPACKED_STORAGE
	const Quaternion quaternion = moleculeQuaternions[ moleculeIndex ];
#else
	const Quaternion quaternion = packQuaternion( moleculeQuaternions, moleculeIndex );
#endif

#ifndef TEST_QUATERNION_MATRIX_CONVERSION
#	ifndef CUDA_UNPACKED_STORAGE
	moleculeRotations[ moleculeIndex ] = quaternion.toRotMatrix3x3();
#	else
	unpackMatrix3x3( moleculeRotations, moleculeIndex, quaternion.toRotMatrix3x3() );
#	endif
#else
#	warning CUDA: testing quaternion matrix conversion
	const Matrix3x3 convertedQuaternion = quaternion.toRotMatrix3x3();
	const Matrix3x3 &correctRotation = moleculeRotations[ moleculeIndex ];

	const floatType error = length( convertedQuaternion.rows[0] - correctRotation.rows[0] ) +
			length( convertedQuaternion.rows[1] - correctRotation.rows[1] ) +
			length( convertedQuaternion.rows[2] - correctRotation.rows[2] );

	if( error > 1e-9 ) {
		printf( "bad quaternion conversion (molecule %i)\n", moleculeIndex );
	}
#endif
}

#ifndef CUDA_WARP_BLOCK_CELL_PROCESSOR

__shared__ CellStatsCollector<BLOCK_SIZE> globalStatsCollector;
__device__ MoleculePairHandler<typeof(globalStatsCollector), globalStatsCollector> moleculePairHandler;

__device__ MoleculeStorage moleculeStorage;

#ifndef REFERENCE_IMPLEMENTATION
#	ifndef CUDA_HW_CACHE_ONLY
__shared__ SharedMoleculeLocalStorage<BLOCK_SIZE, typeof(moleculeStorage), moleculeStorage> moleculeLocalStorage;
#	else
__device__ WriteThroughMoleculeLocalStorage<BLOCK_SIZE, typeof(moleculeStorage), moleculeStorage> moleculeLocalStorage;
#	endif
__device__ HighDensityCellProcessor<BLOCK_SIZE, BLOCK_SIZE,
	typeof(moleculeStorage), moleculeStorage,
	typeof(moleculeLocalStorage), moleculeLocalStorage,
	typeof(moleculePairHandler), moleculePairHandler>
		cellProcessor;

#else
__device__ ReferenceCellProcessor<
	typeof(moleculeStorage), moleculeStorage,
	typeof(moleculePairHandler), moleculePairHandler>
		cellProcessor;
#endif

__global__ void processCellPair() {
	const int threadIndex = threadIdx.y * WARP_SIZE + threadIdx.x;

	const int jobIndex = blockIdx.y * gridDim.x + blockIdx.x;
	if( jobIndex >= PairTraverser::numJobs ) {
		return;
	}

	int cellIndex = PairTraverser::getCellIndexFromJobIndex( jobIndex );
	int neighborIndex = PairTraverser::getNeighborCellIndex( cellIndex );

	CellInfoEx cellA = cellInfoFromCellIndex( cellIndex );
	CellInfoEx cellB = cellInfoFromCellIndex( neighborIndex );

	if( cellA.length == 0 || cellB.length == 0 ) {
		return;
	}

	// TODO: move the swapping bit into the cell processor!?
	/*int cellLength = cellInfos[ cellIndex + 1 ] - cellInfos[ cellIndex ];
	int neighborLength = cellInfos[ neighborIndex + 1 ] - cellInfos[ neighborIndex ];

	// ensure that cellA_length <= cellB_length (which will use fewer data transfers)
	// (numTransfersA + numTransfersA * numTransfersB) * BLOCK_SIZE
	if( cellLength > neighborLength ) {
		// swap cellIndex and neighborIndex
		cellIndex = neighborIndex;
		neighborIndex -= neighborOffset;
	}*/

	globalStatsCollector.initThreadLocal( threadIndex );
	cellProcessor.processCellPair( threadIndex, cellA, cellB );

	globalStatsCollector.reduceAndStore( threadIndex, cellIndex, neighborIndex );
}

//__launch_bounds__(BLOCK_SIZE, 2)
__global__ void processCell() {
	const int threadIndex = threadIdx.y * WARP_SIZE + threadIdx.x;

	int jobIndex = blockIdx.y * gridDim.x + blockIdx.x;
	if( jobIndex >= PairTraverser::numJobs ) {
		return;
	}

	int cellIndex = PairTraverser::getInnerCellIndexFromJobIndex(jobIndex);
	CellInfoEx cell = cellInfoFromCellIndex( cellIndex );
	if( cell.length == 0 ) {
		return;
	}

	globalStatsCollector.initThreadLocal( threadIndex );
	cellProcessor.processCell( threadIndex, cell );

	globalStatsCollector.reduceAndStore( threadIndex, cellIndex, cellIndex );
}
#else

__device__ WarpBlockMode::CellScheduler< NUM_WARPS > *cellScheduler;
__device__ WarpBlockMode::CellPairScheduler< NUM_WARPS > *cellPairScheduler;

__global__ void createSchedulers() {
	if( threadIdx.x + threadIdx.y == 0 ) {
		cellScheduler = new WarpBlockMode::CellScheduler< NUM_WARPS >();
		cellPairScheduler = new WarpBlockMode::CellPairScheduler< NUM_WARPS >();
	}
}

__global__ void destroySchedulers() {
	if( threadIdx.x + threadIdx.y == 0 ) {
		delete cellScheduler;
		delete cellPairScheduler;
	}
}

__shared__ CellStatsCollector<BLOCK_SIZE> globalStatsCollector;
__device__ MoleculePairHandler<typeof(globalStatsCollector), globalStatsCollector> moleculePairHandler;
__device__ MoleculeStorage moleculeStorage;
__device__ WarpBlockMode::CellProcessor<NUM_WARPS,
	typeof(moleculeStorage), moleculeStorage,
	typeof(moleculePairHandler), moleculePairHandler>
		cellProcessor;

__global__ void processCellPair() {
	const int threadIndex = threadIdx.y * WARP_SIZE + threadIdx.x;

	__shared__ WarpBlockMode::ThreadBlockInfo<NUM_WARPS, WarpBlockMode::WarpBlockPairInfo> warpInfos;

	do {
		cellPairScheduler->scheduleWarpBlocks( warpInfos );

		while( !warpInfos.warpJobQueue[threadIdx.y].isEmpty() ) {
			globalStatsCollector.initThreadLocal( threadIndex );

			WarpBlockMode::WarpBlockPairInfo warpBlockPairInfo = warpInfos.warpJobQueue[threadIdx.y].pop();
			cellProcessor.processCellPair( warpBlockPairInfo );

			globalStatsCollector.reduceAndStoreWarp( threadIndex, warpBlockPairInfo.warpBlockA.cellIndex );
		}
	} while( warpInfos.hasMoreJobs );

	WARP_PRINTF( "terminating..\n" );
}

__global__ void processCell() {
	// TODO: remove?
	const int threadIndex = threadIdx.y * WARP_SIZE + threadIdx.x;

	__shared__ WarpBlockMode::ThreadBlockInfo<NUM_WARPS, WarpBlockMode::WarpBlockPairInfo> warpInfos;

	do {
		cellScheduler->scheduleWarpBlocks( warpInfos );

		while( !warpInfos.warpJobQueue[threadIdx.y].isEmpty() ) {
			globalStatsCollector.initThreadLocal( threadIndex );

			WarpBlockMode::WarpBlockPairInfo warpBlockPairInfo = warpInfos.warpJobQueue[threadIdx.y].pop();
			cellProcessor.processCellPair( warpBlockPairInfo );

			globalStatsCollector.reduceAndStoreWarp( threadIndex, warpBlockPairInfo.warpBlockA.cellIndex );
		}
	} while( warpInfos.hasMoreJobs );

	WARP_PRINTF( "terminating\n" );
}

#endif

}
