#include "cutil_math.h"

#include "moleculeStorage.cum"

#include "pairTraverser.cum"

#include "cellProcessor.cum"

#include "globalStats.cum"

#include "cellInfo.cum"

#include "molecule.cum"

#include "potForce.cum"

#include "moleculePairHandler.cum"

#define BLOCK_SIZE 4*32

__global__ void processCellPair( int startIndex, int2 dimension, int3 gridOffsets, int neighborOffset ) {
	const int threadIndex = threadIdx.y * warpSize + threadIdx.x;

	int cellIndex = getCellIndex( startIndex, dimension, gridOffsets );
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

	MoleculeStorage moleculeStorage;

	MoleculePairHandler<typeof(globalStatsCollector)> moleculePairHandler( globalStatsCollector );

	FastCellProcessor<BLOCK_SIZE, Molecule, typeof(moleculeStorage), typeof(moleculePairHandler)> cellProcessor(moleculeStorage, moleculePairHandler);

	cellProcessor.processCellPair( threadIndex, fromCellIndex( cellIndex ), fromCellIndex( neighborIndex ) );

	globalStatsCollector.reduceAndSave( threadIndex, cellIndex, neighborIndex );
}

__global__ void processCell() {
	const int threadIndex = threadIdx.y * warpSize + threadIdx.x;

	int cellIndex = blockIdx.x;

	__shared__ CellStatsCollector<BLOCK_SIZE> globalStatsCollector;
	globalStatsCollector.initThreadLocal( threadIndex );

	MoleculeStorage moleculeStorage;

	MoleculePairHandler<typeof(globalStatsCollector)> moleculePairHandler( globalStatsCollector );

	FastCellProcessor<BLOCK_SIZE, Molecule, typeof(moleculeStorage), typeof(moleculePairHandler)> cellProcessor(moleculeStorage, moleculePairHandler);

	cellProcessor.processCell( threadIndex, fromCellIndex( cellIndex ) );

	globalStatsCollector.reduceAndSave( threadIndex, cellIndex, cellIndex );
}
