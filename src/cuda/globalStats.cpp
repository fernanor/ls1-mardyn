/*
 * globalStats.cpp
 *
 *  Created on: Jun 6, 2011
 *      Author: andreas
 */

#include "globalStats.h"

virtual void GlobalStats::preForceCalculation() {
	_cellStatsBuffer.resizeElements<CellStats>( _linkedCells.getCells().size() );
	_cellStatsBuffer.zeroDevice();
}

virtual void GlobalStats::postForceCalculation() {
	std::vector<CellStats> cellStats;
	_cellStatsBuffer.copyElementsToHost( cellStats );

	const std::vector<unsigned long> &innerCellIndices = _linkedCells.getInnerCellIndices();
	const std::vector<unsigned long> &boundaryCellIndices = _linkedCells.getBoundaryCellIndices();

	potential = 0.0f;
	virial = 0.0f;
	for( int i = 0 ; i < innerCellIndices.size() ; i++ ) {
		int innerCellIndex = innerCellIndices[i];
#ifdef TEST_CELL_COVERAGE
		if( (int) _domainValues[ innerCellIndex ].x != 26 ) {
			printf( "%i (badly covered inner cell - coverage: %f)\n", innerCellIndex, _domainValues[ innerCellIndex ].x );
		}
#endif
		potential += cellStats[ innerCellIndex ].potential;
		virial += cellStats[ innerCellIndex ].virial;
	}
	for( int i = 0 ; i < boundaryCellIndices.size() ; i++ ) {
		int boundaryCellIndex = boundaryCellIndices[ i ];

#ifdef TEST_CELL_COVERAGE
		if( (int) _domainValues[ boundaryCellIndex ].x != 26 ) {
			printf( "%i (badly covered inner cell - coverage: %f)\n", boundaryCellIndex, _domainValues[ boundaryCellIndex ].x );
		}
#endif

		potential += cellStats[ boundaryCellIndex ].potential;
		virial += cellStats[ boundaryCellIndex ].virial;
	}

	// every contribution is added twice so divide by 2
	potential /= 2.0f;
	virial /= 2.0f;

	// TODO: I have no idea why the sign is different in the GPU code...
	virial = -virial;
}

