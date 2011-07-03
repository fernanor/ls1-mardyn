/*
 * globalStats.cpp
 *
 *  Created on: Jun 6, 2011
 *      Author: andreas
 */

#include "globalStats.h"

void GlobalStats::preForceCalculation() {
	_cellStatsBuffer.resize( _linkedCells.getCells().size() );
	_cellStatsBuffer.zeroDevice();
	_cellStats.set( _cellStatsBuffer );
}

void GlobalStats::postForceCalculation() {
	std::vector<CellStatsStorage> cellStats;
	_cellStatsBuffer.copyToHost( cellStats );

	const std::vector<unsigned long> &innerCellIndices = _linkedCells.getInnerCellIndices();
	const std::vector<unsigned long> &boundaryCellIndices = _linkedCells.getBoundaryCellIndices();

	_potential = 0.0f;
	_virial = 0.0f;
	for( int i = 0 ; i < innerCellIndices.size() ; i++ ) {
		int innerCellIndex = innerCellIndices[i];
#ifdef TEST_CELL_COVERAGE
		if( (int) _domainValues[ innerCellIndex ].x != 26 ) {
			printf( "%i (badly covered inner cell - coverage: %f)\n", innerCellIndex, _domainValues[ innerCellIndex ].x );
		}
#endif
		_potential += cellStats[ innerCellIndex ].potential;
		_virial += cellStats[ innerCellIndex ].virial;
	}
	for( int i = 0 ; i < boundaryCellIndices.size() ; i++ ) {
		int boundaryCellIndex = boundaryCellIndices[ i ];

#ifdef TEST_CELL_COVERAGE
		if( (int) _domainValues[ boundaryCellIndex ].x != 26 ) {
			printf( "%i (badly covered inner cell - coverage: %f)\n", boundaryCellIndex, _domainValues[ boundaryCellIndex ].x );
		}
#endif

		_potential += cellStats[ boundaryCellIndex ].potential;
		_virial += cellStats[ boundaryCellIndex ].virial;
	}

	// every contribution is added twice so divide by 2
	_potential /= 2.0f;
	_virial /= 2.0f;

	// TODO: I have no idea why the sign is different in the GPU code compared to the CPU code..
	_virial = -_virial;
}

