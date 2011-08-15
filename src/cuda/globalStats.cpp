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
		_potential += cellStats[ innerCellIndex ].potential;
		_virial += cellStats[ innerCellIndex ].virial;
	}
	for( int i = 0 ; i < boundaryCellIndices.size() ; i++ ) {
		int boundaryCellIndex = boundaryCellIndices[ i ];
		_potential += cellStats[ boundaryCellIndex ].potential;
		_virial += cellStats[ boundaryCellIndex ].virial;
	}

	// every contribution is added twice so divide by 2
	_potential /= 2.0f;
	_virial /= 2.0f;

	// the CPU code has moleculeAtoB = A - B (which is obviously semantically wrong) and I use B - A
	// this only affects the virial sign
	_virial = -_virial;
}
