/*
 * moleculeStorage.cpp
 *
 *  Created on: Jun 2, 2011
 *      Author: andreas
 */

#include "moleculeStorage.h"

#include "molecules/Molecule.h"
#include "cutil_math.h"

void MoleculeStorage::uploadState() {
	std::vector<float3> positions;
	std::vector<int> startIndices;

	int numCells = _linkedCells.getCells().size();

	int currentIndex = 0;
	for( int i = 0 ; i < numCells ; i++ ) {
		const Cell &cell = _linkedCells.getCells()[i];

		startIndices.push_back( currentIndex );

		const std::list<Molecule*> &particles = cell.getParticlePointers();
		for( std::list<Molecule*>::const_iterator iterator = particles.begin() ; iterator != particles.end() ; iterator++ ) {
			Molecule &molecule = **iterator;

			float3 position = make_float3( molecule.r(0), molecule.r(1), molecule.r(2) );
			positions.push_back( position );
			currentIndex++;
		}
	}

	startIndices.push_back( currentIndex );

	_startIndexBuffer.copyToDevice( startIndices );
	_positionBuffer.copyToDevice( positions );

	_forceBuffer.resize( positions.size() );
	_forceBuffer.zeroDevice();

	_moleculePositions.set( _positionBuffer.devicePtr() );
	_moleculeForces.set( _forceBuffer.devicePtr() );
	_cellStartIndices.set( _startIndexBuffer.devicePtr() );
}

void MoleculeStorage::downloadResults() {
	std::vector<float3> forces;

	_forceBuffer.copyToHost( forces );

	int numCells = _linkedCells.getCells().size();

	int currentIndex = 0;
	for( int i = 0 ; i < numCells ; i++ ) {
		const Cell &cell = _linkedCells.getCells()[i];

		const std::list<Molecule*> &particles = cell.getParticlePointers();
		for( std::list<Molecule*>::const_iterator iterator = particles.begin() ; iterator != particles.end() ; iterator++ ) {
			Molecule &molecule = **iterator;

			molecule.Fljcenterset( 0, (float*) &forces[currentIndex++] );
		}
	}
}
