/*
 * moleculeStorage.cpp
 *
 *  Created on: Jun 2, 2011
 *      Author: andreas
 */

#include "moleculeStorage.h"

#include "molecules/Molecule.h"
#include "cutil_math.h"

#include "math.h"

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

	_forceBuffer.resize( currentIndex );
	_forceBuffer.zeroDevice();

	_moleculePositions.set( _positionBuffer );
	_moleculeForces.set( _forceBuffer );

	_cellStartIndices.set( _startIndexBuffer );
}

void MoleculeStorage::compareResultsToCPURef( const std::vector<float3> &forces ) {
	double totalError = 0.0;
	double totalRelativeError = 0.0;
	float epsilon = 5.96e-06f;

	double avgCPUMagnitude = 0.0, avgCUDAMagnitude = 0.0;

	const int numCells = _linkedCells.getCells().size();

	int currentIndex = 0;
	for( int i = 0 ; i < numCells ; i++ ) {
		const Cell &cell = _linkedCells.getCells()[i];

		const std::list<Molecule*> &particles = cell.getParticlePointers();
		for( std::list<Molecule*>::const_iterator iterator = particles.begin() ; iterator != particles.end() ; iterator++ ) {
			const Molecule &molecule = **iterator;
			const float3 &cudaForce = forces[currentIndex++];

			if( !cell.isBoundaryCell() && !cell.isInnerCell() ) {
				continue;
			}

			const double *cpuForceD = molecule.ljcenter_F(0);
			float3 cpuForce = make_float3( cpuForceD[0], cpuForceD[1], cpuForceD[2] );
			float3 deltaForce = cudaForce - cpuForce;

			double cpuForceLength = length( cpuForce );
			double cudaForceLength = length( cudaForce );
			double deltaForceLength = length( deltaForce );

			//if( isfinite(cpuForceLength) && isfinite( cudaForceLength ) && isfinite( deltaForceLength ) ) {
				avgCPUMagnitude += cpuForceLength;
				avgCUDAMagnitude += cudaForceLength;

				totalError += deltaForceLength;

				if( cpuForceLength > epsilon ) {
					double relativeError = deltaForceLength / cpuForceLength;
					totalRelativeError += relativeError;
				}
			/*}
			else {
				;
			}*/
		}
	}

	avgCPUMagnitude /= currentIndex;
	avgCUDAMagnitude /= currentIndex;

	printf( "Average CPU Mag:  %f\n"
			"Average CUDA Mag: %f\n"
			"Average Error: %f\n"
			"Average Relative Error: %f\n", avgCPUMagnitude, avgCUDAMagnitude, totalError / currentIndex, totalRelativeError / currentIndex );
}

void MoleculeStorage::downloadResults() {
	std::vector<float3> forces;

	_forceBuffer.copyToHost( forces );

	compareResultsToCPURef( forces );

	const int numCells = _linkedCells.getCells().size();

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
