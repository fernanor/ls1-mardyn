/*
 * moleculeStorage.h
 *
 *  Created on: May 28, 2011
 *      Author: andreas
 */

#ifndef MOLECULESTORAGE_H_
#define MOLECULESTORAGE_H_

#include "cudaComponent.h"

class MoleculeStorage : public CUDAComponentModule {
public:
	MoleculeStorage( const CUDA::Module &module, LinkedCells &linkedCells ) : CUDAComponentModule(module, linkedCells), _maxMoleculeStorage( 0 ) {
		_moleculePositions = module.getGlobal("moleculePositions");
		_moleculeForces = module.getGlobal("moleculeForces");
		_cellStartIndices = module.getGlobal("cellStartIndices");
	}

	void preForceCalculation() {
		uploadState();
	}

	void postForceCalculation() {
		downloadResults();
	}

protected:
	void uploadState();
	void downloadResults();

	CUDA::Global _moleculePositions, _moleculeForces, _cellStartIndices;

	CUDA::DeviceBuffer _positionBuffer, _forceBuffer, _startIndexBuffer;
};

#endif /* MOLECULESTORAGE_H_ */
