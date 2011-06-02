/*
 * moleculeStorage.h
 *
 *  Created on: May 28, 2011
 *      Author: andreas
 */

#ifndef MOLECULESTORAGE_H_
#define MOLECULESTORAGE_H_

#include "cudaComponent.h"

class MoleculeStorage : public CUDAComponent {
public:
	MoleculeStorage( const CUDA::Module &module, LinkedCells &linkedCells ) : CUDAComponent(module, linkedCells), _maxMoleculeStorage( 0 ) {
		_moleculePositions = module.getGlobal("moleculePositions");
		_moleculeForces = module.getGlobal("moleculeForces");
		_cellStartIndices = module.getGlobal("cellStartIndices");
	}

	void uploadState();
	void downloadResults();

protected:
	CUDA::Global _moleculePositions, _moleculeForces, _cellStartIndices;

	CUDA::DeviceBuffer _positionBuffer, _forceBuffer, _startIndexBuffer;
};

#endif /* MOLECULESTORAGE_H_ */
