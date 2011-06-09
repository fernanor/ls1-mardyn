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
	MoleculeStorage( const CUDA::Module &module, LinkedCells &linkedCells ) :
		CUDAComponentModule(module, linkedCells),
		_moleculePositions( module.getGlobal<float3 *>("moleculePositions") ),
		_moleculeForces( module.getGlobal<float3 *>("moleculeForces") ),
		_cellStartIndices( module.getGlobal<int *>("cellStartIndices") ) {
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

	CUDA::Global<float3 *> _moleculePositions, _moleculeForces;
	CUDA::Global<int *> _cellStartIndices;

	CUDA::DeviceBuffer<float3> _positionBuffer, _forceBuffer;
	CUDA::DeviceBuffer<int> _startIndexBuffer;
};

#endif /* MOLECULESTORAGE_H_ */
