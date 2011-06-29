/*
 * moleculeStorage.h
 *
 *  Created on: May 28, 2011
 *      Author: andreas
 */

#ifndef MOLECULESTORAGE_H_
#define MOLECULESTORAGE_H_

#include "cudaComponent.h"

#include "config.h"

#include "sharedDecls.h"

class MoleculeStorage : public CUDAForceCalculationComponent {
public:
	MoleculeStorage( const CUDAComponent &component ) :
		CUDAForceCalculationComponent( component ),
		_moleculePositions( _module.getGlobal<floatType3 *>("moleculePositions") ),
		_moleculeRotations( _module.getGlobal<Matrix3x3Storage *>("moleculeRotations") ),
		_moleculeForces( _module.getGlobal<floatType3 *>("moleculeForces") ),
		_moleculeTorque( _module.getGlobal<floatType3 *>("moleculeTorque") ),

		_moleculeComponentTypes( _module.getGlobal<Molecule_ComponentType *>("moleculeComponentTypes") ),

		_cellStartIndices( _module.getGlobal<int *>("cellStartIndices") ),

		_convertQuaternionsToRotations( _module.getFunction( "convertQuaternionsToRotations" ) )
		{
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

	void compareResultsToCPURef( const std::vector<floatType3> &forces, const std::vector<floatType3> &torque );

	CUDA::Global<floatType3 *> _moleculePositions, _moleculeForces, _moleculeTorque;
	CUDA::Global<Matrix3x3Storage *> _moleculeRotations;
	CUDA::Global<Molecule_ComponentType *> _moleculeComponentTypes;

	CUDA::Global<int *> _cellStartIndices;

	CUDA::DeviceBuffer<floatType3> _positionBuffer, _forceBuffer, _torqueBuffer;
	CUDA::DeviceBuffer<Matrix3x3Storage> _rotationBuffer;
	CUDA::DeviceBuffer<Molecule_ComponentType> _componentTypeBuffer;
	CUDA::DeviceBuffer<int> _startIndexBuffer;

	CUDA::Function _convertQuaternionsToRotations;
};

#endif /* MOLECULESTORAGE_H_ */
