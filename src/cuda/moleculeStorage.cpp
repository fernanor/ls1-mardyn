/*
 * moleculeStorage.cpp
 *
 *  Created on: Jun 2, 2011
 *      Author: andreas
 */

#include "config.h"

#include "Domain.h"

#include "moleculeStorage.h"

#include "molecules/Molecule.h"

#include "math.h"

void MoleculeStorage::uploadState() {
	std::vector<floatType3> positions;
	std::vector<QuaternionStorage> quaternions;
	std::vector<Molecule_ComponentType> componentTypes;
#ifdef TEST_QUATERNION_MATRIX_CONVERSION
	std::vector<Matrix3x3Storage> rotations;
#endif

	std::vector<int> startIndices;

	int numCells = _linkedCells.getCells().size();

	int currentIndex = 0;
	for( int i = 0 ; i < numCells ; i++ ) {
		const Cell &cell = _linkedCells.getCells()[i];

		startIndices.push_back( currentIndex );

		const std::list<Molecule*> &particles = cell.getParticlePointers();
#ifdef CUDA_SORT_CELLS_BY_COMPONENTTYPE
		for( int componentType = 0 ; componentType < _domain.getComponents().size() ; componentType++ ) {
#endif
			for( std::list<Molecule*>::const_iterator iterator = particles.begin() ; iterator != particles.end() ; iterator++ ) {
				Molecule &molecule = **iterator;

#ifdef CUDA_SORT_CELLS_BY_COMPONENTTYPE
				if( molecule.componentid() != componentType ) {
					continue;
				}
#endif

				componentTypes.push_back( molecule.componentid() );

				floatType3 position = make_floatType3( molecule.r(0), molecule.r(1), molecule.r(2) );
				positions.push_back( position );

				const Quaternion &dQuaternion = molecule.q();
				QuaternionStorage quaternion;
				quaternion.w = dQuaternion.qw();
				quaternion.x = dQuaternion.qx();
				quaternion.y = dQuaternion.qy();
				quaternion.z = dQuaternion.qz();
				quaternions.push_back( quaternion );

#ifdef TEST_QUATERNION_MATRIX_CONVERSION
				{
					Matrix3x3Storage rot;

					const floatType ww=quaternion.w*quaternion.w;
					const floatType xx=quaternion.x*quaternion.x;
					const floatType yy=quaternion.y*quaternion.y;
					const floatType zz=quaternion.z*quaternion.z;
					const floatType xy=quaternion.x*quaternion.y;
					const floatType zw=quaternion.z*quaternion.w;
					const floatType xz=quaternion.x*quaternion.z;
					const floatType yw=quaternion.y*quaternion.w;
					const floatType yz=quaternion.y*quaternion.z;
					const floatType xw=quaternion.x*quaternion.w;

					rot.rows[0].x=ww+xx-yy-zz;
					rot.rows[0].y=2*(xy-zw);
					rot.rows[0].z=2*(xz+yw);

					rot.rows[1].x=2*(xy+zw);
					rot.rows[1].y=ww-xx+yy-zz;
					rot.rows[1].z=2*(yz-xw);

					rot.rows[2].x=2*(xz-yw);
					rot.rows[2].y=2*(yz+xw);
					rot.rows[2].z=ww-xx-yy+zz;

					rotations.push_back(rot);
				}
#endif

				currentIndex++;
			}
#ifdef CUDA_SORT_CELLS_BY_COMPONENTTYPE
		}
#endif
	}

	startIndices.push_back( currentIndex );

	_startIndexBuffer.copyToDevice( startIndices );

	_positionBuffer.copyToDevice( positions );
	_componentTypeBuffer.copyToDevice( componentTypes );

	_forceBuffer.resize( currentIndex );
	_forceBuffer.zeroDevice();

	_torqueBuffer.resize( currentIndex );
	_torqueBuffer.zeroDevice();

#ifndef TEST_QUATERNION_MATRIX_CONVERSION
	_rotationBuffer.resize( currentIndex );
#else
#	warning CPU testing quaternion matrix conversion
	_rotationBuffer.copyToDevice( rotations );
#endif

	_moleculePositions.set( _positionBuffer );
	_moleculeRotations.set( _rotationBuffer );
	_moleculeForces.set( _forceBuffer );
	_moleculeTorque.set( _torqueBuffer );
	_moleculeComponentTypes.set( _componentTypeBuffer );

	_cellStartIndices.set( _startIndexBuffer );

	CUDA::DeviceBuffer<QuaternionStorage> quaternionBuffer;
	quaternionBuffer.copyToDevice( quaternions );

	_convertQuaternionsToRotations.call().
			parameter(quaternionBuffer.devicePtr()).
			parameter(currentIndex).
			setBlockShape(MAX_BLOCK_SIZE, 1, 1).
			execute(currentIndex / MAX_BLOCK_SIZE + 1, 1);
}

struct CPUCudaVectorErrorMeasure {
	double totalCPUMagnitude, totalCudaMagnitude;
	double totalError, totalRelativeError;
	int numDataPoints;

	const char *name;

	CPUCudaVectorErrorMeasure(const char *name)
	: name(name), totalCPUMagnitude(0.0f), totalCudaMagnitude(0.0f), totalError(0.0f), totalRelativeError(0.0f), numDataPoints(0) {
	}

	void registerErrorFor( const floatType3 &cpuResult, const floatType3 &cudaResult ) {
		const double epsilon = 5.96e-06f;

		// TODO: add convert_double3 macro/define!
#ifndef CUDA_DOUBLE_MODE
		const double3 delta = make_double3(cpuResult) - make_double3(cudaResult);
#else
		const double3 delta = cpuResult - cudaResult;
#endif

		const double cpuLength = length(cpuResult);
		const double cudaLength = length(cudaResult);
		const double deltaLength = length(delta);

		totalCPUMagnitude += cpuLength;
		totalCudaMagnitude += cudaLength;

		totalError += deltaLength;
		if( cpuLength > epsilon ) {
			totalRelativeError += deltaLength / cpuLength;
		}
		numDataPoints++;
	}

	void report() {
		printf( "%s:\n"
				"  average CPU: %f\n"
				"  average CUDA: %f\n"
				"\n"
				"  average error: %f; average relative error: %f\n",
				name,
				totalCPUMagnitude / numDataPoints, totalCudaMagnitude / numDataPoints,
				totalError / numDataPoints, totalRelativeError / numDataPoints
			);
	}
};

void MoleculeStorage::compareResultsToCPURef( const std::vector<floatType3> &forces, const std::vector<floatType3> &torque ) {
	CPUCudaVectorErrorMeasure forceErrorMeasure( "force statistics" ), torqueErrorMeasure( "torque statistics" );

	const int numCells = _linkedCells.getCells().size();

	int currentIndex = 0;
	for( int i = 0 ; i < numCells ; i++ ) {
		const Cell &cell = _linkedCells.getCells()[i];

		const std::list<Molecule*> &particles = cell.getParticlePointers();

#ifdef CUDA_SORT_CELLS_BY_COMPONENTTYPE
		for( int componentType = 0 ; componentType < _domain.getComponents().size() ; componentType++ ) {
#endif
			for( std::list<Molecule*>::const_iterator iterator = particles.begin() ; iterator != particles.end() ; iterator++ ) {
				Molecule &molecule = **iterator;

#ifdef CUDA_SORT_CELLS_BY_COMPONENTTYPE
				if( molecule.componentid() != componentType ) {
					continue;
				}
#endif

				const floatType3 &cudaForce = forces[currentIndex];
				const floatType3 &cudaTorque = torque[currentIndex];
				currentIndex++;

				if( !cell.isBoundaryCell() && !cell.isInnerCell() ) {
					continue;
				}

				// we are going to compare F and M, so combine the sites
				molecule.calcFM();

				const floatType3 cpuForce = make_floatType3( molecule.F(0), molecule.F(1), molecule.F(2) );
				const floatType3 cpuTorque = make_floatType3( molecule.M(0), molecule.M(1), molecule.M(2) );

				forceErrorMeasure.registerErrorFor( cpuForce, cudaForce );
				torqueErrorMeasure.registerErrorFor( cpuTorque, cudaTorque );

				// clear the molecule after comparing the values to make sure that only the GPU values are applied
				molecule.clearFM();
			}
#ifdef CUDA_SORT_CELLS_BY_COMPONENTTYPE
		}
#endif
	}

	forceErrorMeasure.report();
	torqueErrorMeasure.report();
}

void MoleculeStorage::downloadResults() {
	std::vector<floatType3> forces;
	std::vector<floatType3> torque;

	_forceBuffer.copyToHost( forces );
	_torqueBuffer.copyToHost( torque );

#ifdef COMPARE_TO_CPU
	compareResultsToCPURef( forces, torque );
#endif

	const int numCells = _linkedCells.getCells().size();

	int currentIndex = 0;
	for( int i = 0 ; i < numCells ; i++ ) {
		const Cell &cell = _linkedCells.getCells()[i];

		const std::list<Molecule*> &particles = cell.getParticlePointers();
#ifdef CUDA_SORT_CELLS_BY_COMPONENTTYPE
		for( int componentType = 0 ; componentType < _domain.getComponents().size() ; componentType++ ) {
#endif
			for( std::list<Molecule*>::const_iterator iterator = particles.begin() ; iterator != particles.end() ; iterator++ ) {
				Molecule &molecule = **iterator;

#ifdef CUDA_SORT_CELLS_BY_COMPONENTTYPE
				if( molecule.componentid() != componentType ) {
					continue;
				}
#endif

				molecule.setF((floatType*) &forces[currentIndex]);
				molecule.setM((floatType*) &torque[currentIndex]);
				currentIndex++;
			}
#ifdef CUDA_SORT_CELLS_BY_COMPONENTTYPE
		}
#endif
	}
}
