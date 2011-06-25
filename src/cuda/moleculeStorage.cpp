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
		for( std::list<Molecule*>::const_iterator iterator = particles.begin() ; iterator != particles.end() ; iterator++ ) {
			Molecule &molecule = **iterator;

			float3 position = make_float3( molecule.r(0), molecule.r(1), molecule.r(2) );
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

				float ww=quaternion.w*quaternion.w;
				float xx=quaternion.x*quaternion.x;
				float yy=quaternion.y*quaternion.y;
				float zz=quaternion.z*quaternion.z;
				float xy=quaternion.x*quaternion.y;
				float zw=quaternion.z*quaternion.w;
				float xz=quaternion.x*quaternion.z;
				float yw=quaternion.y*quaternion.w;
				float yz=quaternion.y*quaternion.z;
				float xw=quaternion.x*quaternion.w;

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

			componentTypes.push_back( molecule.componentid() );

			currentIndex++;
		}
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
#warning CPU testing quaternion matrix conversion
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

	void registerErrorFor( const float3 &cpuResult, const float3 &cudaResult ) {
		const double epsilon = 5.96e-06f;

		const double3 delta = make_double3(cpuResult) - make_double3(cudaResult);

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

void MoleculeStorage::compareResultsToCPURef( const std::vector<float3> &forces, const std::vector<float3> &torque ) {
	CPUCudaVectorErrorMeasure forceErrorMeasure( "force statistics" ), torqueErrorMeasure( "torque statistics" );

	const int numCells = _linkedCells.getCells().size();

	int currentIndex = 0;
	for( int i = 0 ; i < numCells ; i++ ) {
		const Cell &cell = _linkedCells.getCells()[i];

		const std::list<Molecule*> &particles = cell.getParticlePointers();
		for( std::list<Molecule*>::const_iterator iterator = particles.begin() ; iterator != particles.end() ; iterator++ ) {
			Molecule &molecule = **iterator;
			const float3 &cudaForce = forces[currentIndex];
			const float3 &cudaTorque = torque[currentIndex];
			currentIndex++;

			if( !cell.isBoundaryCell() && !cell.isInnerCell() ) {
				continue;
			}

			// we are going to compare F and M
			// TODO: make sure that we always overwrite F and M
			molecule.calcFM();

			const float3 cpuForce = make_float3( molecule.F(0), molecule.F(1), molecule.F(2) );
			const float3 cpuTorque = make_float3( molecule.M(0), molecule.M(1), molecule.M(2) );

			forceErrorMeasure.registerErrorFor( cpuForce, cudaForce );
			torqueErrorMeasure.registerErrorFor( cpuTorque, cudaTorque );
		}
	}

	forceErrorMeasure.report();
	torqueErrorMeasure.report();
}

void MoleculeStorage::downloadResults() {
	std::vector<float3> forces;
	std::vector<float3> torque;

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
		for( std::list<Molecule*>::const_iterator iterator = particles.begin() ; iterator != particles.end() ; iterator++ ) {
			Molecule &molecule = **iterator;

			molecule.setF((float*) &forces[currentIndex]);
			molecule.setM((float*) &torque[currentIndex]);
			currentIndex++;
		}
	}
}
