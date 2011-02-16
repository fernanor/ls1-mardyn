// Andreas Kirsch 2010

#include "LinkedCellsCUDA.h"
#include "molecules/Molecule.h"
#include "cutil_math.h"
#include "math.h"

double3 operator +=( double3 &a, const double3 &b ) {
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}

double3 operator -=( double3 &a, const double3 &b ) {
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
}

double3 operator -( const double3 &a, const double3 &b ) {
	return make_double3( a.x - b.x, a.y - b.y, a.z - b.z );
}

double3 operator *( const double &s, const double3 &b ) {
	return make_double3( s * b.x, s * b.y, s * b.z );
}

#define OUT

#define CUDA_TIMING

#ifdef CUDA_TIMING
class CUDATimer {
private:
	cudaEvent_t _startEvent, _endEvent;

public:
	CUDATimer() {
		CUDA_THROW_ON_ERROR( cudaEventCreate( &_startEvent ) );
		CUDA_THROW_ON_ERROR( cudaEventCreate( &_endEvent ) );
	}

	~CUDATimer() {
		CUDA_THROW_ON_ERROR( cudaEventDestroy( _startEvent ) );
		CUDA_THROW_ON_ERROR( cudaEventDestroy( _endEvent ) );
	}

	void begin() {
		CUDA_THROW_ON_ERROR( cudaEventRecord( _startEvent ) );
	}

	void end() {
		CUDA_THROW_ON_ERROR( cudaEventRecord( _endEvent ) );
	}

	float getElapsedTime() {
		CUDA_THROW_ON_ERROR( cudaEventSynchronize( _endEvent ) );

		float elapsedTime;
		CUDA_THROW_ON_ERROR( cudaEventElapsedTime( &elapsedTime, _startEvent, _endEvent ) );

		return elapsedTime;
	}

	void printElapsedTime( const char *format ) {
		printf( format, getElapsedTime() );
	}
};
#else
class CUDATimer {
public:
	void begin() {
	}

	void end() {
	}

	float getElapsedTime() {
		return 0.0f;
	}

	void printElapsedTime( const char *format ) {
	}
};
#endif

__device__ void calculateLennardJones( const CUDAPrecisionType3 distance, const CUDAPrecisionType distanceSquared, CUDAPrecisionType epsilon, CUDAPrecisionType sigmaSquared,
		OUT CUDAPrecisionType3 &force, OUT CUDAPrecisionType &potential) {
	CUDAPrecisionType invdr2 = 1.f / distanceSquared;
	CUDAPrecisionType lj6 = sigmaSquared * invdr2; lj6 = lj6 * lj6 * lj6;
	CUDAPrecisionType lj12 = lj6 * lj6;
	CUDAPrecisionType lj12m6 = lj12 - lj6;
	potential = 4.0f * epsilon * lj12m6;
	// result: force = fac * distance = fac * |distance| * normalized(distance)
	CUDAPrecisionType fac = -24.0f * epsilon * (lj12 + lj12m6) * invdr2;
	force = fac * distance;
}

__device__ int getCellIndex( int startIndex, int2 dimension, int3 gridOffsets ) {
	const int idx = blockIdx.x;
	const int3 gridIndex = make_int3(
			idx % dimension.x,
			(idx / dimension.x) % dimension.y,
			idx / dimension.x / dimension.y
		);
	const int cellIndex = startIndex + dot( gridIndex, gridOffsets );

	return cellIndex;
}

#define MEASURE_ERROR
#define USE_REF

//#define TEST_CELL_COVERAGE
#ifdef TEST_CELL_COVERAGE
#	include "LinkedCellsCUDAcellCoverage.cum"
#else
#	ifndef USE_REF
#		include "LinkedCellsCUDAfast.cum"
#	else
#		include "LinkedCellsCUDAref.cum"
#endif
#endif

void LinkedCellsCUDA::traversePairs() {
#ifdef MEASURE_ERROR
	_linkedCells.traversePairs();

	CUDAPrecisionType cpuPotential = _domain.getLocalUpot();
	CUDAPrecisionType cpuVirial = _domain.getLocalVirial();
	printf( "CPU Potential: %f CPU Virial: %f\n", cpuPotential, cpuVirial );
#endif
	LinkedCellsCUDA_Internal::DomainValues domainValues;
	_cudaInternal.calculateForces( domainValues );
	// update the domain values
	_domain.setLocalUpot( domainValues.potential );
	_domain.setLocalVirial( domainValues.virial );
}

void LinkedCellsCUDA_Internal::calculateForces( LinkedCellsCUDA_Internal::DomainValues &domainValues ) {
	manageAllocations();

	initComponentInfos();
	initCellInfosAndCopyPositions();
	prepareDeviceMemory();

	calculateAllLJFoces();

	extractResultsFromDeviceMemory();
	reducePotentialAndVirial( domainValues.potential, domainValues.virial );

	printf( "Potential: %f Virial: %f\n", domainValues.potential, domainValues.virial );
	printf( "Average Potential: %f Average Virial: %f\n", domainValues.potential / _numLJCenters, domainValues.virial / _numLJCenters );

#ifdef MEASURE_ERROR
	determineForceError();
#endif

	//updateMoleculeForces();
}

void LinkedCellsCUDA_Internal::manageAllocations()
{
	// HACK HACK HACK
	_numLJCenters = 2 * _linkedCells.getParticles().size();
	_numCells = _linkedCells.getCells().size();

	// TODO: use memalign like the old code?
	if( _numLJCenters > _maxLJCenters ) {
		_positions.resize( _numLJCenters );
		_forces.resize( _numLJCenters );
		_componentLJCenterIndices.resize( _numLJCenters );

		_maxLJCenters = _numLJCenters;
	}

	if( _numCells > _maxCells ) {
		_cellStartIndices.resize( _numCells + 1 );
		_domainValues.resize( _numCells );

		_maxCells = _numCells;
	}
}

void LinkedCellsCUDA_Internal::freeAllocations()
{
	_positions.resize( 0 );
	_forces.resize( 0 );
	_componentLJCenterIndices.resize( 0 );

	_cellStartIndices.resize( 0 );
	_domainValues.resize( 0 );

	_componentLJCenterInfos.resize( 0 );
	_componentLJCenterOffsetFromFirst.resize( 0 );
	delete[] _componentStartIndices;
}

void LinkedCellsCUDA_Internal::initComponentInfos() {
	const std::vector< Component > &components = _domain.getComponents();
	_componentStartIndices = new int[ components.size() ];

	// TODO: clean up this code..
	// initialize _numComponentLJCenters and _componentStartIndices
	_numComponentLJCenters = 0;
	for( int i = 0 ; i < components.size() ; i++ ) {
		_componentStartIndices[ i ] = _numComponentLJCenters;
		_numComponentLJCenters += components[i].numLJcenters();
	}

	// initialize _componentLJCenterOffsetFromFirst
	_componentLJCenterOffsetFromFirst.resize( _numComponentLJCenters);
	for( int i = 0 ; i < components.size() ; i++ ) {
		for( int j = 0 ; j < components[i].numLJcenters() ; j++ ) {
			_componentLJCenterOffsetFromFirst[ _componentStartIndices[i] + j ] = j;
		}
	}

	// initialize _componentLJCenterInfos
	_componentLJCenterInfos.resize( _numComponentLJCenters * _numComponentLJCenters );

	for( int indexCompA = 0 ; indexCompA < components.size() ; indexCompA++ ) {
		const Component &compA = components[ indexCompA ];
		assert( compA.numLJcenters() <= 2 );

		for( int indexCompB = 0 ; indexCompB < components.size() ; indexCompB++ ) {
			const Component &compB = components[ indexCompB ];
			assert( compB.numLJcenters() <= 2 );

			for( int indexLJCenterA = 0 ; indexLJCenterA < compA.numLJcenters() ; indexLJCenterA++ ) {
				const LJcenter &ljCenterA = compA.ljcenter(indexLJCenterA);

				for( int indexLJCenterB = 0 ; indexLJCenterB < compB.numLJcenters() ; indexLJCenterB++ ) {
					const LJcenter &ljCenterB = compB.ljcenter(indexLJCenterB);

					const int targetIndex = (_componentStartIndices[indexCompA] + indexLJCenterA) * _numComponentLJCenters +
							_componentStartIndices[indexCompB] + indexLJCenterB;
					ComponentLJCenterInfo &ljCenterInfo = _componentLJCenterInfos[ targetIndex ];

					ljCenterInfo.epsilon = sqrt( ljCenterA.eps() * ljCenterB.eps() );
					CUDAPrecisionType sigma = 0.5f * ( ljCenterA.sigma() + ljCenterB.sigma() );
					ljCenterInfo.sigmaSquared = sigma * sigma;
				}
			}
		}
	}

	_componentLJCenterOffsetFromFirst.copyToDevice();
	_componentLJCenterInfos.copyToDevice();
}

void LinkedCellsCUDA_Internal::initCellInfosAndCopyPositions()
{
	int currentIndex = 0;
	for( int i = 0 ; i < _numCells ; i++ ) {
		const Cell &cell = _linkedCells.getCells()[i];

		_cellStartIndices[i] = currentIndex;

		const std::list<Molecule*> &particles = cell.getParticlePointers();
		for( std::list<Molecule*>::const_iterator iterator = particles.begin() ; iterator != particles.end() ; iterator++ ) {
			Molecule &molecule = **iterator;

			const unsigned int numLJCenters = molecule.numLJcenters();
			if( numLJCenters > 2 ) {
				printf( "%i has more than 2 lj centers!\n", currentIndex );
			}

			const Quaternion &q = molecule.q();

			for( int ljCenterIndex = 0 ; ljCenterIndex < numLJCenters ; ljCenterIndex++ ) {
				const double *ljCenterInitialPosition = _domain.getComponents()[ molecule.componentid() ].ljcenter( ljCenterIndex ).r();
				double ljCenterRelativePosition[3];
				q.rotateinv( ljCenterInitialPosition, ljCenterRelativePosition );

				_positions[currentIndex].x = molecule.r(0) + ljCenterRelativePosition[0];
				_positions[currentIndex].y = molecule.r(1) + ljCenterRelativePosition[1];
				_positions[currentIndex].z = molecule.r(2) + ljCenterRelativePosition[2];

				_componentLJCenterIndices[currentIndex] = _componentStartIndices[ molecule.componentid() ] + ljCenterIndex;
				currentIndex++;
			}
		}
	}

	_cellStartIndices[_numCells] = currentIndex;
}

void LinkedCellsCUDA_Internal::prepareDeviceMemory()
{
	// TODO: use page-locked/mapped memory
	int3 *dimensions = (int3*) _linkedCells.getCellDimensions();
	printf( "Num LJ Centers: %i Num Cells: %i (%i x %i x %i)\n", _numLJCenters, _numCells, dimensions->x, dimensions->y, dimensions->z );

	CUDATimer copyTimer;

	copyTimer.begin();

	// copy the input data to the device
	_positions.copyToDevice();
	_componentLJCenterIndices.copyToDevice();
	_cellStartIndices.copyToDevice();

	// reset the output buffers
	_forces.zeroDevice();
	_domainValues.zeroDevice();

	copyTimer.end();
	copyTimer.printElapsedTime( "host to device copying: %f ms\n" );
}

void LinkedCellsCUDA_Internal::extractResultsFromDeviceMemory() {
	_forces.copyToHost();
	_domainValues.copyToHost();
}

void LinkedCellsCUDA_Internal::updateMoleculeForces() {
	int currentIndex = 0;
	for( int i = 0 ; i < _numCells ; i++ ) {
		const Cell &cell = _linkedCells.getCells()[i];

		const std::list<Molecule*> &particles = cell.getParticlePointers();
		for( std::list<Molecule*>::const_iterator iterator = particles.begin() ; iterator != particles.end() ; iterator++ ) {
			Molecule &molecule = **iterator;

			for( int i = 0 ; i < molecule.numLJcenters() ; i++ ) {
				molecule.Fljcenterset( i, (CUDAPrecisionType*) &_forces[currentIndex++] );
			}
		}
	}
}

static double lengthd( const CUDAPrecisionType3 &v ) {
	double lengthSquared = (double)v.x*v.x + v.y*v.y + v.z*v.z;
	return sqrt(lengthSquared);
}

void LinkedCellsCUDA_Internal::determineForceError() {
	double totalError = 0.0;
	double totalRelativeError = 0.0;
	float epsilon = 5.96e-06f;

	double avgCPUMagnitude = 0.0, avgCUDAMagnitude = 0.0;
	int currentIndex = 0;
	for( int i = 0 ; i < _numCells ; i++ ) {
		const Cell &cell = _linkedCells.getCells()[i];

		const std::list<Molecule*> &particles = cell.getParticlePointers();
		for( std::list<Molecule*>::const_iterator iterator = particles.begin() ; iterator != particles.end() ; iterator++ ) {
		    Molecule &molecule = **iterator;
			for( int ljCenter = 0 ; ljCenter < molecule.numLJcenters() ; ljCenter++ ) {
				CUDAPrecisionType3 &cudaForce = _forces[currentIndex++];

				if( !cell.isBoundaryCell() && !cell.isInnerCell() ) {
					continue;
				}

				const double *cpuForceD = molecule.ljcenter_F(ljCenter);
				CUDAPrecisionType3 cpuForce = make_float3( cpuForceD[0], cpuForceD[1], cpuForceD[2] );
				CUDAPrecisionType3 deltaForce = cudaForce - cpuForce;

				CUDAPrecisionType cpuForceLength = lengthd( cpuForce );
				CUDAPrecisionType cudaForceLength = lengthd( cudaForce );
				CUDAPrecisionType deltaForceLength = lengthd( deltaForce );

				if( isfinite(cpuForceLength) && isfinite( cudaForceLength ) && isfinite( deltaForceLength ) ) {
					avgCPUMagnitude += cpuForceLength;
					avgCUDAMagnitude += cudaForceLength;

					totalError += deltaForceLength;

					if( cpuForceLength > epsilon ) {
						double relativeError = deltaForceLength / cpuForceLength;
						totalRelativeError += relativeError;
					}
				}
				else {
					;
				}
			}
		}
	}

	avgCPUMagnitude /= currentIndex;
	avgCUDAMagnitude /= currentIndex;

	printf( "Average CPU Mag:  %f\n"
			"Average CUDA Mag: %f\n"
			"Average Error: %f\n"
			"Average Relative Error: %f\n", avgCPUMagnitude, avgCUDAMagnitude, totalError / _numLJCenters, totalRelativeError / _numLJCenters );
}

void LinkedCellsCUDA_Internal::calculateAllLJFoces() {
	CUDATimer singleCellsTimer, cellPairsTimer;

	// TODO: wtf? this is from the old code
	const float cutOffRadiusSquared = _cutOffRadius * _cutOffRadius;

	const dim3 blockSize = dim3( WARP_SIZE, NUM_WARPS, 1 );

	// intra cell forces
	const int *dimensions = _linkedCells.getCellDimensions();
	assert( dimensions[0] >= 2 && dimensions[1] >= 2 && dimensions[2] >=2 );

	const int3 zero3 = {0,0,0};
	const int3 xDirection = {1,0,0};
	const int3 yDirection = {0,1,0};
	const int3 zDirection = {0,0,1};
	// always make sure that each direction contains one component == 1
	const int3 directions[] = {
			{1,0,0},{0,1,0},{0,0,1},
			{1,1,0},{1,0,1},{0,1,1},
			{-1,1,0},{-1,0,1},{0,-1,1},
			{-1,1,1},{1,-1,1},{1,1,-1},
			{1,1,1}
	};

	cellPairsTimer.begin();

	for( int i = 0 ; i < sizeof( directions ) / sizeof( directions[0] ) ; i++ ) {
		const int3 &direction = directions[i];
		// we are going to iterate over odd and even slices (either xy-, xz- or yz-slices)

		// define: the main direction is the normal of the slice plane

		int neighborOffset = getDirectionOffset( direction );

		// contains the oriented direction as if the main direction was (0,0,1)
		int3 localDirection;
		// dimensions as if the main direction was (0,0,1)
		int3 localDimensions;
		int3 gridOffsets;

		// determine the direction of the plane (xy, xz or yz)
		if( direction.x == 1 ) {
			// yz plane (main direction: x)
			localDirection = make_int3( direction.y, direction.z, direction.x );
			localDimensions = make_int3( dimensions[1], dimensions[2], dimensions[0] );
			gridOffsets = make_int3(
					getDirectionOffset( yDirection ),
					getDirectionOffset( zDirection ),
					getDirectionOffset( xDirection )
				);
		}
		else if( direction.y == 1 ) {
			// xz plane (main direction: y)
			localDirection = make_int3( direction.x, direction.z, direction.y );
			localDimensions = make_int3( dimensions[0], dimensions[2], dimensions[1] );
			gridOffsets = make_int3(
					getDirectionOffset( xDirection ),
					getDirectionOffset( zDirection ),
					getDirectionOffset( yDirection )
				);
		}
		else if( direction.z == 1 ) {
			// xy plane (main direction: z)
			localDirection = direction;
			localDimensions = make_int3( dimensions[0], dimensions[1], dimensions[2] );
			gridOffsets = make_int3(
					getDirectionOffset( xDirection ),
					getDirectionOffset( yDirection ),
					getDirectionOffset( zDirection )
				);
		}
		else {
			assert( false );
		}

		// determine the startOffset as first cell near (0,0,0) so that start + neighborOffset won't be out of bounds
		int evenSlicesStartIndex = getCellOffset( -min( direction, zero3 ) );
		// odd slices start one slice "down"
		int oddSlicesStartIndex = evenSlicesStartIndex + gridOffsets.z;

		// set z to 0
		// adapt the local dimensions in such a way as to avoid out of bounds accesses at the "far corners"
		// the positive components of localSliceDirection affect the max corner of the slice
		// the negative ones the min corner (see *StartIndex). dimensions = max - min => use abs to count both correctly.
		localDimensions -= abs( localDirection );

		// always move 2 slices in local z direction, so we hit either odd or even slices in one kernel call
		gridOffsets.z *= 2;

		// there are floor( dimZ / 2 ) odd slices
		int numOddSlices = localDimensions.z / 2;
		int numEvenSlices = localDimensions.z - numOddSlices;

		int numCellsInSlice = localDimensions.x * localDimensions.y;

		// do all even slices
		Kernel_calculatePairLJForces<<<numEvenSlices * numCellsInSlice, blockSize>>>(
				_positions.devicePtr(), _componentLJCenterIndices.devicePtr(), _forces.devicePtr(),
				_componentLJCenterInfos.devicePtr(), _numComponentLJCenters,
				_cellStartIndices.devicePtr(), _domainValues.devicePtr(),
				evenSlicesStartIndex, make_int2( localDimensions ), gridOffsets,
				neighborOffset,
				cutOffRadiusSquared
			);

		// do all odd slices
		Kernel_calculatePairLJForces<<<numOddSlices * numCellsInSlice, blockSize>>>(
				_positions.devicePtr(), _componentLJCenterIndices.devicePtr(), _forces.devicePtr(),
				_componentLJCenterInfos.devicePtr(), _numComponentLJCenters,
				_cellStartIndices.devicePtr(), _domainValues.devicePtr(),
				oddSlicesStartIndex, make_int2( localDimensions ), gridOffsets,
				neighborOffset,
				cutOffRadiusSquared
			);
	}

	cellPairsTimer.end();

	// inner cell forces
	singleCellsTimer.begin();

	Kernel_calculateInnerLJForces<<<_numCells, blockSize>>>(
			_positions.devicePtr(), _componentLJCenterIndices.devicePtr(), _forces.devicePtr(),
			_componentLJCenterInfos.devicePtr(), _componentLJCenterOffsetFromFirst.devicePtr(), _numComponentLJCenters,
			_cellStartIndices.devicePtr(), _domainValues.devicePtr(),
			cutOffRadiusSquared
		);

	singleCellsTimer.end();

	singleCellsTimer.printElapsedTime( "intra cell LJ forces: %f ms " );
	cellPairsTimer.printElapsedTime( "inter cell LJ forces: %f ms\n" );
}

void LinkedCellsCUDA_Internal::reducePotentialAndVirial( OUT CUDAPrecisionType &potential, OUT CUDAPrecisionType &virial ) {
	const std::vector<unsigned long> &innerCellIndices = _linkedCells.getInnerCellIndices();
	const std::vector<unsigned long> &boundaryCellIndices = _linkedCells.getBoundaryCellIndices();

	potential = 0.0f;
	virial = 0.0f;
	for( int i = 0 ; i < innerCellIndices.size() ; i++ ) {
		int innerCellIndex = innerCellIndices[i];
#ifdef TEST_CELL_COVERAGE
		if( (int) _domainValues[ innerCellIndex ].x != 26 ) {
			printf( "%i (badly covered inner cell - coverage: %f)\n", innerCellIndex, _domainValues[ innerCellIndex ].x );
		}
#endif
		potential += _domainValues[ innerCellIndex ].x;
		virial += _domainValues[ innerCellIndex ].y;
	}
	for( int i = 0 ; i < boundaryCellIndices.size() ; i++ ) {
		int boundaryCellIndex = boundaryCellIndices[ i ];

#ifdef TEST_CELL_COVERAGE
		if( (int) _domainValues[ boundaryCellIndex ].x != 26 ) {
			printf( "%i (badly covered inner cell - coverage: %f)\n", boundaryCellIndex, _domainValues[ boundaryCellIndex ].x );
		}
#endif

		potential += _domainValues[ boundaryCellIndex ].x;
		virial += _domainValues[ boundaryCellIndex ].y;
	}

	// every contribution is added twice so divide by 2
	potential /= 2.0f;
	virial /= 2.0f;

	// TODO: I have no idea why the sign is different in the GPU code...
	virial = -virial;
}
