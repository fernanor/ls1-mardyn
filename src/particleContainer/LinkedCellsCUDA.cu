// Andreas Kirsch 2010

#include "LinkedCellsCUDA.h"
#include "molecules/Molecule.h"
#include "cutil_math.h"

struct CUDAException : public std::exception {
	const cudaError_t errorCode;
	const std::string errorSource;

	CUDAException( cudaError_t errorCode, const std::string &errorSource = "" )
	: errorCode( errorCode ), errorSource( errorSource ) {}

	~CUDAException() throw() {}

    /** Returns a C-style character string describing the general cause
     *  of the current error.  */
    virtual const char* what() const throw() {
    	return errorSource.c_str();
    }
};

#define CUDA_THROW_ON_ERROR( expr ) \
	do { \
		cudaError_t errorCode = (expr); \
		if( errorCode != cudaSuccess ) { \
			throw CUDAException( errorCode, #expr ); \
		} \
	} while( 0 )

#define OUT

__device__ void calculateLennardJones( const float3 distance, const float distanceSquared, float epsilon24, float sigmaSquared,
		OUT float3 &force, OUT float &potential) {
	float invdr2 = 1.f / distanceSquared;
	float lj6 = sigmaSquared * invdr2; lj6 = lj6 * lj6 * lj6;
	float lj12 = lj6 * lj6;
	float lj12m6 = lj12 - lj6;
	potential = epsilon24 * lj12m6;
	float fac = epsilon24 * (lj12 + lj12m6) * invdr2;
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

__global__ void Kernel_calculateInnerLJForces( float3 *positions, OUT float3 *forces, int2 *cellInfos,
		float epsilon24, float sigmaSquared ) {
	const int cellIndex = blockIdx.x;

	const int cellStart = cellInfos[cellIndex].x;
	const int cellLength = cellInfos[cellIndex].y;

	// start with 1 because aIndex = 1 will result in an empty loop cycle
	for( int aIndex = 1 ; aIndex < cellLength ; aIndex++ ) {
		for( int bIndex = 0 ; bIndex < aIndex ; bIndex++ ) {
			const float3 &aPosition = positions[cellStart + aIndex];
			const float3 &bPosition = positions[cellStart + bIndex];

			const float3 distance = bPosition - aPosition;
			const float distanceSquared = dot( distance, distance );

			float potential;
			float3 force;
			calculateLennardJones( distance, distanceSquared, epsilon24, sigmaSquared, force, potential );

			forces[cellStart + aIndex] += force;
			forces[cellStart + bIndex] -= force;
		}
	}
}

__global__ void Kernel_calculatePairLJForces( float3 *positions, OUT float3 *forces, int2 *cellInfos,
		int startIndex, int2 dimension, int3 gridOffsets,
		int neighborOffset,
		float epsilon24, float sigmaSquared ) {
	const int cellIndex = getCellIndex( startIndex, dimension, gridOffsets );
	const int neighborIndex = cellIndex + neighborOffset;

	const int cellA_start = cellInfos[cellIndex].x;
	const int cellA_length = cellInfos[cellIndex].y;
	const int cellB_start = cellInfos[neighborIndex].x;
	const int cellB_length = cellInfos[neighborIndex].y;

	for( int aIndex = 0 ; aIndex < cellA_length ; aIndex++ ) {
		for( int bIndex = 0 ; bIndex < cellB_length ; bIndex++ ) {
			const float3 &aPosition = positions[cellA_start + aIndex];
			const float3 &bPosition = positions[cellB_start + bIndex];

			const float3 distance = bPosition - aPosition;
			const float distanceSquared = dot( distance, distance );

			float potential;
			float3 force;
			calculateLennardJones( distance, distanceSquared, epsilon24, sigmaSquared, force, potential );

			forces[cellA_start + aIndex] += force;
			forces[cellB_start + bIndex] -= force;
		}
	}
}

void LinkedCellsCUDA_Internal::calculateForces() {
	manageAllocations();

	initCellInfosAndCopyPositions();
	prepareDeviceMemory();

	calculateAllLJFoces();

	extractResultsFromDeviceMemory();
	updateMoleculeForces();
}

void LinkedCellsCUDA_Internal::manageAllocations()
{
	_numParticles = _linkedCells.getParticles().size();
	_numCells = _linkedCells.getCells().size();

	// TODO: use memalign like the old code?
	if( _numParticles > _maxParticles ) {
		delete[] _positions;
		delete[] _forces;

		CUDA_THROW_ON_ERROR( cudaFree( _devicePositions ) );
		CUDA_THROW_ON_ERROR( cudaFree( _deviceForces ) );

		_maxParticles = _numParticles;

		_positions = new float3[_numParticles];
		_forces = new float3[_numParticles];

		CUDA_THROW_ON_ERROR( cudaMalloc( &_devicePositions, _numParticles * sizeof(*_devicePositions) ) );
		CUDA_THROW_ON_ERROR( cudaMalloc( &_deviceForces, _numParticles * sizeof(*_deviceForces) ) );
	}

	if( _numCells > _maxCells ) {
		delete[] _cellInfos;
		CUDA_THROW_ON_ERROR( cudaFree( _deviceCellInfos ) );

		_maxCells = _numCells;

		_cellInfos = new int2[_numCells];
		CUDA_THROW_ON_ERROR( cudaMalloc( &_deviceCellInfos, _numCells * sizeof(*_deviceCellInfos) ) );
	}
}

void LinkedCellsCUDA_Internal::freeAllocations()
{
	delete[] _positions;
	delete[] _forces;

	CUDA_THROW_ON_ERROR( cudaFree( _devicePositions ) );
	CUDA_THROW_ON_ERROR( cudaFree( _deviceForces ) );

	delete[] _cellInfos;
	CUDA_THROW_ON_ERROR( cudaFree( _deviceCellInfos ) );
}

void LinkedCellsCUDA_Internal::initCellInfosAndCopyPositions()
{
	int currentIndex = 0;
	for( int i = 0 ; i < _numCells ; i++ ) {
		const Cell &cell = _linkedCells.getCells()[i];

		_cellInfos[i].x = currentIndex;
		_cellInfos[i].y = cell.getMoleculeCount();

		const std::list<Molecule*> &particles = cell.getParticlePointers();
		for( std::list<Molecule*>::const_iterator iterator = particles.begin() ; iterator != particles.end() ; iterator++ ) {
			Molecule &molecule = **iterator;
			_positions[currentIndex].x = molecule.r(0);
			_positions[currentIndex].y = molecule.r(1);
			_positions[currentIndex].z = molecule.r(2);
			currentIndex++;
		}
	}
}

void LinkedCellsCUDA_Internal::prepareDeviceMemory()
{
	// TODO: use page-locked/mapped memory
	printf( "%i\n", _numParticles );

	CUDA_THROW_ON_ERROR( cudaMemcpy( _devicePositions, _positions, _numParticles * sizeof(*_devicePositions), cudaMemcpyHostToDevice ) );
	CUDA_THROW_ON_ERROR( cudaMemcpy( _deviceCellInfos, _cellInfos, _numCells * sizeof(*_deviceCellInfos), cudaMemcpyHostToDevice ) );

	// init device forces to 0
	CUDA_THROW_ON_ERROR( cudaMemset( _deviceForces, 0, _numParticles * sizeof(*_deviceForces) ) );
}

void LinkedCellsCUDA_Internal::extractResultsFromDeviceMemory() {
	CUDA_THROW_ON_ERROR( cudaMemcpy( _forces, _deviceForces, _numParticles * sizeof(*_deviceForces), cudaMemcpyDeviceToHost ) );
}

void LinkedCellsCUDA_Internal::updateMoleculeForces() {
	int currentIndex = 0;
	for( int i = 0 ; i < _numCells ; i++ ) {
		const Cell &cell = _linkedCells.getCells()[i];

		const std::list<Molecule*> &particles = cell.getParticlePointers();
		for( std::list<Molecule*>::const_iterator iterator = particles.begin() ; iterator != particles.end() ; iterator++ ) {
			Molecule &molecule = **iterator;
			molecule.setF( (float*) &_forces[currentIndex] );
			currentIndex++;
		}
	}
}

void LinkedCellsCUDA_Internal::calculateAllLJFoces() {
	// TODO: wtf? this is from the old code
	const float epsilon24 = 24.0f;
	const float sigmaSquared = 1.0f;

	// inner forces first
	Kernel_calculateInnerLJForces<<<_numCells, 1>>>( _devicePositions, _deviceForces,_deviceCellInfos, epsilon24, sigmaSquared );

	// pair forces
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

	for( int i = 0 ; i < sizeof( directions ) / sizeof( directions[0] ) ; i++ ) {
		const int3 &direction = directions[i];
		// we are going to iterate over odd and even slices (either xy-, xz- or yz-slices)

		// define: the main direction is the normal of the slice plane

		int neighborOffset = getDirectionOffset( direction );

		// direction with main direction (axis) set to 0 (main direction will be set to 0 in the code below)
		int3 sliceDirection = direction;
		// contains the oriented direction as if the main direction was (0,0,1)
		int3 localDirection;
		// dimensions as if the main direction was (0,0,1)
		int3 localDimensions;
		int3 gridOffsets;

		// determine the direction of the plane (xy, xz or yz)
		if( direction.x == 1 ) {
			// yz plane (main direction: x)
			sliceDirection.x = 0;

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
			sliceDirection.y = 0;

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
			sliceDirection.z = 0;

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
		int evenSlicesStartIndex = getCellOffset( -min( sliceDirection, zero3 ) );
		// odd slices start one slice "down"
		int oddSlicesStartIndex = evenSlicesStartIndex + gridOffsets.z;

		// set z to 0
		const int3 localSliceDirection = make_int3( make_int2( localDirection ) );
		// adapt the local dimensions in such a way as to avoid out of bounds accesses at the "far corners"
		// the positive components of localSliceDirection affect the max corner of the slice
		// the negative ones the min corner (see *StartIndex). dimensions = max - min => use abs to count both correctly.
		localDimensions -= abs( localSliceDirection );

		// always move 2 slices in local z direction, so we hit either odd or even slices in one kernel call
		gridOffsets.z *= 2;

		// there are floor( dimZ / 2 ) odd slices
		int numOddSlices = localDimensions.z / 2;
		int numEvenSlices = localDimensions.z - numOddSlices;

		int numCellsInSlice = localDimensions.x * localDimensions.y;

		// do all even slices
		Kernel_calculatePairLJForces<<<numEvenSlices * numCellsInSlice,1>>>(
				_devicePositions, _deviceForces, _deviceCellInfos,
				evenSlicesStartIndex, make_int2( localDimensions ), gridOffsets,
				neighborOffset,
				epsilon24, sigmaSquared
			);

		// do all odd slices
		Kernel_calculatePairLJForces<<<numOddSlices * numCellsInSlice,1>>>(
				_devicePositions, _deviceForces, _deviceCellInfos,
				oddSlicesStartIndex, make_int2( localDimensions ), gridOffsets,
				neighborOffset,
				epsilon24, sigmaSquared
			);
	}
}
