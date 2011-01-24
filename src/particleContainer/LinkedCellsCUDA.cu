// Andreas Kirsch 2010

#include "LinkedCellsCUDA.h"
#include "molecules/Molecule.h"
#include "cutil_math.h"

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

__device__ void calculateLennardJones( const float3 distance, const float distanceSquared, float epsilon, float sigmaSquared,
		OUT float3 &force, OUT float &potential) {
	float invdr2 = 1.f / distanceSquared;
	float lj6 = sigmaSquared * invdr2; lj6 = lj6 * lj6 * lj6;
	float lj12 = lj6 * lj6;
	float lj12m6 = lj12 - lj6;
	potential = 4.0f * epsilon * lj12m6;
	// result: force = fac * distance = fac * |distance| * normalized(distance)
	float fac = -24.0f * epsilon * (lj12 + lj12m6) * invdr2;
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


#define MAX_BLOCK_SIZE 512
#define WARP_SIZE 1
#define NUM_WARPS 1
#define BLOCK_SIZE (WARP_SIZE*NUM_WARPS)
#define TRANSFER_SIZE BLOCK_SIZE
// threadIdx.xy = intraWarpIndex | warpIndex

// = ceil( a / b )
__device__ inline int iceil(int a, int b) {
	return (a+b-1) / b;
}

// = b if a % b = 0, a % b otherwise
__device__ inline int shiftedMod( int a, int b ) {
	int r = a % b;
	return (r > 0) ? r : b;
}

__device__ inline void reducePotentialAndVirial( float *potential, float *virial ) {
	// ASSERT: BLOCK_SIZE is power of 2
	for( int power = 2 ; power <= BLOCK_SIZE ; power <<= 1 ) {
		__syncthreads();

		const int index = WARP_SIZE * threadIdx.y + threadIdx.x;
		if( (index & (power-1)) == 0 ) {
			const int neighborIndex = index + (power >> 1);

			potential[index] += potential[neighborIndex];
			virial[index] += virial[neighborIndex];
		}
	}
}

struct InteractionParameters {
	float cutOffRadiusSquared;
	float epsilon;
	float sigmaSquared;
};

__device__ inline void calculateInteraction(
		const InteractionParameters &parameters,
		const float3 &positionA, const float3 &positionB,
		float3 &forceA, float3 &forceB,
		float &totalPotential, float &totalVirial
	) {
	const float3 distance = positionB - positionA;
	const float distanceSquared = dot( distance, distance );

	if( distanceSquared > parameters.cutOffRadiusSquared ) {
		return;
	}

	float3 force;
	float potential;
	calculateLennardJones( distance, distanceSquared, parameters.epsilon, parameters.sigmaSquared, force, potential );

	totalPotential += potential;
	float virial = dot( force, distance );
	totalVirial += virial;

	forceA += force;
	forceB -= force;
}

// if we're processing pairs from the same warp, we only have to process half the pairs
template< bool intraWarp >
__device__ inline void processWarp(
		const int numBCellsInWarp,
		const InteractionParameters &interactionParameters,
		const float3 &positionA, const float3 *positionsB,
		float3 &forceA, float3 *forcesB,
		float &totalPotential, float &totalVirial
		) {
	for( int cellShiftIndex = 0 ; cellShiftIndex < numBCellsInWarp ; cellShiftIndex++ ) {
		const int bThreadIndex = (threadIdx.x + cellShiftIndex) % numBCellsInWarp;
		// if we're not inside the same warp, process all WARP_SIZE * WARP_SIZE pairs inside a warp
		// otherwise only process the lower half
		if( !intraWarp || bThreadIndex < threadIdx.x ) {
			calculateInteraction( interactionParameters,
					positionA, positionsB[bThreadIndex],
					forceA, forcesB[bThreadIndex],
					totalPotential, totalVirial
				  );
		}
	}
}

template< bool intraTransferBlock >
__device__ inline void processWarps(
		bool validACell,
		const int numBCellsInTransferBlock,
		const InteractionParameters &interactionParameters,
		const float3 &positionA, const float3 (*positionsB)[WARP_SIZE],
		float3 &forceA, float3 (*forcesB)[WARP_SIZE],
		float &totalPotential, float &totalVirial
		) {
	const int numBWarps = iceil( numBCellsInTransferBlock, WARP_SIZE );
	for( int warpShiftIndex = 0 ; warpShiftIndex < numBWarps ; warpShiftIndex++ ) {
		const int bWarpIndex = (threadIdx.y + warpShiftIndex) % numBWarps;

		if( validACell ) {
			const int numBCellsInWarp = (bWarpIndex < numBWarps - 1) ? WARP_SIZE : shiftedMod(numBCellsInTransferBlock, WARP_SIZE);
			// if we're not inside the same warp, process all NUM_WARPS * NUM_WARPS pairs inside a transfer block
			// otherwise only process the 'semi'lower half
			if( !intraTransferBlock || bWarpIndex < threadIdx.y ) {
				processWarp<false>( numBCellsInWarp, interactionParameters,
						positionA, positionsB[bWarpIndex],
						forceA, forcesB[bWarpIndex],
						totalPotential, totalVirial
					);
			}
			else if( bWarpIndex == threadIdx.y ) {
				processWarp<true>( numBCellsInWarp, interactionParameters,
						positionA, positionsB[bWarpIndex],
						forceA, forcesB[bWarpIndex],
						totalPotential, totalVirial
					);
			}
		}

		// sync all warps, so that no two different warps will try to access the same warp data block
		__syncthreads();
	}
}

template< bool intraTransferBlock >
__device__ inline void processTransferBlock(
		bool validACell,
		const int numBCellsInTransferBlock,
		const InteractionParameters &interactionParameters,
		const float3 &positionA, float3 *positionsB,
		float3 &forceA, float3 *forcesB,
		float &totalPotential, float &totalVirial
		) {
	__shared__ float3 cachedBForces[NUM_WARPS][WARP_SIZE];
	__shared__ float3 cachedBPositions[NUM_WARPS][WARP_SIZE];

	bool validBCell = false;
	const int bTransferCellIndex = WARP_SIZE * threadIdx.y + threadIdx.x;
	// load B data into cache
	if( bTransferCellIndex < numBCellsInTransferBlock ) {
		validBCell = true;
		cachedBForces[threadIdx.y][threadIdx.x] = forcesB[bTransferCellIndex];
		cachedBPositions[threadIdx.y][threadIdx.x] = positionsB[bTransferCellIndex];
	}

	// I'm working on WARP_SIZE many data entries at once during processing, so there should be a natural synchronization?
	// TODO: remove this __syncthreads maybe?
	__syncthreads();

	// process block
	processWarps<intraTransferBlock>(
			validACell,
			numBCellsInTransferBlock,
			interactionParameters,
			positionA, cachedBPositions,
			forceA, cachedBForces,
			totalPotential, totalVirial
	   );

	// push B data back
	if( validBCell ) {
		forcesB[bTransferCellIndex] = cachedBForces[threadIdx.y][threadIdx.x];
	}
}

__global__ void Kernel_calculatePairLJForces( float3 *positions, OUT float3 *forces, int2 *cellInfos, OUT float2 *domainValues,
		int startIndex, int2 dimension, int3 gridOffsets,
		int neighborOffset,
		float epsilon, float sigmaSquared, float cutOffRadiusSquared ) {
	InteractionParameters parameters;
	parameters.cutOffRadiusSquared = cutOffRadiusSquared;
	parameters.epsilon = epsilon;
	parameters.sigmaSquared = sigmaSquared;

	int cellIndex = getCellIndex( startIndex, dimension, gridOffsets );
	int neighborIndex = cellIndex + neighborOffset;

	// ensure that cellA_length <= cellB_length (which will use fewer data transfers)
	// (numTransfersA + numTransfersA * numTransfersB) * TRANSFER_SIZE
	if( cellInfos[cellIndex].y > cellInfos[neighborIndex].y ) {
		// swap cellIndex and neighborIndex
		cellIndex = neighborIndex;
		neighborIndex -= neighborOffset;
	}

	const int cellA_start = cellInfos[cellIndex].x;
	const int cellA_length = cellInfos[cellIndex].y;
	const int cellB_start = cellInfos[neighborIndex].x;
	const int cellB_length = cellInfos[neighborIndex].y;

	__shared__ float totalThreadPotential[NUM_WARPS][WARP_SIZE];
	__shared__ float totalThreadVirial[NUM_WARPS][WARP_SIZE];

	totalThreadPotential[threadIdx.y][threadIdx.x] = 0.0f;
	totalThreadVirial[threadIdx.y][threadIdx.x] = 0.0f;

	const int numTransfersA = iceil( cellA_length, TRANSFER_SIZE );
	for( int transferIndexA = 0 ; transferIndexA < numTransfersA ; transferIndexA++ ) {
		bool validACell = false;
		float3 cachedAForce;
		float3 cachedAPosition;

		// load A data into (register) cache
		const int aTransferCellIndex = cellA_start + transferIndexA * TRANSFER_SIZE + WARP_SIZE * threadIdx.y + threadIdx.x;
		if( aTransferCellIndex < cellA_length ) {
			validACell = true;
			cachedAForce = forces[aTransferCellIndex];
			cachedAPosition = positions[aTransferCellIndex];
		}

		const int numTransfersB = iceil( cellB_length, TRANSFER_SIZE );
		for( int transferIndexB = 0 ; transferIndexB < numTransfersB ; transferIndexB++ ) {
			const int numBCellsInTransferBlock = (transferIndexB < numTransfersB - 1) ? TRANSFER_SIZE : shiftedMod( cellB_length, TRANSFER_SIZE );
			const int bBlockStartIndex = cellB_start + transferIndexB * TRANSFER_SIZE;

			processTransferBlock<false>(
					validACell,
					numBCellsInTransferBlock,
					parameters,
					cachedAPosition, positions + bBlockStartIndex,
					cachedAForce, forces + bBlockStartIndex,
					totalThreadPotential[threadIdx.y][threadIdx.x], totalThreadVirial[threadIdx.y][threadIdx.x]
				);
		}

		// push A data back
		if( validACell ) {
			forces[aTransferCellIndex] = cachedAForce;
		}
	}

	// reduce the potential and the virial
	// ASSERT: BLOCK_SIZE is power of 2
	reducePotentialAndVirial( (float*) totalThreadPotential, (float*) totalThreadVirial );

	if( threadIdx.x == 0 && threadIdx.y == 0 ) {
		domainValues[cellIndex].x += totalThreadPotential[0][0];
		domainValues[cellIndex].y += totalThreadVirial[0][0];
		domainValues[neighborIndex].x += totalThreadPotential[0][0];
		domainValues[neighborIndex].y += totalThreadVirial[0][0];
	}
}

__global__ void Kernel_calculateInnerLJForces( float3 *positions, OUT float3 *forces, int2 *cellInfos, OUT float2 *domainValues,
		float epsilon, float sigmaSquared, float cutOffRadiusSquared ) {
	InteractionParameters parameters;
	parameters.cutOffRadiusSquared = cutOffRadiusSquared;
	parameters.epsilon = epsilon;
	parameters.sigmaSquared = sigmaSquared;

	const int cellIndex = blockIdx.x;

	const int cellStart = cellInfos[cellIndex].x;
	const int cellLength = cellInfos[cellIndex].y;

	__shared__ float totalThreadPotential[NUM_WARPS][WARP_SIZE];
	__shared__ float totalThreadVirial[NUM_WARPS][WARP_SIZE];

	totalThreadPotential[threadIdx.y][threadIdx.x] = 0.0f;
	totalThreadVirial[threadIdx.y][threadIdx.x] = 0.0f;

	const int numTransfers = iceil( cellLength, TRANSFER_SIZE );
	for( int transferIndexA = 0 ; transferIndexA < numTransfers ; transferIndexA++ ) {
		bool validACell = false;
		float3 cachedAForce;
		float3 cachedAPosition;

		// load A data into (register) cache
		const int aBlockStartIndex = cellStart + transferIndexA * TRANSFER_SIZE;
		const int aTransferCellIndex = aBlockStartIndex + WARP_SIZE * threadIdx.y + threadIdx.x;
		if( aTransferCellIndex < cellLength ) {
			validACell = true;
			cachedAForce = make_float3( 0.0f );
			cachedAPosition = positions[aTransferCellIndex];
		}

		// process the cells inside the transferblock itself
		const int numACellsInTransferBlock = (transferIndexA < numTransfers - 1) ? TRANSFER_SIZE : shiftedMod( cellLength, TRANSFER_SIZE );
		processTransferBlock<true>(
				validACell,
				numACellsInTransferBlock,
				parameters,
				cachedAPosition, positions + aBlockStartIndex,
				cachedAForce, forces + aBlockStartIndex,
				totalThreadPotential[threadIdx.y][threadIdx.x], totalThreadVirial[threadIdx.y][threadIdx.x]
			);

		for( int transferIndexB = 1 ; transferIndexB < transferIndexA ; transferIndexB++ ) {
			const int numBCellsInTransferBlock = (transferIndexB < numTransfers - 1) ? TRANSFER_SIZE : shiftedMod( cellLength, TRANSFER_SIZE );
			const int bBlockStartIndex = cellStart + transferIndexB * TRANSFER_SIZE;

			processTransferBlock<false>(
					validACell,
					numBCellsInTransferBlock,
					parameters,
					cachedAPosition, positions + bBlockStartIndex,
					cachedAForce, forces + bBlockStartIndex,
					totalThreadPotential[threadIdx.y][threadIdx.x], totalThreadVirial[threadIdx.y][threadIdx.x]
				);
		}

		// push A data back
		if( validACell ) {
			forces[aTransferCellIndex] += cachedAForce;
		}
	}

	/// reduce the potential and the virial
	// ASSERT: BLOCK_SIZE is power of 2
	reducePotentialAndVirial( (float*) totalThreadPotential, (float*) totalThreadVirial );

	if( threadIdx.x == 0 && threadIdx.y == 0 ) {
		domainValues[cellIndex].x = totalThreadPotential[0][0] * 2;
		domainValues[cellIndex].y = totalThreadVirial[0][0] * 2;
	}
}

LinkedCellsCUDA_Internal::DomainValues LinkedCellsCUDA_Internal::calculateForces() {
	manageAllocations();

	initCellInfosAndCopyPositions();
	prepareDeviceMemory();

	calculateAllLJFoces();

	DomainValues domainValues;
	extractResultsFromDeviceMemory();
	reducePotentialAndVirial( domainValues.potential, domainValues.virial );

	printf( "Potential: %f Virial: %f\n", domainValues.potential, domainValues.virial );
	printf( "Average Potential: %f Average Virial: %f\n", domainValues.potential / _numParticles, domainValues.virial / _numParticles );

	determineForceError();

	updateMoleculeForces();

	return domainValues;
}

void LinkedCellsCUDA_Internal::manageAllocations()
{
	_numParticles = _linkedCells.getParticles().size();
	_numCells = _linkedCells.getCells().size();

	// TODO: use memalign like the old code?
	if( _numParticles > _maxParticles ) {
		_positions.resize( _numParticles );
		_forces.resize( _numParticles );

		_maxParticles = _numParticles;
	}

	if( _numCells > _maxCells ) {
		_cellInfos.resize( _numCells );
		_domainValues.resize( _numCells );

		_maxCells = _numCells;
	}
}

void LinkedCellsCUDA_Internal::freeAllocations()
{
	_positions.resize( 0 );
	_forces.resize( 0 );

	_cellInfos.resize( 0 );
	_domainValues.resize( 0 );
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

	CUDATimer copyTimer;

	copyTimer.begin();

	_positions.copyToDevice();
	_cellInfos.copyToDevice();

	// init device forces to 0
	_forces.zeroDevice();
	// not needed: _domainValues.zeroDevice();

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
			molecule.Fljcenterset( 0, (float*) &_forces[currentIndex] );
			currentIndex++;
		}
	}
}

void LinkedCellsCUDA_Internal::determineForceError() {
	double totalError = 0.0;
	double totalRelativeError = 0.0;
	float epsilon = 5.96e-06f;

	float avgCPUMagnitude = 0.0, avgCUDAMagnitude = 0.0;
	int currentIndex = 0;
	for( int i = 0 ; i < _numCells ; i++ ) {
		const Cell &cell = _linkedCells.getCells()[i];

		const std::list<Molecule*> &particles = cell.getParticlePointers();
		for( std::list<Molecule*>::const_iterator iterator = particles.begin() ; iterator != particles.end() ; iterator++ ) {
			Molecule &molecule = **iterator;
			float3 &cudaForce = _forces[currentIndex];
			const double *cpuForceD = molecule.ljcenter_F(0);
			float3 cpuForce = make_float3( cpuForceD[0], cpuForceD[1], cpuForceD[2] );
			float3 deltaForce = cudaForce - cpuForce;

			avgCPUMagnitude += length( cpuForce );
			avgCUDAMagnitude += length( cudaForce );

			float error = length( deltaForce );
			totalError += error;

			if( error > epsilon ) {
				float relativeError = error / length( cpuForce );
				totalRelativeError += relativeError;
			}

			currentIndex++;
		}
	}

	avgCPUMagnitude /= _numParticles;
	avgCUDAMagnitude /= _numParticles;

	printf( "Average CPU Mag:  %f\n"
			"Average CUDA Mag: %f\n"
			"Average Error: %f\n"
			"Average Relative Error: %f\n", avgCPUMagnitude, avgCUDAMagnitude, totalError / _numParticles, totalRelativeError / _numParticles );
}

void LinkedCellsCUDA_Internal::calculateAllLJFoces() {
	CUDATimer singleCellsTimer, cellPairsTimer;

	// TODO: wtf? this is from the old code
	const float epsilon = 1.0f;
	const float sigmaSquared = 1.0f;
	const float cutOffRadiusSquared = _cutOffRadius * _cutOffRadius;

	const dim3 blockSize = dim3( WARP_SIZE, NUM_WARPS, 1 );

	singleCellsTimer.begin();

	// inner forces first
	Kernel_calculateInnerLJForces<<<_numCells, blockSize>>>(
			_positions.devicePtr(), _forces.devicePtr(),_cellInfos.devicePtr(), _domainValues.devicePtr(),
			epsilon, sigmaSquared, cutOffRadiusSquared
		);

	singleCellsTimer.end();

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
				_positions.devicePtr(), _forces.devicePtr(),_cellInfos.devicePtr(), _domainValues.devicePtr(),
				evenSlicesStartIndex, make_int2( localDimensions ), gridOffsets,
				neighborOffset,
				epsilon, sigmaSquared, cutOffRadiusSquared
			);

		// do all odd slices
		Kernel_calculatePairLJForces<<<numOddSlices * numCellsInSlice, blockSize>>>(
				_positions.devicePtr(), _forces.devicePtr(),_cellInfos.devicePtr(), _domainValues.devicePtr(),
				oddSlicesStartIndex, make_int2( localDimensions ), gridOffsets,
				neighborOffset,
				epsilon, sigmaSquared, cutOffRadiusSquared
			);
	}

	cellPairsTimer.end();

	singleCellsTimer.printElapsedTime( "intra cell LJ forces: %f ms " );
	cellPairsTimer.printElapsedTime( "inter cell LJ forces: %f ms\n" );
}

void LinkedCellsCUDA_Internal::reducePotentialAndVirial( OUT float &potential, OUT float &virial ) {
	const std::vector<unsigned long> &innerCellIndices = _linkedCells.getInnerCellIndices();
	const std::vector<unsigned long> &boundaryCellIndices = _linkedCells.getBoundaryCellIndices();

	potential = 0.0f;
	for( int i = 0 ; i < innerCellIndices.size() ; i++ ) {
		potential += _domainValues[ i ].x;
		virial += _domainValues[ i ].y;
	}
	for( int i = 0 ; i < boundaryCellIndices.size() ; i++ ) {
		potential += _domainValues[ i ].x;
		virial += _domainValues[ i ].y;
	}

	// every contribution is added twice so divide by 2
	potential /= 2.0f;
	virial /= 2.0f;

	// TODO: I have no idea why the sign is different in the GPU code...
	virial = -virial;
}
