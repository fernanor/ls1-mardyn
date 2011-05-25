#ifndef CUDA_PAIRTRAVERSER_H_
#define CUDA_PAIRTRAVERSER_H_

// we want to use the CUDA types: int3 etc
#include <vector_types.h>

struct CellPairTraverserTemplate {
	struct TaskInfo {
		int numPairs;
		int startIndex;
		int2 localDimensions;
		int3 gridOffsets;
		int neighborOffset;
	};

	void processTask(const TaskInfo &taskInfo);

	int getDirectionOffset( const int3 &direction );
	int getCellOffset( const int3 &cell );
};

template<class CellTemplate>
void cellPairTraverser(int3 dimensions, CellTemplate &cellInterface ) {
	assert( dimensions.x >= 2 && dimensions.y >= 2 && dimensions.z >=2 );

	/*const dim3 blockSize = dim3( WARP_SIZE, NUM_WARPS, 1 );

	// intra cell forces
	const int *dimensions = _linkedCells.getCellDimensions();
	 */

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

	const int3 cellDirections = make_int3(
			cellInterface.getDirectionOffset( xDirection ),
			cellInterface.getDirectionOffset( zDirection ),
			cellInterface.getDirectionOffset( yDirection )
		);

	//cellPairsTimer.begin();

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
			localDimensions = make_int3( dimensions.y, dimensions.z, dimensions.x );
			gridOffsets = make_int3( cellDirections.y, cellDirections.z, cellDirections.x );
		}
		else if( direction.y == 1 ) {
			// xz plane (main direction: y)
			localDirection = make_int3( direction.x, direction.z, direction.y );
			localDimensions = make_int3( dimensions.x, dimensions.z, dimensions.y );
			gridOffsets = make_int3( cellDirections.x, cellDirections.z, cellDirections.y );
		}
		else if( direction.z == 1 ) {
			// xy plane (main direction: z)
			localDirection = direction;
			localDimensions = make_int3( dimensions.x, dimensions.y, dimensions.z );
			gridOffsets = cellDirections.x, cellDirections.y, cellDirections.z;
		}
		else {
			assert( false );
		}

		// determine the startOffset as first cell near (0,0,0) so that start + neighborOffset won't be out of bounds
		int evenSlicesStartIndex = cellDirections.getCellOffset( -min( direction, zero3 ) );
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


		CellPairTraverserTemplate::TaskInfo taskInfo;

		// set the global parameters
		taskInfo.gridOffsets = gridOffsets;
		taskInfo.localDimensions = make_int2( localDimentions );
		taskInfo.neighborOffset = neighborOffset;

		// do all even slices
		taskInfo.numPairs = numEvenSlices * numCellsInSlice;
		taskInfo.startIndex = evenSlicesStartIndex;

		cellTemplate.processTask( taskInfo );

		/*Kernel_calculatePairLJForces<<<numEvenSlices * numCellsInSlice, blockSize>>>(
				_positions.devicePtr(), _componentLJCenterIndices.devicePtr(), _forces.devicePtr(),
				_componentLJCenterInfos.devicePtr(), _numComponentLJCenters,
				_cellStartIndices.devicePtr(), _domainValues.devicePtr(),
				evenSlicesStartIndex, make_int2( localDimensions ), gridOffsets,
				neighborOffset,
				cutOffRadiusSquared
			);*/

		// do all odd slices
		taskInfo.numPairs = numOddSlices * numCellsInSlice;
		taskInfo.startIndex = oddSlicesStartIndex;

		cellTemplate.processTask( taskInfo );

		/*Kernel_calculatePairLJForces<<<numOddSlices * numCellsInSlice, blockSize>>>(
				_positions.devicePtr(), _componentLJCenterIndices.devicePtr(), _forces.devicePtr(),
				_componentLJCenterInfos.devicePtr(), _numComponentLJCenters,
				_cellStartIndices.devicePtr(), _domainValues.devicePtr(),
				oddSlicesStartIndex, make_int2( localDimensions ), gridOffsets,
				neighborOffset,
				cutOffRadiusSquared
			);*/
	}
}

#endif
