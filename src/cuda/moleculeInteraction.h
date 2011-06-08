/*
 * moleculeInteraction.h
 *
 *  Created on: Jun 8, 2011
 *      Author: andreas
 */

#ifndef MOLECULEINTERACTION_H_
#define MOLECULEINTERACTION_H_

#include "helpers.h"

#include "cudaComponent.h"
#include "globalStats.h"
#include "moleculeStorage.h"
#include "pairTraverser.h"

class MoleculeInteraction : public CUDAComponent {
public:
	MoleculeInteraction( const CUDA::Module &module, LinkedCells &linkedCells ) : CUDAComponent(module, linkedCells), _globalStats( module, linkedCells ), _moleculeStorage( module, linkedCells ) {
		_cellPairProcessor = module.getFunction("processCellPair");
		_cellProcessor = module.getFunction("processCell");
	}

	void calculate(float &potential, float &virial) {
		_globalStats.preForceCalculation();
		_moleculeStorage.preForceCalculation();

		struct CellPairTraverser {
			MoleculeInteraction &_parent;

			CellPairTraverser(MoleculeInteraction &parent) : _parent( parent ) {
			}

			void processTask(const CellPairTraverserTemplate::TaskInfo &taskInfo) {
				const dim3 blockSize = dim3( WARP_SIZE, NUM_WARPS, 1 );
				_parent._cellPairProcessor().call().
						setBlockShape( blockSize ).
						parameter( taskInfo.startIndex ).
						parameter( taskInfo.localDimensions ).
						parameter( taskInfo.gridOffsets ).
						parameter( taskInfo.neighborOffset ).
						execute( taskInfo.numPairs, 1 );
			}

			int getDirectionOffset( const int3 &direction ) {
				return parent.getDirectionOffset( direction );
			}

			int getCellOffset( const int3 &cell ) {
				return parent.getCellOffset( cell );
			}
		};

		const int *dimensions = _linkedCells.getCellDimensions();
		assert( dimensions[0] >= 2 && dimensions[1] >= 2 && dimensions[2] >=2 );

		cellPairTraverser<CellPairTraverser>( dimensions, CellPairTraverser(*this) );

		_cellProcessor.call().execute( _linkedCells.getCells().size(), 1 );

		_globalStats.postForceCalculation();
		_moleculeStorage.postForceCalculation();

		potential = _globalStats.getPotential();
		virial = _globalStats.getVirial();
	}

protected:
	CUDA::Function _cellPairProcessor, _cellProcessor;

	GlobalStats _globalStats;
	MoleculeStorage _moleculeStorage;

	int getDirectionOffset( const int3 &direction ) {
		return _linkedCells.cellIndexOf3DIndex( direction.x, direction.y, direction.z );
	}

	int getCellOffset( const int3 &cell ) {
		return getDirectionOffset( cell );
	}
};

#endif /* MOLECULEINTERACTION_H_ */
