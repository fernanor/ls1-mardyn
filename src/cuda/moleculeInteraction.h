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
#include "componentDescriptor.h"
#include "globalStats.h"
#include "moleculeStorage.h"
#include "moleculePairHandler.h"
#include "pairTraverser.h"
#include "config.h"

class MoleculeInteraction : public CUDAComponent {
	struct CellPairTraverserTemplate {
		struct TaskInfo {
			int numPairs;
			int startIndex;
			int2 localDimensions;
			int3 gridOffsets;
			int neighborOffset;
		};

		MoleculeInteraction &_parent;

		CellPairTraverserTemplate(MoleculeInteraction &parent) : _parent( parent ) {
		}

		void processTask(const CellPairTraverserTemplate::TaskInfo &taskInfo) const {
			_parent._startIndex.set( taskInfo.startIndex );
			_parent._dimension.set( taskInfo.localDimensions );
			_parent._gridOffsets.set( taskInfo.gridOffsets );
			_parent._neighborOffset.set( taskInfo.neighborOffset );

			_parent._numCellPairs.set( taskInfo.numPairs );

			_parent._cellPairProcessor.call().
					setBlockShape( WARP_SIZE, NUM_WARPS, 1 ).
					executeAtLeast( taskInfo.numPairs );
		}

		int getDirectionOffset( const int3 &direction ) const {
			return _parent.getDirectionOffset( direction );
		}

		int getCellOffset( const int3 &cell ) const {
			return _parent.getCellOffset( cell );
		}
	};

public:
	MoleculeInteraction( const CUDA::Module &module, LinkedCells &linkedCells, Domain &domain ) :
		CUDAComponent(module, linkedCells, domain),

		_globalStats( *this ),
		_moleculeStorage( *this ),

		_componentDescriptorStorage( *this ),
		_moleculePairHandler( *this ),

		_cellPairProcessor( module.getFunction("processCellPair") ),
		_cellProcessor( module.getFunction("processCell") ),

		_startIndex( module.getGlobal<int>("_ZN13PairTraverser10startIndexE") ),
		_dimension( module.getGlobal<int2>("_ZN13PairTraverser9dimensionE") ),
		_gridOffsets( module.getGlobal<int3>("_ZN13PairTraverser11gridOffsetsE") ),
		_neighborOffset( module.getGlobal<int>("_ZN13PairTraverser14neighborOffsetE") ),

		_numCells( module.getGlobal<int>("numCells") ),
		_numCellPairs( module.getGlobal<int>("numCellPairs") )
	{
		// upload data from CUDAStaticDataComponents
		_moleculePairHandler.upload();
		_componentDescriptorStorage.upload();
	}

	void calculate(double &potential, double &virial) {
		CUDAFrameTimer.begin();

		// pre-force calculation handling from CUDAForceCalculationComponents
		CUDAPreTimer.begin();
		_globalStats.preInteractionCalculation();
		_moleculeStorage.preInteractionCalculation();
		CUDAPreTimer.end();

		CUDATotalProcessingTimer.begin();
		const int *raw_dimensions = _linkedCells.getCellDimensions();
		assert( raw_dimensions[0] >= 2 && raw_dimensions[1] >= 2 && raw_dimensions[2] >=2 );

		CUDAPairProcessingTimer.begin();
		const int3 dimensions = make_int3( raw_dimensions[0], raw_dimensions[1], raw_dimensions[2] );
		const CellPairTraverserTemplate cellInterface(*this);
		traverseCellPairs( dimensions, cellInterface );
		CUDAPairProcessingTimer.end();

		CUDASingleProcessingTimer.begin();
		int numCells = _linkedCells.getCells().size();
		_numCells.set( numCells );
		_cellProcessor.call().
				setBlockShape( WARP_SIZE, NUM_WARPS, 1 ).
				executeAtLeast( numCells );
		CUDASingleProcessingTimer.end();
		CUDATotalProcessingTimer.end();

		// post-force calculation handling from CUDAForceCalculationComponents
		CUDAPostTimer.begin();
		_globalStats.postInteractionCalculation();
		_moleculeStorage.postInteractionCalculation();
		CUDAPostTimer.end();

		potential = _globalStats.getPotential();
		virial = _globalStats.getVirial();

		CUDAFrameTimer.end();

		simulationStats.CUDA_frameTime.addDataPoint(CUDAFrameTimer.getElapsedTime());
		simulationStats.CUDA_preTime.addDataPoint(CUDAPreTimer.getElapsedTime());
		simulationStats.CUDA_postTime.addDataPoint(CUDAPostTimer.getElapsedTime());
		simulationStats.CUDA_pairTime.addDataPoint(CUDAPairProcessingTimer.getElapsedTime());
		simulationStats.CUDA_singleTime.addDataPoint(CUDASingleProcessingTimer.getElapsedTime());
		simulationStats.CUDA_processingTime.addDataPoint(CUDATotalProcessingTimer.getElapsedTime());
	}

protected:
	CUDA::Function _cellPairProcessor, _cellProcessor;

	CUDA::Global<int> _startIndex;
	CUDA::Global<int2> _dimension;
	CUDA::Global<int3> _gridOffsets;
	CUDA::Global<int> _neighborOffset;
	CUDA::Global<int> _numCells;
	CUDA::Global<int> _numCellPairs;

	GlobalStats _globalStats;
	MoleculeStorage _moleculeStorage;
	MoleculePairHandler _moleculePairHandler;
	ComponentDescriptorStorage _componentDescriptorStorage;

	CUDA::EventTimer CUDAFrameTimer, CUDAPreTimer, CUDATotalProcessingTimer, CUDAPairProcessingTimer, CUDASingleProcessingTimer, CUDAPostTimer;

	int getDirectionOffset( const int3 &direction ) {
		return _linkedCells.cellIndexOf3DIndex( direction.x, direction.y, direction.z );
	}

	int getCellOffset( const int3 &cell ) {
		return getDirectionOffset( cell );
	}
};

#endif /* MOLECULEINTERACTION_H_ */
