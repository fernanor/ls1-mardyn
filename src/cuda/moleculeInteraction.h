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
			_parent._cellPairProcessor.call().
					setBlockShape( WARP_SIZE, NUM_WARPS, 1 ).
					parameter( taskInfo.startIndex ).
					parameter( taskInfo.localDimensions ).
					parameter( taskInfo.gridOffsets ).
					parameter( taskInfo.neighborOffset ).
					execute( taskInfo.numPairs, 1 );
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
		_cellProcessor( module.getFunction("processCell") )
	{
		// upload data from CUDAStaticDataComponents
		_moleculePairHandler.upload();
		_componentDescriptorStorage.upload();
	}

	void calculate(double &potential, double &virial) {
		CUDAFrameTimer.begin();

		// pre-force calculation handling from CUDAForceCalculationComponents
		CUDAPreTimer.begin();
		_globalStats.preForceCalculation();
		_moleculeStorage.preForceCalculation();
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
		_cellProcessor.call().setBlockShape( WARP_SIZE, NUM_WARPS, 1 ).execute( _linkedCells.getCells().size(), 1 );
		CUDASingleProcessingTimer.end();
		CUDATotalProcessingTimer.end();

		// post-force calculation handling from CUDAForceCalculationComponents
		CUDAPostTimer.begin();
		_globalStats.postForceCalculation();
		_moleculeStorage.postForceCalculation();
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
