/*
 * cellStats.h
 *
 *  Created on: Jun 6, 2011
 *      Author: andreas
 */

#ifndef GLOBALSTATS_H_
#define GLOBALSTATS_H_

#include "cudaComponent.h"

class GlobalStats : public CUDAComponent {
public:
	GlobalStats( const CUDA::Module &module, LinkedCells &linkedCells ) : CUDAComponent(module, linkedCells), _maxMoleculeStorage( 0 ) {
		_cellStats = module.getGlobal("cellStats");
	}

	virtual void preForceCalculation();
	virtual void postForceCalculation();

	float getPotential() const {
		return _potential;
	}

	float getVirial() const {
		return _virial;
	}

protected:
	float _potential;
	float _virial;

	struct CellStats {
		float potential;
		float virial;
	};

	CUDA::Global _cellStats;

	CUDA::DeviceBuffer _cellStatsBuffer;
};


#endif /* GLOBALSTATS_H_ */
