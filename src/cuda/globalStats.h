/*
 * cellStats.h
 *
 *  Created on: Jun 6, 2011
 *      Author: andreas
 */

#ifndef GLOBALSTATS_H_
#define GLOBALSTATS_H_

#include "cudaComponent.h"

class GlobalStats : public CUDAComponentModule {
public:
	GlobalStats( const CUDA::Module &module, LinkedCells &linkedCells ) :
		CUDAComponentModule(module, linkedCells), _cellStats( module.getGlobal<CellStats *>("cellStats") ), _potential( 0.0f ), _virial( 0.0f ) {
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

	CUDA::Global<CellStats *> _cellStats;

	CUDA::DeviceBuffer<CellStats> _cellStatsBuffer;
};


#endif /* GLOBALSTATS_H_ */
