/*
 * cellStats.h
 *
 *  Created on: Jun 6, 2011
 *      Author: andreas
 */

#ifndef GLOBALSTATS_H_
#define GLOBALSTATS_H_

#include "cudaComponent.h"
#include "sharedDecls.h"

class GlobalStats : public CUDAInteractionCalculationComponent {
public:
	GlobalStats( const CUDAComponent &component ) :
		CUDAInteractionCalculationComponent(component), _cellStats( _module.getGlobal<CellStatsStorage *>("cellStats") ), _potential( 0.0f ), _virial( 0.0f ) {
	}

	virtual void preInteractionCalculation();
	virtual void postInteractionCalculation();

	floatType getPotential() const {
		return _potential;
	}

	floatType getVirial() const {
		return _virial;
	}

protected:
	floatType _potential;
	floatType _virial;

	CUDA::Global<CellStatsStorage *> _cellStats;

	CUDA::DeviceBuffer<CellStatsStorage> _cellStatsBuffer;
};


#endif /* GLOBALSTATS_H_ */
