#ifndef MOLECULEPAIRHANDLER_H_
#define MOLECULEPAIRHANDLER_H_

#include "cudaComponent.h"

class MoleculePairHandler : public CUDAComponentModule {
public:
	MoleculePairHandler( const CUDA::Module &module, LinkedCells &linkedCells ) :
		CUDAComponentModule(module, linkedCells), _cutOffRadiusSquared( module.getGlobal<float>("cutOffRadiusSquared") ) {
	}

	virtual void preForceCalculation() {
		const float cutOffRadius = _linkedCells.getCutoff();
		_cutOffRadiusSquared.set( cutOffRadius * cutOffRadius );
	}

	virtual void postForceCalculation() {}

protected:
	CUDA::Global<float> _cutOffRadiusSquared;
};

#endif /* GLOBALSTATS_H_ */
