#ifndef MOLECULEPAIRHANDLER_H_
#define MOLECULEPAIRHANDLER_H_

#include "cudaComponent.h"

class MoleculePairHandler : public CUDAStaticDataComponent {
public:
	MoleculePairHandler( const CUDAComponent &component ) :
		CUDAStaticDataComponent( component ), _cutOffRadiusSquared( _module.getGlobal<float>("cutOffRadiusSquared") ) {
	}

	virtual void upload() {
		const float cutOffRadius = _linkedCells.getCutoff();
		_cutOffRadiusSquared.set( cutOffRadius * cutOffRadius );
	}

protected:
	CUDA::Global<float> _cutOffRadiusSquared;
};

#endif /* GLOBALSTATS_H_ */
