#ifndef MOLECULEPAIRHANDLER_H_
#define MOLECULEPAIRHANDLER_H_

#include "cudaComponent.h"

class MoleculePairHandler : public CUDAStaticDataComponent {
public:
	MoleculePairHandler( const CUDAComponent &component ) :
		CUDAStaticDataComponent( component ), _cutOffRadiusSquared( _module.getGlobal<floatType>("cutOffRadiusSquared") ) {
	}

	virtual void upload() {
		const floatType cutOffRadius = _linkedCells.getCutoff();
		_cutOffRadiusSquared.set( cutOffRadius * cutOffRadius );
	}

protected:
	CUDA::Global<floatType> _cutOffRadiusSquared;
};

#endif /* GLOBALSTATS_H_ */
