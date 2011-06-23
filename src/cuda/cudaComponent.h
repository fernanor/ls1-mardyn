#ifndef CUDAMODULE_H__
#define CUDAMODULE_H__

#include "helpers.h"
#include "particleContainer/Cell.h"
#include "particleContainer/LinkedCells.h"

struct CUDAComponent {
protected:
	CUDA::Module _module;
	LinkedCells &_linkedCells;
	Domain &_domain;

	CUDAComponent( const CUDA::Module &module, LinkedCells &linkedCells, Domain &domain )
		: _module( module ), _linkedCells( linkedCells ), _domain( domain ) {}

	CUDAComponent( const CUDAComponent &component )
		: _module( component._module ), _linkedCells( component._linkedCells ), _domain( component._domain ) {}
};

struct CUDAForceCalculationComponent : public CUDAComponent {
protected:
	CUDAForceCalculationComponent( const CUDAComponent &component ) : CUDAComponent( component ) {}

	virtual void preForceCalculation() = 0;
	virtual void postForceCalculation() = 0;

	virtual ~CUDAForceCalculationComponent() {}
};

struct CUDAStaticDataComponent: public CUDAComponent {
protected:
	CUDAStaticDataComponent( const CUDAComponent &component ) : CUDAComponent( component ) {}

	virtual void upload() = 0;

	virtual ~CUDAStaticDataComponent() {}
};

#endif
