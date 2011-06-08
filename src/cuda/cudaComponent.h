#ifndef CUDAMODULE_H__
#define CUDAMODULE_H__

#include "helpers.h"
#include "particleContainer/LinkedCells.h"

struct CUDAComponent {
protected:
	CUDA::Module _module;
	LinkedCells &_linkedCells;

	CUDAComponent( const CUDA::Module &module, LinkedCells &linkedCells ) : module( module ), _linkedCells( linkedCells ) {}
};

struct CUDAComponentModule {
protected:
	CUDAComponentModule( const CUDA::Module &module, LinkedCells &linkedCells ) : CUDAComponent( module, linkedCells ) {}

	virtual void preForceCalculation() = 0;
	virtual void postForceCalculation() = 0;
};

#endif
