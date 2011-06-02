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

#endif
