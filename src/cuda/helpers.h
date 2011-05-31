#ifndef CUDA_HELPERS_H_
#define CUDA_HELPERS_H_

#include <malloc.h>
#include <vector>
#include <cuda.h>
#include <assert.h>

// TODO: use a namespace instead of the slightly stupid CUDA prefix

struct CUDAException : public std::exception {
	const CUresult errorCode;
	const std::string errorSource;

	CUDAException( CUresult errorCode, const std::string &errorSource = "" )
	: errorCode( errorCode ), errorSource( errorSource ) {}

	~CUDAException() throw() {}

    /** Returns a C-style character string describing the general cause
     *  of the current error.  */
    virtual const char* what() const throw() {
    	return errorSource.c_str();
    }
};

#define CUDA_THROW_ON_ERROR( expr ) do {\
        CUresult result = (expr);\
        if( result != CUDA_SUCCESS ) {\
            assert(false);\
            throw CUDAException( result, #expr ); \
        }\
    } while(false)

template<typename type>
class CUDABuffer {
protected:
	type *_hostBuffer;
	CUdeviceptr *_deviceBuffer;

	int _byteSize;

public:
	CUDABuffer() : _hostBuffer( 0 ), _deviceBuffer( 0 ), _byteSize( 0 ) {
	}

	~CUDABuffer() {
		delete[] _hostBuffer;
		if( _byteSize )
			CUDA_THROW_ON_ERROR( cuMemFree( _deviceBuffer ) );
	}

	type & operator []( int index ) {
		return _hostBuffer[ index ];
	}

	const type & operator []( int index ) const {
		return _hostBuffer[ index ];
	}

	type & operator *() {
		return *_hostBuffer;
	}

	const type & operator *() const {
		return *_hostBuffer;
	}

	operator type *() {
		return _hostBuffer;
	}

	operator const type *() const {
		return _hostBuffer;
	}

	void resize(int count) {
		delete[] _hostBuffer;
		if( _byteSize )
			CUDA_THROW_ON_ERROR( cuMemFree( _deviceBuffer ) );

		if( count > 0 ) {
			_byteSize = count * sizeof( type );

			_hostBuffer = new type[count];
			CUDA_THROW_ON_ERROR( cuMemAlloc( &_deviceBuffer, _byteSize ) );
		}
		else {
			_hostBuffer = 0;
			_deviceBuffer = 0;

			_byteSize = 0;
		}
	}

	void zeroDevice() {
		CUDA_THROW_ON_ERROR( cuMemsetD8( _deviceBuffer, 0, _byteSize ) );
	}

	void copyToDevice() {
		CUDA_THROW_ON_ERROR( cuMemcpyHtoD( _deviceBuffer, _hostBuffer, _byteSize ) );
	}

	void copyToHost() {
		CUDA_THROW_ON_ERROR( cuMemcpyDtoH( _hostBuffer, _deviceBuffer, _byteSize ) );
	}

	CUdeviceptr devicePtr() const {
		return _deviceBuffer;
	}
};

class CUDATimer {
private:
    CUevent _startEvent, _endEvent;

public:
    CUDATimer() {
        expectCUDASuccess( cuEventCreate( &_startEvent, ::CU_EVENT_DEFAULT ) );
        expectCUDASuccess( cuEventCreate( &_endEvent, ::CU_EVENT_DEFAULT ) );
    }

    ~CUDATimer() {
        expectCUDASuccess( cuEventDestroy( _startEvent ) );
        expectCUDASuccess( cuEventDestroy( _endEvent ) );
    }

    void begin() {
        expectCUDASuccess( cuEventRecord( _startEvent, 0 ) );
    }

    void end() {
        expectCUDASuccess( cuEventRecord( _endEvent, 0 ) );
    }

    float getElapsedTime() {
        expectCUDASuccess( cuEventSynchronize( _endEvent ) );

        float elapsedTime;
        expectCUDASuccess( cuEventElapsedTime( &elapsedTime, _startEvent, _endEvent ) );

        return elapsedTime;
    }
};

class ZeroTimer {
public:
    void begin() {
    }

    void end() {
    }

    float getElapsedTime() {
        return 0.0f;
    }

};

typedef CUDATimer EventTimer;

template<typename DataType>
class CUDAGlobal {
protected:
	CUdeviceptr _dataPointer;

	friend class CUDAModule;

	CUDAGlobal( CUdeviceptr dataPointer ) : _dataPointer( dataPointer ) {}
public:

	void set( const DataType &data ) {
		CUDA_THROW_ON_ERROR( cuMemcpyHtoD( _dataPointer, &data, sizeof( DataType ) ) );
	}
};

class CUDAFunctionCall {
private:
    int _offset;

    int _gridWidth, _gridHeight;

    CUfunction _function;

    CUDAFunctionCall( const CUfunction &function ) : _function( function ), _offset( 0 ),_gridWidth( 1 ), _gridHeight( 1 ) {}

public:
    template<typename T>
    CUDAFunctionCall & parameter( const T &param ) {
        // align with parameter size
    	_offset = (_offset + (__alignof(T) - 1)) & ~(__alignof(T) - 1);
    	CUDA_THROW_ON_ERROR( cuParamSetv( function, offset, (void *) &param, sizeof(T) ) );
        _offset += sizeof(T);

        return *this;
    }

    CUDAFunctionCall & setBlockShape( int x, int y, int z ) {
    	CUDA_THROW_ON_ERROR( cuFuncSetBlockShape( _function, x, y, z ) );s

    	return *this;
    }

    CUDAFunctionCall & setGridSize( int gridWidth, int gridHeight ) {
    	_gridWidth = gridWidth;
    	_gridHeight = gridHeight;

    	return *this;
    }

    void execute() {
    	CUDA_THROW_ON_ERROR( cuParamSetSize( _function, _offset ) );

        CUDA_THROW_ON_ERROR( cuLaunch( _function ) );
    }
};

class CUDAFunction {
protected:
	CUfunction _function;

	friend class CUDAModule;

	CUDAFunction( CUfunction function ) : _function( function ) {}

public:
	CUDAFunctionCall call() {
		return CUDAFunctionCall( _function );
	}

};

class CUDAModule {
protected:
	CUmodule _module;

	friend class CUDA;

	CUDAModule(CUmodule module) : _module( module ) {}

public:

	template<typename DataType>
	CUDAGlobal<DataType> getGlobal(const char *name) {
		CUdeviceptr dptr;
		size_t bytes;

		CUDA_THROW_ON_ERROR( cuModuleGetGlobal( &dptr, &bytes, _module, name ) );

		assert( bytes == sizeof( DataType ) );

		return CUDAGlobal(dptr);
	}

	CUDAFunction getFunction(const char*name) {
		CUfunction function;

		CUDA_THROW_ON_ERROR( cuModuleGetFunction( &function, _module, name ) );

		return CUADFunction( function );
	}
};

class CUDA {
protected:
	CUcontext context;

	static CUDA *singleton;

	CUDA(CUcontext context) : context(context) {}
public:

	static CUDA &get() {
		assert( singleton );
		return *singleton;
	}

	static void create(int deviceIndex) {
		assert( !singleton );

		CUDA_THROW_ON_ERROR( cuInit( 0 ) );

		CUdevice device;
		CUDA_THROW_ON_ERROR( cuDeviceGet( &device, deviceIndex ) );

		CUcontext context;
		CUDA_THROW_ON_ERROR( cuCtxCreate( &context, 0, device ) );

		singleton = new CUDA( context );
	}

	static void destruct() {
		CUDA_THROW_ON_ERROR( cuCtxDestroy( singleton->context ) );

		delete singleton;
		singleton = 0;
	}

	CUDAModule loadModule(const char *name) {
		CUmodule module;
		CUDA_THROW_ON_ERROR( cuModuleLoad( &module, name ) );

		return CUDAModule( module );
	}
};

inline CUDA &cuda() {
	return CUDA.get();
}
