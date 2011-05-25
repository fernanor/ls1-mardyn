#ifndef CUDA_HELPERS_H_
#define CUDA_HELPERS_H_

#include <malloc.h>
#include <vector>
#include <cuda_runtime.h>
#include <assert.h>

typedef float CUDAPrecisionType;
typedef float3 CUDAPrecisionType3;

struct CUDAException : public std::exception {
	const cudaError_t errorCode;
	const std::string errorSource;

	CUDAException( cudaError_t errorCode, const std::string &errorSource = "" )
	: errorCode( errorCode ), errorSource( errorSource ) {}

	~CUDAException() throw() {}

    /** Returns a C-style character string describing the general cause
     *  of the current error.  */
    virtual const char* what() const throw() {
    	return errorSource.c_str();
    }
};

#define CUDA_THROW_ON_ERROR( expr ) \
	do { \
		cudaError_t errorCode = (expr); \
		if( errorCode != cudaSuccess ) { \
			throw CUDAException( errorCode, #expr ); \
		} \
	} while( 0 )

template<typename type>
class CUDABuffer {
protected:
	type *_hostBuffer;
	type *_deviceBuffer;

	int _byteSize;

public:
	CUDABuffer() : _hostBuffer( 0 ), _deviceBuffer( 0 ), _byteSize( 0 ) {
	}

	~CUDABuffer() {
		delete[] _hostBuffer;
		CUDA_THROW_ON_ERROR( cudaFree( _deviceBuffer ) );
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
		CUDA_THROW_ON_ERROR( cudaFree( _deviceBuffer ) );

		if( count > 0 ) {
			_byteSize = count * sizeof( type );

			_hostBuffer = new type[count];
			CUDA_THROW_ON_ERROR( cudaMalloc( &_deviceBuffer, _byteSize ) );
		}
		else {
			_hostBuffer = 0;
			_deviceBuffer = 0;

			_byteSize = 0;
		}
	}

	void zeroDevice() {
		CUDA_THROW_ON_ERROR( cudaMemset( _deviceBuffer, 0, _byteSize ) );
	}

	void copyToDevice() {
		CUDA_THROW_ON_ERROR( cudaMemcpy( _deviceBuffer, _hostBuffer, _byteSize, cudaMemcpyHostToDevice ) );
	}

	void copyToHost() {
		CUDA_THROW_ON_ERROR( cudaMemcpy( _hostBuffer, _deviceBuffer, _byteSize, cudaMemcpyDeviceToHost ) );
	}

	type *devicePtr() {
		return _deviceBuffer;
	}
};

#define CUDA_TIMING

#ifdef CUDA_TIMING
class CUDATimer {
private:
	cudaEvent_t _startEvent, _endEvent;

public:
	CUDATimer() {
		CUDA_THROW_ON_ERROR( cudaEventCreate( &_startEvent ) );
		CUDA_THROW_ON_ERROR( cudaEventCreate( &_endEvent ) );
	}

	~CUDATimer() {
		CUDA_THROW_ON_ERROR( cudaEventDestroy( _startEvent ) );
		CUDA_THROW_ON_ERROR( cudaEventDestroy( _endEvent ) );
	}

	void begin() {
		CUDA_THROW_ON_ERROR( cudaEventRecord( _startEvent ) );
	}

	void end() {
		CUDA_THROW_ON_ERROR( cudaEventRecord( _endEvent ) );
	}

	float getElapsedTime() {
		CUDA_THROW_ON_ERROR( cudaEventSynchronize( _endEvent ) );

		float elapsedTime;
		CUDA_THROW_ON_ERROR( cudaEventElapsedTime( &elapsedTime, _startEvent, _endEvent ) );

		return elapsedTime;
	}

	void printElapsedTime( const char *format ) {
		printf( format, getElapsedTime() );
	}
};
#else
class CUDATimer {
public:
	void begin() {
	}

	void end() {
	}

	float getElapsedTime() {
		return 0.0f;
	}

	void printElapsedTime( const char *format ) {
	}
};
#endif

#endif
