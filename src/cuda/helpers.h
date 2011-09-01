#ifndef CUDA_HELPERS_H_
#define CUDA_HELPERS_H_

#include <cuda.h>
#include <vector_types.h>

#include <malloc.h>
#include <vector>

#include <assert.h>
#include <string>
#include <stddef.h>

// TODO: use a namespace instead of the slightly stupid CUDA prefix

namespace CUDADetail {
	template<typename DataType> struct TypeInfo {
		const static size_t size = sizeof( DataType );
		const static size_t alignof = __alignof( DataType );
	};

	template<> struct TypeInfo<double3> {
		const static size_t size = sizeof( double3 );
		const static size_t alignof = 8;
	};

	template<typename DataType> struct TypeInfo<DataType *> {
		const static size_t size = sizeof( CUdeviceptr );
		const static size_t alignof = __alignof( CUdeviceptr );
	};
}

class CUDA {
public:
	struct Exception : public std::exception {
		const CUresult errorCode;
		const std::string errorSource;

		Exception( CUresult errorCode, const std::string &errorSource = "" )
		: errorCode( errorCode ), errorSource( errorSource ) {}

		~Exception() throw() {}

		/** Returns a C-style character string describing the general cause
		 *  of the current error.  */
		virtual const char* what() const throw() {
			return errorSource.c_str();
		}
	};

	#define CUDA_THROW_ON_ERROR_EX( expr, msgformat, ... ) do {\
			CUresult result = (expr);\
			if( result != CUDA_SUCCESS ) {\
				printf( "CUDA error %i in '" #expr "'!\n" msgformat, result, ##__VA_ARGS__ );\
				throw CUDA::Exception( result, #expr ); \
			}\
		} while(false)

	#define CUDA_THROW_ON_ERROR( expr ) CUDA_THROW_ON_ERROR_EX( expr, "" )

	template<typename DataType>
	class DeviceBuffer {
	protected:
		CUdeviceptr _deviceBuffer;

		int _byteSize;

	public:
		DeviceBuffer() : _deviceBuffer( 0 ), _byteSize( 0 ) {
		}

		~DeviceBuffer() {
			if( _byteSize )
				CUDA_THROW_ON_ERROR( cuMemFree( _deviceBuffer ) );
		}

		void resize(int count) {
			if( _byteSize )
				CUDA_THROW_ON_ERROR( cuMemFree( _deviceBuffer ) );

			_byteSize = count * sizeof( DataType );
			if( _byteSize > 0 ) {
				CUDA_THROW_ON_ERROR_EX( cuMemAlloc( &_deviceBuffer, _byteSize ), "\twhile allocating %i MB\n", _byteSize / (1<<20) );
			}
			else {
				_deviceBuffer = 0;
				// if _byteSize < 0
				_byteSize = 0;
			}
		}

		void zeroDevice() {
			CUDA_THROW_ON_ERROR( cuMemsetD8( _deviceBuffer, 0, _byteSize ) );
		}

		void copyToDevice(const DataType *data, int count) {
			const int byteSize = count * sizeof( DataType );
			if( _byteSize < byteSize )
				resize( count );

			CUDA_THROW_ON_ERROR( cuMemcpyHtoD( _deviceBuffer, data, byteSize ) );
		}

		void copyToDevice(const std::vector<DataType> &collection) {
			copyToDevice( &collection.front(), collection.size() );
		}

		void copyToHost(std::vector<DataType> &collection) {
			if( !_byteSize ) {
				return;
			}

			int count = _byteSize / sizeof( DataType );
			collection.resize( count );

			CUDA_THROW_ON_ERROR( cuMemcpyDtoH( &collection.front(), _deviceBuffer, count * sizeof( DataType ) ) );
		}

		CUdeviceptr devicePtr() const {
			return _deviceBuffer;
		}
	};

	class Timer {
	private:
		CUevent _startEvent, _endEvent;

		Timer( const Timer & );
		Timer & operator =( const Timer & );
	public:
		Timer() {
			CUDA_THROW_ON_ERROR( cuEventCreate( &_startEvent, ::CU_EVENT_DEFAULT ) );
			CUDA_THROW_ON_ERROR( cuEventCreate( &_endEvent, ::CU_EVENT_DEFAULT ) );
		}

		~Timer() {
			CUDA_THROW_ON_ERROR( cuEventDestroy( _startEvent ) );
			CUDA_THROW_ON_ERROR( cuEventDestroy( _endEvent ) );
		}

		void begin() {
			CUDA_THROW_ON_ERROR( cuEventRecord( _startEvent, 0 ) );
		}

		void end() {
			CUDA_THROW_ON_ERROR( cuEventRecord( _endEvent, 0 ) );
		}

		float getElapsedTime() {
			CUDA_THROW_ON_ERROR( cuEventSynchronize( _endEvent ) );

			float elapsedTime;
			CUDA_THROW_ON_ERROR( cuEventElapsedTime( &elapsedTime, _startEvent, _endEvent ) );

			return elapsedTime;
		}
	};

	class NullTimer {
	public:
		void begin() {
		}

		void end() {
		}

		float getElapsedTime() {
			return 0.0f;
		}
	};

	typedef Timer EventTimer;

	class Function;
	class Module;

	template<typename DataType>
	class Global {
	protected:
		CUdeviceptr _dataPointer;

		friend class Module;

		Global( CUdeviceptr dataPointer ) : _dataPointer( dataPointer ) {}

	public:
		void set( const DataType &data ) {
			CUDA_THROW_ON_ERROR( cuMemcpyHtoD( _dataPointer, &data, sizeof( DataType ) ) );
		}
	};

	template<typename DataType>
	class Global<DataType *> {
	protected:
		CUdeviceptr _dataPointer;

		friend class Module;

		Global( CUdeviceptr dataPointer ) : _dataPointer( dataPointer ) {}

	public:
		void set( const CUdeviceptr data ) {
			CUDA_THROW_ON_ERROR( cuMemcpyHtoD( _dataPointer, &data, sizeof( CUdeviceptr ) ) );
		}

		void set( const DeviceBuffer<DataType> &buffer ) {
			set( buffer.devicePtr() );
		}
	};

	class FunctionCall {
	protected:
		CUfunction _function;
		int _offset;

		friend class Function;

		FunctionCall( const CUfunction &function ) : _function( function ), _offset( 0 ) {}

	public:
		template<typename T>
		FunctionCall & parameter( const T &param ) {
			using namespace CUDADetail;

			// align with parameter size
			_offset = (_offset + (TypeInfo<T>::alignof - 1)) & ~(TypeInfo<T>::alignof - 1);
			CUDA_THROW_ON_ERROR( cuParamSetv( _function, _offset, (void *) &param, TypeInfo<T>::size ) );
			_offset += TypeInfo<T>::size;

			return *this;
		}

		template<typename T>
		FunctionCall & parameter( const DeviceBuffer<T> &param ) {
			using namespace CUDADetail;

			// align with parameter size
			_offset = (_offset + (TypeInfo<CUdeviceptr>::alignof - 1)) & ~(TypeInfo<CUdeviceptr>::alignof - 1);

			CUdeviceptr devicePtr = param.devicePtr();
			CUDA_THROW_ON_ERROR( cuParamSetv( _function, _offset, (void *) &devicePtr, TypeInfo<CUdeviceptr>::size ) );

			_offset += TypeInfo<CUdeviceptr>::size;

			return *this;
		}

		FunctionCall & setBlockShape( int x, int y, int z ) {
			CUDA_THROW_ON_ERROR( cuFuncSetBlockShape( _function, x, y, z ) );

			return *this;
		}

		FunctionCall & setBlockShape( const dim3 &shape) {
			CUDA_THROW_ON_ERROR( cuFuncSetBlockShape( _function, shape.x, shape.y, shape.z ) );

			return *this;
		}

		void execute(int gridWidth, int gridHeight) const {
			CUDA_THROW_ON_ERROR( cuParamSetSize( _function, _offset ) );

			CUDA_THROW_ON_ERROR_EX( cuLaunchGrid( _function, gridWidth, gridHeight ), "\twhile launching a grid of size %i x %i\n", gridWidth, gridHeight );
		}

		void execute(int numJobs) const {
			execute( numJobs, 1 );
		}

		void executeAtLeast( int numJobs ) const {
			for( int log2_y = 0 ; log2_y < 15 ; log2_y++ ) {
				int y = 1 << log2_y;
				int x = numJobs >> log2_y;
				if( x < 65536 ) {
					execute( x, y );
					return;
				}
			}

			CUDA_THROW_ON_ERROR_EX( CUDA_ERROR_INVALID_VALUE, "\tnumJobs = %i > 65535^2\n", numJobs );
		}

		void execute() const {
			CUDA_THROW_ON_ERROR( cuParamSetSize( _function, _offset ) );

			CUDA_THROW_ON_ERROR( cuLaunch( _function ) );
		}
	};

	class Function {
	protected:
		CUfunction _function;

		friend class Module;

		Function( CUfunction function ) : _function( function ) {}

	public:
		FunctionCall call() const {
			return FunctionCall( _function );
		}
	};

	class Module {
	protected:
		CUmodule _module;

		friend class CUDA;

		Module(CUmodule module) : _module( module ) {}

	public:

		template<typename DataType>
		Global<DataType> getGlobal(const char *name) const {
			CUdeviceptr dptr;
			size_t dataSize;

			CUDA_THROW_ON_ERROR( cuModuleGetGlobal( &dptr, &dataSize, _module, name ) );

			assert( dataSize == CUDADetail::TypeInfo<DataType>::size );

			return Global<DataType>(dptr);
		}

		Function getFunction(const char *name) const {
			CUfunction function;

			CUDA_THROW_ON_ERROR( cuModuleGetFunction( &function, _module, name ) );

			return Function( function );
		}
	};

protected:
	CUdevice device;
	CUcontext context;

	CUDA(int deviceIndex, bool preferL1Cache) {
		CUDA_THROW_ON_ERROR( cuInit( 0 ) );

		CUDA_THROW_ON_ERROR( cuDeviceGet( &device, deviceIndex ) );

		printMemoryInfo();

		CUDA_THROW_ON_ERROR( cuCtxCreate( &context, 0, device ) );

		CUDA_THROW_ON_ERROR( cuCtxSetCacheConfig( preferL1Cache ? CU_FUNC_CACHE_PREFER_L1 : CU_FUNC_CACHE_PREFER_SHARED ) );
	}

	~CUDA() {
		CUDA_THROW_ON_ERROR( cuCtxDestroy( context ) );

		printMemoryInfo();
	}

	void printMemoryInfo() {
		CUcontext infoContext;
		CUDA_THROW_ON_ERROR( cuCtxCreate( &infoContext, 0, device ) );

		size_t free, total;
		CUDA_THROW_ON_ERROR( cuMemGetInfo( &free, &total ) );

		CUDA_THROW_ON_ERROR( cuCtxDestroy( infoContext ) );

		printf( "CUDA memory info: %i MB free (%i MB total); %f%% free\n", free / (1<<20), total / (1<<20), (float) free / total );
	}

	static CUDA *singleton;

public:

	static CUDA &get() {
		assert( singleton );
		return *singleton;
	}

	static void create(int deviceIndex, bool preferL1Cache) {
		assert( !singleton );
		singleton = new CUDA(deviceIndex, preferL1Cache);
	}

	static void destruct() {
		delete singleton;
		singleton = NULL;
	}

	Module loadModule(const char *name) {
		CUmodule module;
		CUDA_THROW_ON_ERROR( cuModuleLoad( &module, name ) );

		return Module( module );
	}

	Module loadModuleData(const char *imageStart, const char *imageEnd, int maxRegisterCount) {
		std::string image(imageStart, imageEnd);

		CUmodule module;
		CUjit_option options = CU_JIT_MAX_REGISTERS;
		CUDA_THROW_ON_ERROR( cuModuleLoadDataEx( &module, image.c_str(), 1, &options, (void**) &maxRegisterCount ) );

		return Module( module );
	}
};

inline CUDA &cuda() {
	return CUDA::get();
}

#endif
