#ifndef CUDA_HELPERS_H_
#define CUDA_HELPERS_H_

#include <cuda.h>
#include <vector_types.h>

#include <malloc.h>
#include <vector>

#include <assert.h>
#include <string>

// TODO: use a namespace instead of the slightly stupid CUDA prefix

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

	#define CUDA_THROW_ON_ERROR( expr ) do {\
			CUresult result = (expr);\
			if( result != CUDA_SUCCESS ) {\
				assert(false);\
				throw CUDA::Exception( result, #expr ); \
			}\
		} while(false)


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

			const int byteSize = count * sizeof( DataType );
			if( byteSize > 0 ) {
				_byteSize = byteSize;

				CUDA_THROW_ON_ERROR( cuMemAlloc( &_deviceBuffer, _byteSize ) );
			}
			else {
				_deviceBuffer = 0;

				_byteSize = 0;
			}
		}

		void zeroDevice() {
			CUDA_THROW_ON_ERROR( cuMemsetD8( _deviceBuffer, 0, _byteSize ) );
		}

		void copyToDevice(const DataType *data, int count) {
			const int byteSize = count * sizeof( data );
			if( _byteSize < byteSize )
				resize( byteSize );

			CUDA_THROW_ON_ERROR( cuMemcpyHtoD( _deviceBuffer, data, _byteSize ) );
		}

		void copyToDevice(const std::vector<DataType> &collection) {
			copyToDevice( &collection.front(), collection.size() );
		}

		void copyToHost(std::vector<DataType> &collection) {
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
			// align with parameter size
			_offset = (_offset + (__alignof(T) - 1)) & ~(__alignof(T) - 1);
			CUDA_THROW_ON_ERROR( cuParamSetv( _function, _offset, (void *) &param, sizeof(T) ) );
			_offset += sizeof(T);

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

			CUDA_THROW_ON_ERROR( cuLaunchGrid( _function, gridWidth, gridHeight ) );
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

	template<typename DataType> struct TypeInfo {
		const static int size = sizeof( DataType );
	};

	template<typename DataType> struct TypeInfo<DataType *> {
		const static int size = sizeof( CUdeviceptr );
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
			size_t bytes;

			CUDA_THROW_ON_ERROR( cuModuleGetGlobal( &dptr, &bytes, _module, name ) );

			assert( bytes == sizeof( TypeInfo<DataType> ) );

			return Global<DataType>(dptr);
		}

		Function getFunction(const char*name) const {
			CUfunction function;

			CUDA_THROW_ON_ERROR( cuModuleGetFunction( &function, _module, name ) );

			return Function( function );
		}
	};

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
		singleton = NULL;
	}

	Module loadModule(const char *name) {
		CUmodule module;
		CUDA_THROW_ON_ERROR( cuModuleLoad( &module, name ) );

		return Module( module );
	}
};

inline CUDA &cuda() {
	return CUDA::get();
}

#endif
