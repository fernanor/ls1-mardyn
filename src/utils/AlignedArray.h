/**
 * \file
 * \brief AlignedArray.h
 */

#ifndef ALIGNEDARRAY_H_
#define ALIGNEDARRAY_H_

#ifdef __SSE3__
#include <xmmintrin.h>
#endif
#include <malloc.h>
#include <new>
#include <cstring>

/**
 * \brief An aligned array.
 * \details Has pointer to T semantics.
 * \tparam T The type of the array elements.
 * \tparam alignment The alignment restriction. Must be a power of 2, should not be 8.
 * \author Johannes Heckl
 */
template<class T, size_t alignment = 64>
class AlignedArray {
public:
	/**
	 * \brief Construct an empty array.
	 */
	AlignedArray() :
			_n(0), _p(0) {
	}

	/**
	 * \brief Construct an array of n elements.
	 */
	AlignedArray(size_t n) :
			_n(n), _p(_allocate(n)) {
		if (!_p)
			throw std::bad_alloc();
	}

	/**
	 * \brief Construct a copy of another AlignedArray.
	 */
	AlignedArray(const AlignedArray & a) :
			_n(a._n), _p(_allocate(a._n)) {
		if (!_p)
			throw std::bad_alloc();
		_assign(a);
	}

	/**
	 * \brief Assign a copy of another AlignedArray.
	 */
	AlignedArray & operator=(const AlignedArray & a) {
		resize(a._n);
		_assign(a);
		return *this;
	}

	/**
	 * \brief Free the array.
	 */
	~AlignedArray() {
		_free();
	}

	/**
	 * \brief Reallocate the array. All content may be lost.
	 */
	void resize(size_t n, size_t num_non_zero=0) {
		if (n == _n)
			return;
		_free();
		_p = 0;
		_p = _allocate(n, num_non_zero);
		if (!_p)
			throw std::bad_alloc();
		_n = n;
	}

	inline size_t get_size() const {
		return this->_n;
	}

	/**
	 * \brief Implicit conversion into pointer to T.
	 */
	operator T*() const {
		return _p;
	}

	/**
	 * \brief Return amount of allocated storage + .
	 */
	size_t get_dynamic_memory() const {
		return _n * sizeof(T);
	}

private:
	void _assign(T * p) const {
		std::memcpy(_p, p, _n * sizeof(T));
	}
	static T* _allocate(size_t elements, size_t num_non_zero=0) {
#if defined(__SSE3__) && ! defined(__PGI)
		T* ptr = static_cast<T*>(_mm_malloc(sizeof(T) * elements, alignment));
		//std::memset(ptr + num_non_zero, 0, (elements - num_non_zero) * sizeof(T));
		std::memset(ptr, 0, sizeof(T) * elements);
		return ptr;
#else
		T* ptr = static_cast<T*>(memalign(alignment, sizeof(T) * elements));
		//std::memset(ptr + num_non_zero, 0, (elements - num_non_zero) * sizeof(T));
		std::memset(ptr, 0, elements * sizeof(T));
		return ptr;
#endif
	}

	void _free()
	{
#if defined(__SSE3__) && ! defined(__PGI)
		_mm_free(_p);
#else
		free(_p);
#endif
	}

	size_t _n;
	T * _p;
};

#endif
