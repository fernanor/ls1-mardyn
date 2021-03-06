#ifndef ACCUMULATOR_H
#define ACCUMULATOR_H

/*
 * Copyright (c) 2013      Christoph Niethammer <christoph.niethammer@gmail.com>
 */

#include <algorithm>
#include "utils/mardyn_assert.h"
#include <cmath>
#include <cstdlib>


/** A generalized Accumlator for scalar values.
 *
 * The accumulator uses a sliding window to calculate average values.
 */
template <typename T>
class Accumulator {
public:
    /** Constructor creating new Accumlator
     * @param[in]  windowLength  Number of elements in the sliding window.
     */
    Accumulator(size_t windowLength = 100) : _windowLength(windowLength), _insertPosition(0), _size(0) {
        _data = new T[windowLength];
    }
    /** Destructor */
    ~Accumulator(){
		delete[] _data;
	}

    /** Return the average of the values in the sliding window.
     * @return Average of the sliding window.
     */
    T getAverage() {
        T sum = 0;
        for( size_t i = 0; i < _size; i++ ) {
            sum += _data[i];
		}
        return sum / _size;
    }

    /** Return the standard deviation of the values in the sliding window.
	 * @return Standard deviation of the sliding window
	 */
    T getStddev() {
        T avg = getAverage();
        T stddev = 0;
        for( size_t i = 0; i < _size; i++ ) {
            T diff = _data[i] - avg;
            stddev += diff*diff;
        }
        stddev /= getSize();
        stddev = sqrt(stddev);
        return stddev;
    }

    /** Return percentile
	 * @param[in]  p  percentile as real, e.g. 0.5 for the median (50\%)
	 * @return Requested percentile of the sliding window
	 */
    T getPercentile(double p) {
        T *sorted_data = new T[_size];
        std::copy(_data, _data + _size, sorted_data);
        std::sort(sorted_data, sorted_data + _size);
        T retValue = sorted_data[static_cast<size_t>(_size * p) - 1];
        delete[] sorted_data;
        return retValue;
    }

    /** Return inter percentile range
	 * @param[in]  pmin  lower percentile as real, e.g. 0.25 for the median (25\%)
	 * @param[in]  pmax  upper percentile as real, e.g. 0.75 for the median (75\%)
	 * @return Requested inter percentile range of the sliding window
	 */
    T getInterPercentileRange(double pmin, double pmax) {
        T *sorted_data = new T[_size];
        std::copy(_data, _data + _size, sorted_data);
        std::sort(sorted_data, sorted_data + _size);
        T qmin = sorted_data[static_cast<size_t>(_size * pmin) - 1];
        T qmax = sorted_data[static_cast<size_t>(_size * pmax) - 1];
        T retValue = qmax - qmin;
        delete[] sorted_data;
        return retValue;
    }

    /** Move sliding window forward by adding a new value replacing the oldest entry.
     * @param[in]  value  value to be inserted at the begin of the moved window.
     */
    void addEntry(T value) {
        _data[_insertPosition] = value;
        _insertPosition = (_insertPosition + 1) % _windowLength;
		if (_size < _windowLength)
			_size += 1;
    }

    /** Return the last added value.
     * @return    Last value added to accumulator.
     */
    T getLastEntry() {
        return getEntry(1);
    }

    /** Return the entry added id times before.
     * @param[in]  id  ID of element, 1 gives the last added element.
     * @return     value added id times before
     */
    T getEntry(size_t id) {
        mardyn_assert(0 < id && id <= _windowLength);
        return _data[(_insertPosition + _windowLength - id) % _windowLength];
    }

	/** Get the size of the sliding window.
	 * @return     Size of the sliding window
	 */
    size_t getWindowLength() {
        return _windowLength;
    }

    /** Get the number of elements within the sliding window.
	 * The number of elements can be less than the window length as long as not enough
	 * elements were added to the accumulator. This method allows to query for the
	 * number of elements in the window.
	 * @return     Number of elements in the sliding window.
	 */
    size_t getSize() {
		return _size;
	}

private:
    size_t _windowLength;   /**< Number of entries . */
    size_t _insertPosition; /**< Current insertion position in the data array. */
    size_t _size;           /**< Actual elements in the window. */
    T *_data;               /**< Array storing values inside the slidling window */
};

#endif /* ACCUMULATOR_H */
