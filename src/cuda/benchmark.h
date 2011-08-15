/*
 * benchmark.h
 *
 *  Created on: Aug 15, 2011
 *      Author: andreas
 */

#ifndef BENCHMARK_H_
#define BENCHMARK_H_

#include <vector>
#include <algorithm>

class Measure {
protected:
	const char *name;

	std::vector< double > values;
public:
	Measure( const char *name ) : name( name ) {}

	const char *getName() const {
		return name;
	}

	void addDataPoint(const double value) {
		values.push_back( value );
	}

	double getAverage() const {
		return std::accumulate( values.begin(), values.end(), 0.0 ) / values.size();
	}
};

#endif /* BENCHMARK_H_ */
