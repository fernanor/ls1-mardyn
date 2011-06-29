/*
 * sharedDecls.h
 *
 *  Created on: Jun 21, 2011
 *      Author: andreas
 */

#ifndef SHAREDDECLS_H_
#define SHAREDDECLS_H_

#include <host_defines.h>

#include "config.h"

typedef char Molecule_ComponentType;

struct CellStatsStorage {
	floatType potential;
	floatType virial;
};

struct Matrix3x3Storage {
	floatType3 rows[3];
};

struct QuaternionStorage {
	floatType w, x, y, z;
};

struct LJParameters {
	floatType epsilon;
	floatType sigma;
};

struct ComponentDescriptor {
	int numLJCenters;

	int padding;

	struct LJCenter {
		floatType3 relativePosition;

		LJParameters ljParameters;
	};

	LJCenter ljCenters[MAX_NUM_LJCENTERS];
};


#endif /* SHAREDDECLS_H_ */
