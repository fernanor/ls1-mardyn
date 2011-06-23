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

struct Matrix3x3Storage {
	float3 rows[3];
};

struct QuaternionStorage {
	float w, x, y, z;
};

struct LJParameters {
	float epsilon;
	float sigmaSquared;
};

struct ComponentDescriptor {
	int numLJCenters;

	struct LJCenter {
		float3 position;

		LJParameters ljParameters;
	};

	LJCenter ljCenters[MAX_NUM_LJCENTERS];
};


#endif /* SHAREDDECLS_H_ */
