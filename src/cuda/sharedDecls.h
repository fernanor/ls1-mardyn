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
	int numCharges;
	int numDipoles;

	// double3 has a 4byte alignment in gcc x86 and an 8 byte alignment in nvcc
	// it is defined in vector_types.h and its alignment can't be changed easily.
	// FIXME: manual padding required!
	int padding;
	struct LJCenter {
		floatType3 relativePosition;

		LJParameters ljParameters;
	};

	struct Dipole {
		floatType3 relativePosition;
		floatType3 relativeE;

		floatType absMy;
	};

	struct Charge {
		floatType3 relativePosition;

		floatType q;
	};

	LJCenter ljCenters[MAX_NUM_LJCENTERS];
	Charge charges[MAX_NUM_CHARGES];
	Dipole dipoles[MAX_NUM_DIPOLES];
};


#endif /* SHAREDDECLS_H_ */
