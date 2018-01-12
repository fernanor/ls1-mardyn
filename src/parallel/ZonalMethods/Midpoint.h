/*
 * CommunicationScheme.h
 *
 *  Created on: 14.07.2017
 *      Author: sauermann
 */

#pragma once

#include "ZonalMethod.h"

/**
 * This class implements the Midpoint method. Using the Midpoint method the haloRegion is of size r_c/2.
 * Otherwise the halos are identical to the FullShell method.
 * The Midpoint method requires forces to be exchanged.
 */
class Midpoint: public ZonalMethod  {
public:
	Midpoint(){}
	virtual ~Midpoint(){}

	/**
	 * Returns up to 26 halo Regions of the process.
	 * If a process is spanning a whole dimension, then fewer regions can be returned.
	 * The regions indicate, where the processes lie that require halo copies from the current process.
	 * @param initialRegion boundary of the current process
	 * @param cutoffRadius
	 * @return vector of regions
	 */
	virtual std::vector<HaloRegion> getHaloImportForceExportRegions(HaloRegion& initialRegion, double cutoffRadius,
			bool coversWholeDomain[3]) override {
		return getLeavingExportRegions(initialRegion, cutoffRadius/2., coversWholeDomain);
	}
};
