/*
 * ParticleContainerFactory.cpp
 *
 * @Date: 21.09.2010
 * @Author: eckhardw
 */

#include "particleContainer/tests/ParticleContainerFactory.h"
#include "particleContainer/ParticleContainer.h"
#include "particleContainer/LinkedCells.h"

#include "parallel/DomainDecompBase.h"
#ifdef ENABLE_MPI
#include "parallel/DomainDecomposition.h"
#endif
#include "Domain.h"

#include "io/ASCIIReader.h"
#include "utils/Logger.h"

#ifdef MARDYN_AUTOPAS

#include "particleContainer/AutoPasContainer.h"
#endif

#include <list>

using namespace Log;

ParticleContainer* ParticleContainerFactory::createEmptyParticleContainer(Type type) {
	if (type == LinkedCell) {
		double bBoxMin[] = {0.0, 0.0, 0.0, 0.0};
		double bBoxMax[] = {2.0, 2.0, 2.0, 2.0};
		double cutoffRadius = 1.0;
#ifndef MARDYN_AUTOPAS
		LinkedCells* container = new LinkedCells(bBoxMin, bBoxMax, cutoffRadius);
#else
		AutoPasContainer* container = new AutoPasContainer();
		container->setCutoff(cutoffRadius);
		container->rebuild(bBoxMin, bBoxMax);
#endif
		return container;

	} else {
		global_log->error() << "ParticleContainerFactory: Unsupported type requested! " << std::endl;
		return NULL;
	}
}



ParticleContainer* ParticleContainerFactory::createInitializedParticleContainer(
		Type type, Domain* domain, DomainDecompBase* domainDecomposition, double cutoff, const std::string& fileName) {
	global_simulation->setcutoffRadius(cutoff);
	global_simulation->setLJCutoff(cutoff);

	   ASCIIReader inputReader;
	inputReader.setPhaseSpaceHeaderFile(fileName.c_str());
	inputReader.setPhaseSpaceFile(fileName.c_str());
	inputReader.readPhaseSpaceHeader(domain, 1.0);
	double bBoxMin[3];
	double bBoxMax[3];
	for (int i = 0; i < 3; i++) {
		bBoxMin[i] = domainDecomposition->getBoundingBoxMin(i, domain);
		bBoxMax[i] = domainDecomposition->getBoundingBoxMax(i, domain);
	}

	ParticleContainer* moleculeContainer;
	if (type == Type::LinkedCell) {
#ifndef MARDYN_AUTOPAS
		moleculeContainer = new LinkedCells(bBoxMin, bBoxMax, cutoff);
#else
		moleculeContainer = new AutoPasContainer();
		moleculeContainer->setCutoff(cutoff);
		moleculeContainer->rebuild(bBoxMin, bBoxMax);
#endif
		#ifdef ENABLE_MPI
		DomainDecomposition * temp = 0;
		temp = dynamic_cast<DomainDecomposition *>(domainDecomposition);
		if (temp != 0) {
			temp->initCommunicationPartners(cutoff, domain, moleculeContainer);
		}
		#endif
	} else {
		global_log->error() << "ParticleContainerFactory: Unsupported type requested! " << std::endl;
		return NULL;
	}

	inputReader.readPhaseSpace(moleculeContainer, domain, domainDecomposition);
	moleculeContainer->deleteOuterParticles();
	moleculeContainer->update();
	moleculeContainer->updateMoleculeCaches();

	domain->initParameterStreams(cutoff, cutoff);
	return moleculeContainer;
}
