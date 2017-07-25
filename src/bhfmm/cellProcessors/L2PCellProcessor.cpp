/*
 * L2PCellProcessor.cpp
 *
 *  Created on: Feb 10, 2015
 *      Author: tchipev
 */

#include "L2PCellProcessor.h"
#include "Simulation.h"
#include "Domain.h"
#include "parallel/DomainDecompBase.h"
#include "bhfmm/containers/PseudoParticleContainer.h"


namespace bhfmm {

L2PCellProcessor::L2PCellProcessor(
		PseudoParticleContainer * pseudoParticleContainer) :
		_pseudoParticleContainer(pseudoParticleContainer) {
#ifdef ENABLE_MPI
	global_simulation->setOutputString("L2P_CELL_PROCESSOR_L2P", "FMM: Time spent in L2P ");
	//global_simulation->setSyncTimer("L2P_CELL_PROCESSOR_L2P", false); //it is per default false
#endif
}

L2PCellProcessor::~L2PCellProcessor() {
}

void L2PCellProcessor::initTraversal() {
//	using std::cout;
//	using std::endl;
//	Domain* domain = global_simulation->getDomain();
//	cout << "L2P init: LocalUpot     " << domain->getLocalUpot() << endl;
//	cout << "L2P init: LocalVirial   " << domain->getLocalVirial() << endl;
//	cout << "L2P init: LocalP_xx     " << domain->getLocalP_xx() << endl;
//	cout << "L2P init: LocalP_yy     " << domain->getLocalP_yy() << endl;
//	cout << "L2P init: LocalP_zz     " << domain->getLocalP_zz() << endl;
	global_simulation->startTimer("L2P_CELL_PROCESSOR_L2P");
}

void L2PCellProcessor::processCell(ParticleCellPointers& cell) {
	if (!cell.isHaloCell()) {
		_pseudoParticleContainer->processFarField(cell);
	}
}

void L2PCellProcessor::printTimers() {
	DomainDecompBase& domainDecomp = global_simulation->domainDecomposition();
	int numprocs = domainDecomp.getNumProcs();
	int myrank = domainDecomp.getRank();
	for (int i = 0; i < numprocs; i++) {
		if (i == myrank) {
			std::cout << "rank: " << myrank << std::endl;
			std::cout << "\t\t" << global_simulation->getTime("L2P_CELL_PROCESSOR_L2P") << "\t\t" << "s in L2P" << std::endl;
			global_simulation->printTimer("L2P_CELL_PROCESSOR_L2P");
		}
		domainDecomp.barrier();
	}
}

void L2PCellProcessor::endTraversal() {
	global_simulation->stopTimer("L2P_CELL_PROCESSOR_L2P");
}

} /* namespace bhfmm */
