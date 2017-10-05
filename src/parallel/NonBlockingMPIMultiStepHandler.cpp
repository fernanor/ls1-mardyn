/*
 * NonBlockingMPIMultiStepHandler.cpp
 *
 *  Created on: Apr 28, 2016
 *      Author: seckler
 */

#include "Simulation.h"
#include "NonBlockingMPIMultiStepHandler.h"
#include <assert.h>
#include "utils/Logger.h"
#include "DomainDecompMPIBase.h"
#include "particleContainer/ParticleContainer.h"
#include "particleContainer/adapter/CellProcessor.h"

using Log::global_log;
using namespace std;

NonBlockingMPIMultiStepHandler::NonBlockingMPIMultiStepHandler(DomainDecompMPIBase* domainDecomposition,
		ParticleContainer* moleculeContainer, Domain* domain, CellProcessor* cellProcessor) :
				NonBlockingMPIHandlerBase(domainDecomposition, moleculeContainer, domain, cellProcessor) {
	// just calling the constructor of the base class is enough
}

NonBlockingMPIMultiStepHandler::~NonBlockingMPIMultiStepHandler() {
	// nothing to be done
}

void NonBlockingMPIMultiStepHandler::performComputation() {
	int stageCount = _domainDecomposition->getNonBlockingStageCount();

	mardyn_assert(stageCount > 0);

	global_simulation->timers()->start("SIMULATION_COMPUTATION");
	global_simulation->timers()->start("SIMULATION_FORCE_CALCULATION");
	_cellProcessor->initTraversal();
	global_simulation->timers()->stop("SIMULATION_FORCE_CALCULATION");
	global_simulation->timers()->stop("SIMULATION_COMPUTATION");
	for (unsigned int i = 0; i < static_cast<unsigned int>(stageCount); ++i) {
		global_simulation->timers()->start("SIMULATION_DECOMPOSITION");
		_domainDecomposition->prepareNonBlockingStage(false, _moleculeContainer, _domain, i);
		global_simulation->timers()->stop("SIMULATION_DECOMPOSITION");

		// Force calculation and other pair interaction related computations
		global_log->debug() << "Traversing innermost cells" << std::endl;
		global_simulation->timers()->start("SIMULATION_COMPUTATION");
		global_simulation->timers()->start("SIMULATION_FORCE_CALCULATION");
		_moleculeContainer->traversePartialInnermostCells(*_cellProcessor, i, stageCount);
		global_simulation->timers()->stop("SIMULATION_FORCE_CALCULATION");
		global_simulation->timers()->stop("SIMULATION_COMPUTATION");

		global_simulation->timers()->start("SIMULATION_DECOMPOSITION");
		_domainDecomposition->finishNonBlockingStage(false, _moleculeContainer, _domain, i);
		global_simulation->timers()->stop("SIMULATION_DECOMPOSITION");
	}

	global_simulation->timers()->start("SIMULATION_DECOMPOSITION");
	_moleculeContainer->updateBoundaryAndHaloMoleculeCaches();  // update the caches of the other molecules (non-inner cells)
	global_simulation->timers()->stop("SIMULATION_DECOMPOSITION");

	// remaining force calculation and other pair interaction related computations
	global_log->debug() << "Traversing non-innermost cells" << std::endl;
	global_simulation->timers()->start("SIMULATION_COMPUTATION");
	global_simulation->timers()->start("SIMULATION_FORCE_CALCULATION");
	_moleculeContainer->traverseNonInnermostCells(*_cellProcessor);
	_cellProcessor->endTraversal();
	global_simulation->timers()->stop("SIMULATION_FORCE_CALCULATION");
	global_simulation->timers()->stop("SIMULATION_COMPUTATION");
}

void NonBlockingMPIMultiStepHandler::initBalanceAndExchange(bool forceRebalancing, double etime) {

	mardyn_assert(!forceRebalancing);

	global_simulation->timers()->start("SIMULATION_DECOMPOSITION");
	_domainDecomposition->balanceAndExchangeInitNonBlocking(forceRebalancing, _moleculeContainer, _domain);

	// The cache of the molecules must be updated/build after the exchange process,
	// as the cache itself isn't transferred
	_moleculeContainer->updateInnerMoleculeCaches(); // only the caches of the innermost molecules have to be updated.

	global_simulation->timers()->stop("SIMULATION_DECOMPOSITION");
}

