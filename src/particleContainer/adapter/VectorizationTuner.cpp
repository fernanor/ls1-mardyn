/*
 * VectorizationTuner.cpp
 *
 *  Created on: Jun 12, 2015
 *      Author: tchipevn
 */

#include "VectorizationTuner.h"

#include <vector>

#include "ensemble/EnsembleBase.h"
#include "molecules/Component.h"
#include "Simulation.h"
#include "particleContainer/ParticleCell.h"
#include "CellDataSoA.h"
#include "Domain.h"
#include "parallel/DomainDecompBase.h"
#include "molecules/Molecule.h"
#include "utils/Logger.h"
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <stdio.h>
//#include <string.h>

//preprocessing macros
//#define MASKING

VectorizationTuner::VectorizationTuner(double cutoffRadius, double LJCutoffRadius, CellProcessor **cellProcessor):
	_outputPrefix("Mardyn"), _minMoleculeCnt(2), _maxMoleculeCnt(512), _moleculeCntIncreaseType(both),
	_cellProcessor(cellProcessor), _cutoffRadius(cutoffRadius), _LJCutoffRadius(LJCutoffRadius), _flopCounterBigRc(NULL), _flopCounterNormalRc(NULL), _flopCounterZeroRc(NULL) {

}

VectorizationTuner::~VectorizationTuner() {
	// TODO Auto-generated destructor stub
}

void VectorizationTuner::readXML(XMLfileUnits& xmlconfig) {
	_outputPrefix = "mardyn";

	xmlconfig.getNodeValue("outputprefix", _outputPrefix);
	global_log->info() << "Output prefix: " << _outputPrefix << std::endl;

	xmlconfig.getNodeValue("minmoleculecnt", _minMoleculeCnt);
	global_log->info() << "Minimal molecule count: " << _minMoleculeCnt << std::endl;

	xmlconfig.getNodeValue("maxmoleculecnt", _maxMoleculeCnt);
	global_log->info() << "Maximal molecule count: " << _maxMoleculeCnt << std::endl;

	//! @todo This is a very improper way to do this - user does not know what the int values stand for
	int type=2;
	xmlconfig.getNodeValue("moleculecntincreasetype", type);
	_moleculeCntIncreaseType = static_cast<MoleculeCntIncreaseTypeEnum>(type);
	global_log->info() << "Molecule count increase type: " << _moleculeCntIncreaseType << std::endl;

}

void VectorizationTuner::initOutput(ParticleContainer* /*particleContainer*/,
			DomainDecompBase* /*domainDecomp*/, Domain* /*domain*/) {
	_flopCounterNormalRc = new FlopCounter(_cutoffRadius, _LJCutoffRadius);
	_flopCounterBigRc = new FlopCounter(_cutoffRadiusBig, _LJCutoffRadiusBig);
	_flopCounterZeroRc = new FlopCounter( 0., 0.);
	tune(*(_simulation.getEnsemble()->getComponents()));
	std::vector<std::vector<double>> ownValues {};
	std::vector<std::vector<double>> faceValues {};
	std::vector<std::vector<double>> edgeValues {};
	std::vector<std::vector<double>> cornerValues {};
	//tune(*(_simulation.getEnsemble()->getComponents()), ownValues, faceValues, edgeValues, cornerValues);
}

void VectorizationTuner::writeFile(const TunerLoad& vecs){
		int rank = global_simulation->domainDecomposition().getRank();
		ofstream myfile;
		string resultfile(_outputPrefix + '.' + std::to_string(rank) + ".VT.data");
		global_log->info() << "VT: Writing to file " << resultfile << endl;
		myfile.open(resultfile.c_str(), ofstream::out | ofstream::trunc);
		TunerLoad::write(myfile, vecs);
}

bool VectorizationTuner::readFile(TunerLoad& times) {
		ifstream myfile;
		int rank = global_simulation->domainDecomposition().getRank();
		string resultfile(_outputPrefix + '.' + std::to_string(rank) + ".VT.data");
		global_log->info() << "VT: Reading from file " << resultfile << endl;
		myfile.open(resultfile.c_str(), ifstream::in);
		if(!myfile.good()){
			return false;
		}
		times = TunerLoad::read(myfile);
		return true;
}

using namespace std;

void VectorizationTuner::tune(std::vector<Component> ComponentList) {

	global_log->info() << "VT: begin VECTORIZATION TUNING "<< endl;

    double gflopsOwnBig=0., gflopsPairBig=0., gflopsOwnNormal=0., gflopsPairNormalFace=0., gflopsPairNormalEdge=0., gflopsPairNormalCorner=0.,gflopsOwnZero=0., gflopsPairZero=0.;

    string resultfile(_outputPrefix+".VT.csv");
    global_log->info() << "VT: Writing to file " << resultfile << endl;
    int rank = global_simulation->domainDecomposition().getRank();

    ofstream myfile;
    if (rank == 0) {
    	myfile.open(resultfile.c_str(), ofstream::out | ofstream::trunc);
    	myfile << "Vectorization Tuner File" << endl
			<< "The Cutoff Radii were: " << endl
			<< "NormalRc=" << _cutoffRadius << " , LJCutoffRadiusNormal=" << _LJCutoffRadius << endl
			<< "BigRC=" << _cutoffRadiusBig << " , BigLJCR=" << _LJCutoffRadiusBig << endl;
    }

    if(_moleculeCntIncreaseType==linear or _moleculeCntIncreaseType==both){
    	if (rank==0) {
			myfile << "Linearly distributed molecule counts" << endl;
			myfile << "Num. of Molecules, " << "Gflops for Own BigRc, " << "Gflops for Pair BigRc, " << "Gflops for Own NormalRc, " << "Gflops for Pair NormalRc Face, "
					<< "Gflops for Pair NormalRc Edge, "  << "Gflops for Pair NormalRc Corner, "  << "Gflops for Zero Rc (Own), " << "Gflops for Zero Rc (Pair)" << endl;
    	}
		for(unsigned int i = _minMoleculeCnt; i <= std::min(32u, _maxMoleculeCnt); i++){
			iterate(ComponentList, i,  gflopsOwnBig, gflopsPairBig, gflopsOwnNormal, gflopsPairNormalFace, gflopsPairNormalEdge, gflopsPairNormalCorner, gflopsOwnZero, gflopsPairZero);
			if (rank==0) {
				myfile << i << ", " << gflopsOwnBig << ", " << gflopsPairBig << ", " << gflopsOwnNormal << ", "
						<< gflopsPairNormalFace << ", " << gflopsPairNormalEdge << ", " << gflopsPairNormalCorner << ", "
						<< gflopsOwnZero << ", " << gflopsPairZero << endl;
			}
		}
		if (rank == 0) {
			myfile << endl;
		}
    }
    if(_moleculeCntIncreaseType==exponential or _moleculeCntIncreaseType==both){
    	if (rank == 0) {
			myfile << "Exponentially distributed molecule counts" << endl;
			myfile << "Num. of Molecules," << "Gflops for Own BigRc, " << "Gflops for Pair BigRc, " << "Gflops for Own NormalRc, " << "Gflops for Pair NormalRc Face, "
					<< "Gflops for Pair NormalRc Edge, "  << "Gflops for Pair NormalRc Corner, "  << "Gflops for Zero Rc (Own), " << "Gflops for Zero Rc (Pair)" << endl;
    	// logarithmically scaled axis -> exponentially increasing counts
    	}

    	for(unsigned int i = _minMoleculeCnt; i <= _maxMoleculeCnt; i*=2){
    		iterate(ComponentList, i, gflopsOwnBig, gflopsPairBig, gflopsOwnNormal, gflopsPairNormalFace, gflopsPairNormalEdge, gflopsPairNormalCorner, gflopsOwnZero, gflopsPairZero);
    		if (rank == 0) {
				myfile << i << ", " << gflopsOwnBig << ", " << gflopsPairBig << ", " << gflopsOwnNormal << ", "
									<< gflopsPairNormalFace << ", " << gflopsPairNormalEdge << ", " << gflopsPairNormalCorner << ", "
									<< gflopsOwnZero << ", " << gflopsPairZero << endl;
    		}
    	}
    }

    if (rank == 0) {
    	myfile.close();
    }

    _flopCounterZeroRc->resetCounters();
    _flopCounterBigRc->resetCounters();
    _flopCounterNormalRc->resetCounters();

	global_log->info() << "VECTORIZATION TUNING completed "<< endl;

}

void VectorizationTuner::iterateOwn(long long int numRepetitions,
		ParticleCell& cell, double& gflopsPair, FlopCounter& flopCounter) {
	runOwn(flopCounter, cell, 1);
	// run simulation for a pair of cells
	global_simulation->timers()->start("VECTORIZATION_TUNER_TUNER");
	runOwn(**_cellProcessor, cell, numRepetitions);
	global_simulation->timers()->stop("VECTORIZATION_TUNER_TUNER");
	// get Gflops for pair computations
	double tuningTime = global_simulation->timers()->getTime("VECTORIZATION_TUNER_TUNER");
	gflopsPair = flopCounter.getTotalFlopCount() * numRepetitions / tuningTime / (1024*1024*1024);
	global_log->info() << "FLOP-Count per Iteration: " << flopCounter.getTotalFlopCount() << " FLOPs" << endl;
	global_log->info() << "FLOP-rate: " << gflopsPair << " GFLOPS" << endl;
	global_log->info() << "number of iterations: " << numRepetitions << endl;
	global_log->info() << "total time: " << tuningTime << "s" << endl;
	global_log->info() << "time per iteration: " << tuningTime / numRepetitions << "s " << endl << endl;
	flopCounter.resetCounters();
	global_simulation->timers()->reset("VECTORIZATION_TUNER_TUNER");
}

void VectorizationTuner::iterateOwn (long long int numRepetitions,
		ParticleCell& cell, double& gflops, double& flopCount, double& time, FlopCounter& flopCounter) {
	runOwn(flopCounter, cell, 1);
	// run simulation for a pair of cells
	global_simulation->timers()->start("VECTORIZATION_TUNER_TUNER");
	runOwn(**_cellProcessor, cell, numRepetitions);
	global_simulation->timers()->stop("VECTORIZATION_TUNER_TUNER");
	// get Gflops for pair computations
	double tuningTime = global_simulation->timers()->getTime("VECTORIZATION_TUNER_TUNER");
	gflops = flopCounter.getTotalFlopCount() * numRepetitions / tuningTime / (1024 * 1024 * 1024);
	flopCount = flopCounter.getTotalFlopCount();
	time = tuningTime / numRepetitions;
	//global_log->info() << "flop count per iterations: " << flopCount << endl;
	//global_log->info() << "time per iteration: " << time << "s " << endl;
	flopCounter.resetCounters();
	global_simulation->timers()->reset("VECTORIZATION_TUNER_TUNER");
}

void VectorizationTuner::iteratePair (long long int numRepetitions,
		ParticleCell& firstCell, ParticleCell& secondCell, double& gflops, double& flopCount, double& time, FlopCounter& flopCounter) {
	runPair(flopCounter, firstCell, secondCell, 1);
	// run simulation for a pair of cells
	global_simulation->timers()->start("VECTORIZATION_TUNER_TUNER");
	runPair(**_cellProcessor, firstCell, secondCell, numRepetitions);
	global_simulation->timers()->stop("VECTORIZATION_TUNER_TUNER");
	// get Gflops for pair computations
	double tuningTime = global_simulation->timers()->getTime("VECTORIZATION_TUNER_TUNER");
	gflops = flopCounter.getTotalFlopCount() * numRepetitions / tuningTime / (1024 * 1024 * 1024);
	flopCount = flopCounter.getTotalFlopCount();
	time = tuningTime / numRepetitions;
	//global_log->info() << "flop count per iterations: " << flopCount << endl;
	//global_log->info() << "time per iteration: " << time << "s " << endl;
	flopCounter.resetCounters();
	global_simulation->timers()->reset("VECTORIZATION_TUNER_TUNER");
}

void VectorizationTuner::initCells(ParticleCell& main, ParticleCell& face, ParticleCell& edge, ParticleCell& corner){
		main.assignCellToInnerRegion();
		face.assignCellToInnerRegion();
		edge.assignCellToInnerRegion();
		corner.assignCellToInnerRegion();

	    #ifdef MASKING
	    srand(time(NULL));
	    #else
	    srand(5);//much random, much wow :D
	    #endif

		double BoxMin[3] = {0., 0., 0.};
		double BoxMax[3] = {1., 1., 1.};

		double BoxMinFace[3] = { 1., 0., 0. };
		double BoxMaxFace[3] = { 2., 1., 1. };

		double BoxMinEdge[3] = { 1., 1., 0. };
		double BoxMaxEdge[3] = { 2., 2., 1. };

		double BoxMinCorner[3] = { 1., 1., 1. };
		double BoxMaxCorner[3] = { 2., 2., 2. };

		main.setBoxMin(BoxMin);
		face.setBoxMin(BoxMinFace);
		edge.setBoxMin(BoxMinEdge);
		corner.setBoxMin(BoxMinCorner);

		main.setBoxMax(BoxMax);
		face.setBoxMax(BoxMaxFace);
		edge.setBoxMax(BoxMaxEdge);
		corner.setBoxMax(BoxMaxCorner);
}

void VectorizationTuner::tune(std::vector<Component> componentList, TunerLoad& times, std::vector<int> particleNums, bool generateNewFiles, bool useExistingFiles){

		/*
		 * MPI parallelization strategy:
		 * Every processor measures its own values (but with less iterations) divided by the number of total processors.
		 * After the measurements the values between all processors are allreduced using a sum
		 *
		 * This does not work when there are different processor types present, which is why this option is currently deactivated
		 */
		bool allowMpi = false;

		if(useExistingFiles && readFile(times)){
			global_log->info() << "Read tuner values from file" << endl;
			return;
		} else if(useExistingFiles) {
			global_log->info() << "Couldn't read tuner values from file" << endl;
		}

		global_log->info() << "starting tuning..." << endl;

		//init the cells
		ParticleCell mainCell;
		ParticleCell faceCell;
		ParticleCell edgeCell;
		ParticleCell cornerCell;
		initCells(mainCell, faceCell, edgeCell, cornerCell);

		mardyn_assert(componentList.size() == particleNums.size());

		if(componentList.size() > 2){
			global_log->error_always_output() << "The tuner currently supports only two different particle types!" << endl;
			Simulation::exit(1);
		}

		int maxMols = particleNums.at(0);
		int maxMols2 = particleNums.size() < 2 ? 0 : particleNums.at(1);
		int numProcs = allowMpi ? global_simulation->domainDecomposition().getNumProcs() : 1;

		Component c1 = componentList[0];
		Component c2 = componentList.size() >= 2 ? componentList[1] : c1;

		FlopCounter counter = FlopCounter {1, 1};

		const double restoreCutoff = (**_cellProcessor).getCutoffRadiusSquare();
		const double restoreLJCutoff = (**_cellProcessor).getLJCutoffRadiusSquare();
		(**_cellProcessor).setCutoffRadius(1.);
		(**_cellProcessor).setLJCutoffRadius(1.);
		std::vector<double> ownValues;
		std::vector<double> faceValues;
		std::vector<double> edgeValues;
		std::vector<double> cornerValues;

		ownValues.reserve((maxMols+1)*(maxMols2+1));
		faceValues.reserve((maxMols+1)*(maxMols2+1));
		edgeValues.reserve((maxMols+1)*(maxMols2+1));
		cornerValues.reserve((maxMols+1)*(maxMols2+1));

		for(int numMols1 = 0; numMols1 <= maxMols; numMols1++){
			global_log->info() << numMols1 << " Molecule(s)" << endl;
			for(int numMols2 = 0; numMols2 <= maxMols2; ++numMols2){

				initUniformRandomMolecules(c1, c2, mainCell, numMols1, numMols2);
				initUniformRandomMolecules(c1, c2, faceCell, numMols1, numMols2);
				initUniformRandomMolecules(c1, c2, edgeCell, numMols1, numMols2);
				initUniformRandomMolecules(c1, c2, cornerCell, numMols1, numMols2);

				mainCell.buildSoACaches();
				faceCell.buildSoACaches();
				edgeCell.buildSoACaches();
				cornerCell.buildSoACaches();
				unsigned int weight = numMols1+numMols2;
				long long int numRepetitions;

				numRepetitions = std::max(10000u / std::max(1u, (weight*weight*numProcs)), 10u);

				//the gflops and flopCount are ignored only the time is needed
				double gflops = 0;
				double flopCount = 0;
				double time = 0;
				counter.resetCounters();

				iterateOwn(numRepetitions, faceCell, gflops, flopCount, time, counter);
				ownValues.push_back(time/numProcs);
				clearMolecules(mainCell);
				counter.resetCounters();
				iteratePair(numRepetitions, mainCell, faceCell, gflops, flopCount, time, counter);
				faceValues.push_back(time/numProcs);
				counter.resetCounters();
				iteratePair(numRepetitions, mainCell, edgeCell, gflops, flopCount, time, counter);
				edgeValues.push_back(time/numProcs);
				counter.resetCounters();
				iteratePair(numRepetitions, mainCell, cornerCell, gflops, flopCount, time, counter);
				cornerValues.push_back(time/numProcs);
				counter.resetCounters();

				clearMolecules(mainCell);
				clearMolecules(faceCell);
				clearMolecules(edgeCell);
				clearMolecules(cornerCell);
			}
		}
		//The following ifdefs are only for compilation, since the function is only used in the KDDecomposition which needs mpi anyway
	#ifdef ENABLE_MPI
		if(allowMpi){
			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Allreduce(MPI_IN_PLACE, ownValues.data(), ownValues.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			MPI_Allreduce(MPI_IN_PLACE, faceValues.data(), faceValues.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			MPI_Allreduce(MPI_IN_PLACE, edgeValues.data(), edgeValues.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			MPI_Allreduce(MPI_IN_PLACE, cornerValues.data(), cornerValues.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		}
	#endif
		times = TunerLoad {(maxMols+1), (maxMols2+1), std::move(ownValues), std::move(faceValues), std::move(edgeValues), std::move(cornerValues)};

		if(generateNewFiles){
			writeFile(times);
		}

		(**_cellProcessor).setCutoffRadiusSquare(restoreCutoff);
		(**_cellProcessor).setLJCutoffRadiusSquare(restoreLJCutoff);
		global_log->info() << "finished tuning" << endl;
}


void VectorizationTuner::iteratePair(long long int numRepetitions, ParticleCell& firstCell,
		ParticleCell& secondCell, double& gflopsPair, FlopCounter& flopCounter) {
	//count/calculate the needed flops
	runPair(flopCounter, firstCell, secondCell, 1);
	// run simulation for a pair of cells
	global_simulation->timers()->start("VECTORIZATION_TUNER_TUNER");
	runPair(**_cellProcessor, firstCell, secondCell, numRepetitions);
	global_simulation->timers()->stop("VECTORIZATION_TUNER_TUNER");
	// get Gflops for pair computations
	double tuningTime = global_simulation->timers()->getTime("VECTORIZATION_TUNER_TUNER");
	gflopsPair = flopCounter.getTotalFlopCount() * numRepetitions / tuningTime / (1024 * 1024 * 1024);
	global_log->info() << "FLOP-Count per Iteration: " << flopCounter.getTotalFlopCount() << " FLOPs" << endl;
	global_log->info() << "FLOP-rate: " << gflopsPair << " GFLOPS" << endl;
	global_log->info() << "number of iterations: " << numRepetitions << endl;
	global_log->info() << "total time: " << tuningTime << "s" << endl;
	global_log->info() << "time per iteration: " << tuningTime / numRepetitions << "s " << endl << endl;
	flopCounter.resetCounters();
	global_simulation->timers()->reset("VECTORIZATION_TUNER_TUNER");
}

void VectorizationTuner::iterate(std::vector<Component> ComponentList, unsigned int numMols, double& gflopsOwnBig,
		double& gflopsPairBig, double& /*gflopsOwnNormal*/, double& /*gflopsPairNormalFace*/, double& /*gflopsPairNormalEdge*/,
		double& /*gflopsPairNormalCorner*/, double& gflopsOwnZero, double& gflopsPairZero) {


	// get (first) component
	Component comp = ComponentList[0];

	// construct two cells
	ParticleCell firstCell;
	ParticleCell secondCell;
	firstCell.assignCellToInnerRegion();
	secondCell.assignCellToInnerRegion();

    #ifdef MASKING
    srand(time(NULL));
    #else
    srand(5);//much random, much wow :D
    #endif

	double BoxMin[3] = {0., 0., 0.};
	double BoxMax[3] = {1., 1., 1.};
	double dirxplus[3] = { 1., 0., 0.};
	double BoxMin2[3] = { 1., 0., 0. };
	double BoxMax2[3] = { 2., 1., 1. };

	firstCell.setBoxMin(BoxMin);
	secondCell.setBoxMin(BoxMin2);

	firstCell.setBoxMax(BoxMax);
	secondCell.setBoxMax(BoxMax2);

	//double diryplus[3] = {0., 1., 0.};
	//double dirzplus[3] = {0., 0., 1.};

	global_log->info() << "--------------------------Molecule count: " << numMols << "--------------------------" << endl;

	//initialize both cells with molecules between 0,0,0 and 1,1,1
    initUniformRandomMolecules(comp, firstCell, numMols);
    initUniformRandomMolecules(comp, secondCell, numMols);
    //moveMolecules(dirxplus, secondCell);

	firstCell.buildSoACaches();
	secondCell.buildSoACaches();

	long long int numRepetitions = std::max(20000000u / (numMols*numMols), 10u);


	//0a,0b: 0RC
		(**_cellProcessor).setCutoffRadius(0.);
		(**_cellProcessor).setLJCutoffRadius(0.);
		iterateOwn(numRepetitions, firstCell, gflopsOwnZero, *_flopCounterZeroRc);
		iteratePair(numRepetitions, firstCell, secondCell, gflopsPairZero, *_flopCounterZeroRc);
    //1+2: bigRC
	(**_cellProcessor).setCutoffRadius(_cutoffRadiusBig);
	(**_cellProcessor).setLJCutoffRadius(_LJCutoffRadiusBig);
    //1. own, bigRC
		iterateOwn(numRepetitions, firstCell, gflopsOwnBig, *_flopCounterBigRc);
	//2. pair, bigRC
		iteratePair(numRepetitions, firstCell, secondCell, gflopsPairBig, *_flopCounterBigRc);
#if 0
		TODO: redo these with mesh of molecules
	//3,...: normalRC
	(**_cellProcessor).setCutoffRadius(_cutoffRadius);
	(**_cellProcessor).setLJCutoffRadius(_LJCutoffRadius);
	//3. own, normalRC
		iterateOwn(numRepetitions, firstCell, gflopsOwnNormal, *_flopCounterNormalRc);
	//4. pair, normalRC face
		iteratePair(numRepetitions, firstCell, secondCell, gflopsPairNormalFace, *_flopCounterNormalRc); //cell2s particles moved by 1,0,0 - common face
	//5. pair, normalRC edge
		moveMolecules(diryplus, secondCell);
		iteratePair(numRepetitions, firstCell, secondCell, gflopsPairNormalEdge, *_flopCounterNormalRc); //cell2s particles moved by 1,1,0 - common edge
	//6. pair, normalRC corner
		moveMolecules(dirzplus, secondCell);
		iteratePair(numRepetitions, firstCell, secondCell, gflopsPairNormalCorner, *_flopCounterNormalRc); //cell2s particles moved by 1,1,1 - common corner
#endif

	// clear cells
	clearMolecules(firstCell);
	clearMolecules(secondCell);

}

// returns a uniformly distributed random number between zero and one. (zero, one excluded)
double uniformRandom()
{
  return ( (double)(rand()) + 1. )/( (double)(RAND_MAX) + 2. );
}

 // return a normally distributed random number (uses box-muller transform)
double normalRandom()
{
  double u1=uniformRandom();
  double u2=uniformRandom();
  return cos(8.*atan(1.)*u2)*sqrt(-2.*log(u1)); //box muller transform: cos(2*pi*u2)*sqrt(-2ln(u1))
}


// pass also second cell as argument, when it comes to that
void VectorizationTuner::runOwn(CellProcessor& cp, ParticleCell& cell1, int numRepetitions) {

	cp.initTraversal();

	cp.preprocessCell(cell1);

	for (int i = 0; i < numRepetitions; ++i) {
		cp.processCell(cell1);
	}

	cp.postprocessCell(cell1);
	cp.endTraversal();
}

void VectorizationTuner::runPair(CellProcessor& cp, ParticleCell& cell1, ParticleCell& cell2, int numRepetitions) {


	cp.initTraversal();

	cp.preprocessCell(cell1);
	cp.preprocessCell(cell2);

	for (int i = 0; i < numRepetitions; ++i) {
        cp.processCellPair(cell1, cell2);
	}

	cp.postprocessCell(cell1);
	cp.postprocessCell(cell2);
	cp.endTraversal();

}

// should work also if molecules are initialized via initMeshOfMolecules, initUniformRandomMolecules, initNormalRandomMolecules
void VectorizationTuner::clearMolecules(ParticleCell & cell) {
	cell.deallocateAllParticles();
}


void VectorizationTuner::initMeshOfMolecules(double boxMin[3], double boxMax[3], Component& comp, ParticleCell& cell1, ParticleCell& cell2) {
	//TODO: Bmax, Bmin
	int numMoleculesX = 3, numMoleculesY = 3, numMoleculesZ = 3;

	unsigned long id = 0;
	double vel[3] = { 0.0, 0.0, 0.0 };
	double orientation[4] = { 1.0, 0.0, 0.0, 0.0 }; // NOTE the 1.0 in the first coordinate
	double angularVelocity[3] = { 0.0, 0.0, 0.0 };

	double start_pos1[3] = { 0.0, 0.0, 0.0 }; // adapt for pair (Bmax Bmin)
	double start_pos2[3] = { 1.0, 0.0, 0.0 };
	double pos[3];


	double dx = (boxMax[0] - boxMin[0]) / numMoleculesX;

	for(int z = 0; z < numMoleculesZ; ++z) {
		for(int y = 0; y < numMoleculesY; ++y) {
			for(int x = 0; x < numMoleculesX; ++x) {

				pos[0] = start_pos1[0] + x*dx;
				pos[1] = start_pos1[1] + y*dx;
				pos[2] = start_pos1[2] + z*dx;

				Molecule m = Molecule(
						id, &comp,
						pos[0], pos[1], pos[2],
						vel[0], vel[1], vel[2],
						orientation[0], orientation[1], orientation[2], orientation[3],
						angularVelocity[0], angularVelocity[1], angularVelocity[2]
						);

				cell1.addParticle(m);
				id++; // id's need to be distinct
			}
		}
	}

    id = 0;
	for(int z = 0; z < numMoleculesZ; ++z) {
		for(int y = 0; y < numMoleculesY; ++y) {
			for(int x = 0; x < numMoleculesX; ++x) {

				pos[0] = start_pos2[0] + x*dx;
				pos[1] = start_pos2[1] + y*dx;
				pos[2] = start_pos2[2] + z*dx;

				Molecule m = Molecule(
						id, &comp,
						pos[0], pos[1], pos[2],
						vel[0], vel[1], vel[2],
						orientation[0], orientation[1], orientation[2], orientation[3],
						angularVelocity[0], angularVelocity[1], angularVelocity[2]
						);

				cell2.addParticle(m);
				id++; // id's need to be distinct
			}
		}
	}
}

void VectorizationTuner::initUniformRandomMolecules(Component& comp, ParticleCell& cell, unsigned int numMols) {
	unsigned long id = 0;
	double vel[3] = { 0.0, 0.0, 0.0 };
	double orientation[4] = { 1.0, 0.0, 0.0, 0.0 }; // NOTE the 1.0 in the first coordinate
	double angularVelocity[3] = { 0.0, 0.0, 0.0 };
	double pos[3];

	for(unsigned int i = 0; i < numMols; ++i) {
		pos[0] = cell.getBoxMin(0) + ((double)rand()/(double)RAND_MAX)*(cell.getBoxMax(0) - cell.getBoxMin(0));
		pos[1] = cell.getBoxMin(1) + ((double)rand()/(double)RAND_MAX)*(cell.getBoxMax(1) - cell.getBoxMin(1));
		pos[2] = cell.getBoxMin(2) + ((double)rand()/(double)RAND_MAX)*(cell.getBoxMax(2) - cell.getBoxMin(2));

		Molecule m = Molecule(
				id, &comp,
				pos[0], pos[1], pos[2],
				vel[0], vel[1], vel[2],
				orientation[0], orientation[1], orientation[2], orientation[3],
				angularVelocity[0], angularVelocity[1], angularVelocity[2]
				);
		cell.addParticle(m);
		id++; // id's need to be distinct
		//global_log->info() << pos[0] << " " << pos[1] << " " << pos[2] << endl;
	}
}

void VectorizationTuner::initUniformRandomMolecules(Component& comp1, Component& comp2, ParticleCell& cell, unsigned int numMolsFirst, unsigned int numMolsSecond) {
	unsigned long id = 0;
	double vel[3] = { 0.0, 0.0, 0.0 };
	double orientation[4] = { 1.0, 0.0, 0.0, 0.0 }; // NOTE the 1.0 in the first coordinate
	double angularVelocity[3] = { 0.0, 0.0, 0.0 };
	double pos[3];

	for(int c = 0; c < 2; ++c){
		const unsigned int numMols = c==0 ? numMolsFirst : numMolsSecond;
		Component *comp = c == 0 ? &comp1 : &comp2;

		for(unsigned int i = 0; i < numMols; ++i) {
			pos[0] = cell.getBoxMin(0) + ((double)rand()/(double)RAND_MAX)*(cell.getBoxMax(0) - cell.getBoxMin(0));
			pos[1] = cell.getBoxMin(1) + ((double)rand()/(double)RAND_MAX)*(cell.getBoxMax(1) - cell.getBoxMin(1));
			pos[2] = cell.getBoxMin(2) + ((double)rand()/(double)RAND_MAX)*(cell.getBoxMax(2) - cell.getBoxMin(2));

			Molecule m = Molecule(
					id, comp,
					pos[0], pos[1], pos[2],
					vel[0], vel[1], vel[2],
					orientation[0], orientation[1], orientation[2], orientation[3],
					angularVelocity[0], angularVelocity[1], angularVelocity[2]
			);
			cell.addParticle(m);
			id++; // id's need to be distinct
			//global_log->info() << pos[0] << " " << pos[1] << " " << pos[2] << endl;
		}
	}
}


void VectorizationTuner::initNormalRandomMolecules(double /*boxMin*/[3], double /*boxMax*/[3], Component& comp,
		ParticleCell& cell1, ParticleCell& /*cell2*/, unsigned int numMols) {
//TODO: currently only cell 1
//TODO: does not really have/need/can_use Bmax, Bmin - is normal dist. proper???

	unsigned long id = 0;
	double vel[3] = { 0.0, 0.0, 0.0 };
	double orientation[4] = { 1.0, 0.0, 0.0, 0.0 }; // NOTE the 1.0 in the first coordinate
	double angularVelocity[3] = { 0.0, 0.0, 0.0 };

	double start_pos[3] = { 0.0, 0.0, 0.0 }; // adapt for pair (Bmax Bmin)
	double pos[3];

//	double dx = boxMax[0] - boxMin[0] / numMoleculesX;


	for(unsigned int i = 0; i < numMols; ++i) {


		pos[0] = normalRandom();
		pos[1] = start_pos[1] + normalRandom();
		pos[2] = start_pos[2] + normalRandom();

		Molecule m = Molecule(
				id, &comp,
				pos[0], pos[1], pos[2],
				vel[0], vel[1], vel[2],
				orientation[0], orientation[1], orientation[2], orientation[3],
				angularVelocity[0], angularVelocity[1], angularVelocity[2]
				);

		cell1.addParticle(m);
		id++; // id's need to be distinct
	}
}

void VectorizationTuner::moveMolecules(double direction[3], ParticleCell& cell){
	SingleCellIterator begin = cell.iteratorBegin();
	SingleCellIterator end = cell.iteratorEnd();

	for(SingleCellIterator it = begin; it != end; ++it ) {
		Molecule& mol = *it;
		mol.move(0, direction[0]);
		mol.move(1, direction[1]);
		mol.move(2, direction[2]);
		//global_log->info() << mol->r(0) << " " << mol->r(1) << " " << mol->r(2) << endl;
	}
}
