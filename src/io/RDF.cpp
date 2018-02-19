#include "io/RDF.h"

#include "Domain.h"
#include "molecules/Component.h"
#include "parallel/DomainDecompBase.h"
#include "Simulation.h"
#include "utils/Logger.h"

#include <sstream>
#include <fstream>
#include <map>

using namespace std;
using namespace Log;

RDF::RDF() :
	_intervalLength(0),
	_bins(0),
	_numberOfComponents(0),
	_components(_simulation.getEnsemble()->getComponents()),
	_numberOfRDFTimesteps(0),
	_accumulatedNumberOfRDFTimesteps(0),
	_maxDistanceSquare(0.0),
	_doCollectSiteRDF(false),
	_writeFrequency(1),
	_outputPrefix("ls1-mardyn"),
	_initialized(false),
	_readConfig(false)
{}

void RDF::init() {
	if(!_readConfig){
		global_log->error() << "RDF initialized without reading the configuration, exiting" << std::endl;
		Simulation::exit(25);
	}
	_numberOfComponents = _components->size();
	_doCollectSiteRDF = false;
	_numberOfRDFTimesteps = 0;
	_accumulatedNumberOfRDFTimesteps = 0;
	_maxDistanceSquare = _intervalLength*_intervalLength*_bins*_bins;


	{
		const unsigned int nC = _numberOfComponents;

		resizeExactly(_globalCtr, nC);
		resizeExactly(_globalAccumulatedCtr, nC);
		resizeExactly(_distribution.local, nC);
		resizeExactly(_distribution.global, nC);
		resizeExactly(_globalAccumulatedDistribution, nC);

		resizeExactly(_siteDistribution.local, nC);
		resizeExactly(_siteDistribution.global, nC);
		resizeExactly(_globalAccumulatedSiteDistribution, nC);
	}


	for(unsigned i = 0; i < _numberOfComponents; i++) {
		_globalCtr[i] = 0;
		_globalAccumulatedCtr[i] = 0;

		{
			const unsigned int nCmi = _numberOfComponents - i;

			resizeExactly(_distribution.local[i], nCmi);
			resizeExactly(_distribution.global[i], nCmi);
			resizeExactly(_globalAccumulatedDistribution[i], nCmi);

			resizeExactly(_siteDistribution.local[i], nCmi);
			resizeExactly(_siteDistribution.global[i], nCmi);
			resizeExactly(_globalAccumulatedSiteDistribution[i], nCmi);
		}

		unsigned ni = (*_components)[i].numSites();

		for(unsigned k=0; i+k < _numberOfComponents; k++) {
			resizeExactly(_distribution.local[i][k], _bins);
			resizeExactly(_distribution.global[i][k], _bins);
			resizeExactly(_globalAccumulatedDistribution[i][k], _bins);

			for(unsigned l=0; l < _bins; l++) {
				_distribution.local[i][k][l] = 0;
				_distribution.global[i][k][l] = 0;
				_globalAccumulatedDistribution[i][k][l] = 0;
			}

			unsigned nj = (*_components)[i+k].numSites();
			if(ni+nj > 2) {
				_doCollectSiteRDF = true;

				resizeExactly(_siteDistribution.local[i][k], ni);
				resizeExactly(_siteDistribution.global[i][k], ni);
				resizeExactly(_globalAccumulatedSiteDistribution[i][k], ni);

				for(unsigned m=0; m < ni; m++) {
					resizeExactly(_siteDistribution.local[i][k][m], nj);
					resizeExactly(_siteDistribution.global[i][k][m], nj);
					resizeExactly(_globalAccumulatedSiteDistribution[i][k][m], nj);
					for(unsigned n=0; n < nj; n++) {
						resizeExactly(_siteDistribution.local[i][k][m][n], _bins);
						resizeExactly(_siteDistribution.global[i][k][m][n], _bins);
						resizeExactly(_globalAccumulatedSiteDistribution[i][k][m][n], _bins);
						for(unsigned l=0; l < _bins; l++) {
							_siteDistribution.local[i][k][m][n][l] = 0;
							_siteDistribution.global[i][k][m][n][l] = 0;
							_globalAccumulatedSiteDistribution[i][k][m][n][l] = 0;
							// cout << "init " << i << "\t" << k << "\t" << m << "\t" << n << "\t" << l << "\n";
						}
					}
				}
			}
		}
	}
	_initialized=true;
}


void RDF::readXML(XMLfileUnits& xmlconfig) {
	_writeFrequency = 1;
	xmlconfig.getNodeValue("writefrequency", _writeFrequency);
	global_log->info() << "Write frequency: " << _writeFrequency << endl;

	_outputPrefix = "mardyn";
	xmlconfig.getNodeValue("outputprefix", _outputPrefix);
	global_log->info() << "Output prefix: " << _outputPrefix << endl;

	_bins = 1;
	xmlconfig.getNodeValue("bins", _bins);
	global_log->info() << "Number of bins: " << _bins << endl;

	_intervalLength = 1;
	xmlconfig.getNodeValueReduced("intervallength", _intervalLength);
	global_log->info() << "Interval length: " << _intervalLength << endl;
	_readConfig = true;
}

void RDF::initOutput(ParticleContainer* /*particleContainer*/, DomainDecompBase* /*domainDecomp*/, Domain* /*domain*/) {
	init();
}

void RDF::finishOutput(ParticleContainer* /*particleContainer*/, DomainDecompBase* /*domainDecomp*/, Domain* /*domain*/) {
}



RDF::~RDF() {
	// nothing to do since refactoring to vectors
}

void RDF::accumulateNumberOfMolecules(vector<Component>& components) {
	for (size_t i = 0; i < components.size(); i++) {
		_globalCtr[i] += components[i].getNumMolecules();
	}
}


void RDF::accumulateRDF() {
	if(0 >= _numberOfRDFTimesteps) return;
	_accumulatedNumberOfRDFTimesteps += _numberOfRDFTimesteps;
	for(unsigned i=0; i < _numberOfComponents; i++) {
		_globalAccumulatedCtr[i] += _globalCtr[i];
		unsigned ni = (*_components)[i].numSites();
		for(unsigned k=0; i+k < _numberOfComponents; k++) {
			unsigned nj = (*_components)[i+k].numSites();
			for(unsigned l=0; l < _bins; l++) {
				_globalAccumulatedDistribution[i][k][l] += _distribution.global[i][k][l];
				if(ni + nj > 2) {
					for(unsigned m=0; m < ni; m++) {
						for(unsigned n=0; n < nj; n++) {
							_globalAccumulatedSiteDistribution[i][k][m][n][l] += _siteDistribution.global[i][k][m][n][l];
						}
					}
				}
			}
		}
	}
}

void RDF::collectRDF(DomainDecompBase* dode) {
	// Communicate component-component RDFs
	dode->collCommInit(_bins * _numberOfComponents * (_numberOfComponents+1)/2);

	for(unsigned i=0; i < _numberOfComponents; i++) {
		for(unsigned k=0; i+k < _numberOfComponents; k++) {
			for(unsigned l=0; l < _bins; l++) {
				dode->collCommAppendUnsLong(_distribution.local[i][k][l]);
			}
		}
	}

	dode->collCommAllreduceSum();
	for(unsigned i=0; i < _numberOfComponents; i++) {
		for(unsigned k=0; i+k < _numberOfComponents; k++) {
			for(unsigned l=0; l < _bins; l++) {
				_distribution.global[i][k][l] = dode->collCommGetUnsLong();
			}
		}
	}
	dode->collCommFinalize();

	// communicate site-site RDFs
	for(unsigned i=0; i < _numberOfComponents; i++) {
		unsigned ni = (*_components)[i].numSites();
		for(unsigned k=0; i+k < _numberOfComponents; k++) {
			unsigned nj = (*_components)[i+k].numSites();
			if(ni+nj > 2) {
				dode->collCommInit(_bins * ni * nj);
				for(unsigned l=0; l < _bins; l++) {
					for(unsigned m=0; m < ni; m++) {
						for(unsigned n=0; n < nj; n++) {
							dode->collCommAppendUnsLong(_siteDistribution.local[i][k][m][n][l]);
						}
					}
				}

				dode->collCommAllreduceSum();

				for(unsigned l=0; l < _bins; l++) {
					for(unsigned m=0; m < ni; m++) {
						for(unsigned n=0; n < nj; n++) {
							_siteDistribution.global[i][k][m][n][l] = dode->collCommGetUnsLong();
						}
					}
				}
				dode->collCommFinalize();

			}
		}
	}
}


void RDF::reset() {
	_numberOfRDFTimesteps = 0;
	for(unsigned i=0; i < _numberOfComponents; i++) {
		_globalCtr[i] = 0;

		unsigned ni = (*_components)[i].numSites();
		for(unsigned k=0; i+k < _numberOfComponents; k++) {
			unsigned nj = (*_components)[i+k].numSites();
			for(unsigned l=0; l < _bins; l++) {
				_distribution.local[i][k][l] = 0;
				_distribution.global[i][k][l] = 0;

				if(ni+nj > 2) {
					for(unsigned m=0; m < ni; m++) {
						for(unsigned n=0; n < nj; n++) {
							_siteDistribution.local[i][k][m][n][l] = 0;
							_siteDistribution.global[i][k][m][n][l] = 0;
						}
					}
				}
			}
		}
	}
}


void RDF::doOutput(ParticleContainer* /*particleContainer*/, DomainDecompBase* domainDecomposition, Domain* domain,
		unsigned long simStep, std::list<ChemicalPotential>* /*lmu*/, map<unsigned, CavityEnsemble>* /*mcav*/) {
	if(_numberOfRDFTimesteps <= 0) return;

	if((simStep > 0) && (simStep % _writeFrequency == 0)) {
		collectRDF(domainDecomposition);
		
		if( domainDecomposition->getRank() == 0 ) {
			accumulateRDF();
			for (unsigned i = 0; i < _numberOfComponents; i++) {
				for (unsigned j = i; j < _numberOfComponents; j++) {
					ostringstream osstrm;
					osstrm << _outputPrefix << "_" << i << "-" << j << ".";
					osstrm.fill('0');
					osstrm.width(9);
					osstrm << std::right << simStep << ".rdf";
					writeToFile(domain, osstrm.str(), i, j);
				}
			}
		}
		reset();
	}
}


void RDF::writeToFile(const Domain* domain, std::string filename, unsigned i, unsigned j) const {
	ofstream rdfout(filename);
	if( rdfout.fail() ) {
		global_log->error() << "[RDF] Failed opening output file '" << filename << "'" << endl;
		return;
	}

	global_log->debug() << "[RDF] Writing output" << endl;
	unsigned ni = (*_components)[i].numSites();
	unsigned nj = (*_components)[j].numSites();

	double V = domain->getGlobalVolume();
	double N_i = _globalCtr[i] / (double)_numberOfRDFTimesteps;
	double N_Ai = _globalAccumulatedCtr[i] / (double)_accumulatedNumberOfRDFTimesteps;
	double N_j = _globalCtr[j] / (double)_numberOfRDFTimesteps;
	double N_Aj = _globalAccumulatedCtr[j] / (double)_accumulatedNumberOfRDFTimesteps;
	double rho_i = N_i / V;
	double rho_Ai = N_Ai / V;
	double rho_j = N_j / V;
	double rho_Aj = N_Aj / V;

	std::map< unsigned, std::map<unsigned, double> > Nsite_pair_int, Nsite_Apair_int;
	rdfout.precision(5);
	rdfout << "# r\tcurr.{loc, int}\taccu.{loc, int}\t\tdV\tNpair(curr.)\tNpair(accu.)\t\tnorm(curr.)\tnorm(accu.)";
	if(ni+nj > 2) {
		for(unsigned m=0; m < ni; m++) {
			rdfout << "\t";
			for(unsigned n=0; n < nj; n++) {
				Nsite_pair_int[m][n] = 0.0;
				Nsite_Apair_int[m][n] = 0.0;
				rdfout << "\t(" << m << ", " << n << ")_curr{loc, int}   (" << m << ", " << n << ")_accu{loc, int}";
			}
		}
	}
	rdfout << "\n";
	rdfout << "# \n# ctr_i: " << _globalCtr[i] << "\n# ctr_j: " << _globalCtr[j]
	       << "\n# V: " << V << "\n# _universalRDFTimesteps: " << _numberOfRDFTimesteps
	       << "\n# _universalAccumulatedTimesteps: " << _accumulatedNumberOfRDFTimesteps
	       << "\n# rho_i: " << rho_i << " (acc. " << rho_Ai << ")"
	       << "\n# rho_j: " << rho_j << " (acc. " << rho_Aj << ")"
	       << "\n# \n";

	// new or alternative heading
	rdfout << "#r\trdf\trdf_integral}\tacc_rdf acc_rdf_integral}\t\tdV\tNpair(curr.)\tNpair(accu.)\t\tnormalization(curr.)\tnormalization(accu.)";
	if(ni+nj > 2) {
		for(unsigned m=0; m < ni; m++) {
			rdfout << "\t";
			for(unsigned n=0; n < nj; n++) {
				Nsite_pair_int[m][n] = 0.0;
				Nsite_Apair_int[m][n] = 0.0;
				rdfout << "\t(" << m << "," << n << ")_curr{rdf, rdf_integral}   (" << m << "," << n << ")_accu{rdf, rdf_integral}";
			}
		}
	}
	rdfout << "\n";
    // end

	double N_pair_int = 0.0;
	double N_Apair_int = 0.0;
	for(unsigned int l = 0; l < numBins(); ++l) {
		double rmin = l * binwidth();
		double rmid = (l+0.5) * binwidth();
		double rmax = (l+1.0) * binwidth();
		double r3min = rmin*rmin*rmin;
		double r3max = rmax*rmax*rmax;
		double dV = (4.0 / 3.0) * M_PI * (r3max - r3min);

		double N_pair = _distribution.global[i][j-i][l] / (double)_numberOfRDFTimesteps;
		N_pair_int += N_pair;
		double N_Apair = _globalAccumulatedDistribution[i][j-i][l] / (double)_accumulatedNumberOfRDFTimesteps;
		N_Apair_int += N_Apair;
		double N_pair_norm = 0.0;
		double N_Apair_norm = 0.0;
		double N_pair_int_norm = 0.0;
		double N_Apair_int_norm = 0.0;

		if(i == j) {
			N_pair_norm = 0.5*N_i*(N_i-1.0) * dV/V;
			N_Apair_norm = 0.5*N_Ai*(N_Ai-1.0) * dV/V;
			N_pair_int_norm = 0.5*N_i*(N_i-1.0) * 4.1887902*r3max/V;
			N_Apair_int_norm = 0.5*N_Ai*(N_Ai-1.0) * 4.1887902*r3max/V;
		}
		else {
			N_pair_norm = N_i*N_j * dV/V;
			N_Apair_norm = N_Ai*N_Aj * dV/V;
			N_pair_int_norm = N_i*N_j * 4.1887902*r3max/V;
			N_Apair_int_norm = N_Ai*N_Aj * 4.1887902*r3max/V;
		}

		rdfout << rmid << "\t" << N_pair/N_pair_norm << " " << N_pair_int/N_pair_int_norm
				<< "\t" << N_Apair/N_Apair_norm << " " << N_Apair_int/N_Apair_int_norm
				<< "\t\t" << dV << "\t" << N_pair << "\t" << N_Apair
				<< "\t\t" << N_pair_norm << "\t" << N_Apair_norm;

		if(ni+nj > 2) {
			for(unsigned m=0; m < ni; m++) {
				rdfout << "\t";
				for(unsigned n=0; n < nj; n++) {
					double p = _siteDistribution.global[i][j-i][m][n][l] / (double)_numberOfRDFTimesteps;
					Nsite_pair_int[m][n] += p;
					double ap = _globalAccumulatedSiteDistribution[i][j-i][m][n][l] / (double)_accumulatedNumberOfRDFTimesteps;
					Nsite_Apair_int[m][n] += ap;
					rdfout << "\t" << p/N_pair_norm << " " << Nsite_pair_int[m][n]/N_pair_int_norm
					       << "   " << ap/N_Apair_norm << " " << Nsite_Apair_int[m][n]/N_Apair_int_norm;
				}
			}
		}
		rdfout << "\n";
	}
	rdfout.close();
}
