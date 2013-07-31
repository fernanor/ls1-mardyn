#ifndef MMSPDWRITER_H_
#define MMSPDWRITER_H_

// by Stefan Becker <stefan.becker@mv.uni-kl.de>
//output writer wirting the output in the *.mmspd file format required for MegaMol
// a detailed documentation of this file format can be obtained from the MegaMol documentation (see website)

#include "io/OutputBase.h"
#include "ensemble/GrandCanonical.h"
#include <string>

class ParticleContainer;
class DomainDecompBase; 
class Domain;

class MmspdWriter : public OutputBase{
  public:
	//! @brief: writes a mmspd file used by MegaMol
	//!
	//! Depending on write frequency (for example: every timestep, or every 10th, 100th, 1000th ...) number of frames
	//! can be controlled. The *.mmspd-file can be visualized by visualization software like MegaMol.
	//! (for detail information visit: https://svn.vis.uni-stuttgart.de/trac/megamol/)
	//!
	//! @param filename Name of the *.mmspd-file (including path)
	//! @param particleContainer The molecules that have to be written to the file are stored here
	//! @param domainDecomp In the parallel version, the file has to be written by more than one process.
	//!                     Methods to achieve this are available in domainDecomp
	//! @param writeFrequency Controls the frequency of writing out the data (every timestep, every 10th, 100th, ... timestep)
    MmspdWriter(unsigned long writeFrequency, std::string filename, unsigned long numberOfTimesteps, bool incremental);
	~MmspdWriter();
    
    void initOutput(ParticleContainer* particleContainer,
			DomainDecompBase* domainDecomp, Domain* domain);
	//! @todo comment
    void doOutput( ParticleContainer* particleContainer,
		   DomainDecompBase* domainDecomp, Domain* domain,
		   unsigned long simstep, std::list<ChemicalPotential>* lmu
	);
	//! @todo comment
    void finishOutput( ParticleContainer* particleContainer,
		       DomainDecompBase* domainDecomp, Domain* domain);
  private:
      std::string _filename;
      unsigned long _numberOfTimesteps;
      unsigned long _writeFrequency;
      bool _filenameisdate;
      bool _incremental;
};

#endif /*MMSPDWRITER_H_*/
