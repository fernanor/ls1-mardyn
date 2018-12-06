/*
 * Events.h
 *
 *  Created on: 4. Dec 2018
 *      Author: Oliver Fernandes
 */

///
/// \file Events.h
/// Events. Drops out enumerated events to a file with a time stamps
///

#ifndef SRC_PLUGINS_EVENTS_H_
#define SRC_PLUGINS_EVENTS_H_

#include "PluginBase.h"
#include "molecules/MoleculeForwardDeclaration.h"

#include <cstdlib>
#include <chrono>
#include <string>
#include <sstream>
#include <iomanip>
#include <vector>
#include <memory>

/**************TIME PROBE SHIT***************/
#include "resprobe.h"
unsigned int const EVENT_INIT=0;
unsigned int const EVENT_BEFOREEVENTNEWTIMESTEP=1;
unsigned int const EVENT_BEFOREFORCES=2;
unsigned int const EVENT_AFTERFORCES=3;
unsigned int const EVENT_ENDSTEP=4;

/*************remove when done***************/

class Events : public PluginBase {
public:
    Events(); 
    virtual ~Events() {};

    void init(ParticleContainer* particleContainer,
            DomainDecompBase* domainDecomp, Domain* domain
    );

    void readXML(XMLfileUnits& xmlconfig);

    void beforeEventNewTimestep(
            ParticleContainer* particleContainer, DomainDecompBase* domainDecomp,
            unsigned long simstep
    );

    void beforeForces(
            ParticleContainer* particleContainer, DomainDecompBase* domainDecomp,
            unsigned long simstep
    );

    void afterForces(
            ParticleContainer* particleContainer, DomainDecompBase* domainDecomp,
            unsigned long simstep
    );

    void endStep(
            ParticleContainer* particleContainer, DomainDecompBase* domainDecomp,
            Domain* domain, unsigned long simstep
    );
    
    void finish(ParticleContainer* particleContainer,
            DomainDecompBase* domainDecomp, Domain* domain
    ) {}

    std::string getPluginName() {
        return std::string("Events");
    }

    static PluginBase* createInstance() { return new Events(); }

private:
    // plugin-internal timer
    // stores pairs of time_stamp and event for comparison to resource probe
    std::vector<unsigned int> _eventData;
    unsigned int _eventCount = 0;
    unsigned int _dumpInterval;
    std::string _fnamePrefix;
    std::stringstream _fname;
    void _addEvent(int rank, unsigned int event);
};
#endif /* SRC_PLUGINS_EVENTS_H_ */
