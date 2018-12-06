/*
 * Events.cpp
 *
 *  Created on: 4. Dec 2018
 *      Author: Oliver Fernandes
 */

#include "Events.h"
#include "utils/xmlfileUnits.h"
#include "utils/Logger.h"
#include "molecules/Molecule.h"
#include "particleContainer/ParticleContainer.h"
#include "Simulation.h"
#include "Domain.h"
#include "parallel/DomainDecompBase.h"

using Log::global_log;
using RP = UIntResProbe;

Events::Events() {

}

void Events::init(ParticleContainer* particleContainer,
        DomainDecompBase* domainDecomp, Domain* domain) {
    std::string const nodeName(std::getenv("NODE_HOSTNAME"));
    global_log->info() << "    EVENTS: File name rank tag: "
            << nodeName << std::endl;
    _fname << _fnamePrefix << "." << nodeName << ".dat";
    std::ofstream events;
    events.open(_fname.str(), std::ios::trunc);
    events.close();
    _addEvent(domainDecomp->getRank(), EVENT_INIT);
}

void Events::readXML(XMLfileUnits& xmlconfig) {
    _dumpInterval = 100;
    xmlconfig.getNodeValue("dumpInterval", _dumpInterval);
    global_log->info() << "    EVENTS: Dump Interval: "
            << _dumpInterval << std::endl;
    _fnamePrefix.assign("events");
    xmlconfig.getNodeValue("fnamePrefix", _fnamePrefix);
    global_log->info() << "    EVENTS: File name prefix: "
            << _fnamePrefix << std::endl;
}

void Events::beforeEventNewTimestep(
        ParticleContainer* particleContainer, DomainDecompBase* domainDecomp,
        unsigned long simstep) {
    _addEvent(domainDecomp->getRank(), EVENT_BEFOREEVENTNEWTIMESTEP);
}

void Events::beforeForces(
        ParticleContainer* particleContainer, DomainDecompBase* domainDecomp,
        unsigned long simstep) {
    _addEvent(domainDecomp->getRank(), EVENT_BEFOREFORCES);
}

void Events::afterForces(
        ParticleContainer* particleContainer, DomainDecompBase* domainDecomp,
        unsigned long simstep) {
    _addEvent(domainDecomp->getRank(), EVENT_AFTERFORCES);
}

void Events::endStep(ParticleContainer* particleContainer,
        DomainDecompBase* domainDecomp, Domain* domain, unsigned long simstep) {
    _addEvent(domainDecomp->getRank(), EVENT_ENDSTEP);
}

void Events::_addEvent(int rank, unsigned int event) {
    unsigned int timeStampQuantized = RP::getQuantizedTime();
    _eventData.push_back(timeStampQuantized);
    _eventData.push_back(event);
    if ((_eventCount % _dumpInterval) == 0) {
        global_log->info() << "    EVENTS: Dumping events, total event count: " << _eventCount << std::endl;
        std::ofstream events;
        events.open(_fname.str(), std::ios::app);
        events.write(reinterpret_cast<char*>(_eventData.data()), sizeof(unsigned int)*_eventData.size());
        events.close();
        _eventData.clear();
    }
    ++_eventCount;
}
