/*
 * Events.h
 *
 *  Created on: 19 Jul 2018
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

//define the precision to store jiffy count
using JiffyPrecision = unsigned int;
//define the precision to store the time stamp. affects time quantization.
using TimeStampPrecision = unsigned int;

Events::Events() {

}

void Events::init(ParticleContainer* particleContainer,
        DomainDecompBase* domainDecomp, Domain* domain) {
    _addEvent(domainDecomp->getRank(), EVENT_INIT);

    std::stringstream fname;
    fname << "events." << std::setfill('0') << std::setw(6) << domainDecomp->getRank() << ".dat";
    std::ofstream events;
    events.open(fname.str(), std::ios::trunc);
    events.close();
}

void Events::readXML(XMLfileUnits& xmlconfig) {
    // _snapshotInterval = 20;
    // xmlconfig.getNodeValue("snapshotInterval", _snapshotInterval);
    // global_log->info() << "    ISM: Snapshot interval: "
    //         << _snapshotInterval << std::endl;
    // _connectionName.assign("tcp://127.0.0.1:33333");
    // xmlconfig.getNodeValue("connectionName", _connectionName);
    // global_log->info() << "    ISM: Ping to Megamol on: <" 
    //         << _connectionName << ">" << std::endl;
    // _replyBufferSize = 16384;
    // xmlconfig.getNodeValue("replyBufferSize", _replyBufferSize);
    // global_log->info() << "    ISM: Megamol reply buffer size (defaults to 16384 byte): "
    //         << _replyBufferSize << std::endl;
    // _syncTimeout = 10;
    // xmlconfig.getNodeValue("syncTimeout", _syncTimeout);
    // global_log->info() << "    ISM: Synchronization timeout (s): "
    //         << _syncTimeout << std::endl;
    // _ringBufferSize = 5;
    // xmlconfig.getNodeValue("ringBufferSize", _ringBufferSize);
    // global_log->info() << "    ISM: Ring buffer size: "
    //         << _ringBufferSize << std::endl;
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
    std::stringstream fname;
    fname << "events." << std::setfill('0') << std::setw(6) << rank << ".dat";
    unsigned int timeStampQuantized = ResProbe<JiffyPrecision, TimeStampPrecision>::getQuantizedTime();
    _eventData.push_back(timeStampQuantized);
    _eventData.push_back(event);
    if ((_eventCount % 100) == 0) {
        global_log->info() << "    ISM: Dumping events, total event count: " << _eventCount << std::endl;
        std::ofstream events;
        events.open(fname.str(), std::ios::ate);
        events.write(reinterpret_cast<char*>(_eventData.data()), sizeof(unsigned int)*_eventData.size());
        events.close();
        _eventData.clear();
    }
    ++_eventCount;
}
