/*
 * NeighbourCommunicationScheme.h
 *
 *  Created on: Sep 29, 2016
 *      Author: seckler
 */

#pragma once

#include "parallel/CommunicationPartner.h"

#include <vector>

class DomainDecompMPIBase;
class Domain;
class ZonalMethod;
class HaloRegion;
class NeighbourCommunicationScheme {
	friend class NeighbourCommunicationSchemeTest;
	
public:
	/**
	 * Specifies the amount of sequential communication steps needed for the communication scheme.
	 * This is also the outer size of DomainDecompMPIBase::_neighbours
	 * @return
	 */
	unsigned int getCommDims() {
		return _commDimms;
	}
	NeighbourCommunicationScheme() = delete;
	NeighbourCommunicationScheme(unsigned int commDimms, ZonalMethod* zonalMethod, bool pushPull);

	virtual ~NeighbourCommunicationScheme();

	NeighbourCommunicationScheme(NeighbourCommunicationScheme const &) = delete;
	void operator=(NeighbourCommunicationScheme const &other) = delete;

	virtual void prepareNonBlockingStageImpl(ParticleContainer* moleculeContainer, Domain* domain,
			unsigned int stageNumber, MessageType msgType, bool removeRecvDuplicates,
			DomainDecompMPIBase* domainDecomp) = 0;

	virtual void finishNonBlockingStageImpl(ParticleContainer* moleculeContainer, Domain* domain,
			unsigned int stageNumber, MessageType msgType, bool removeRecvDuplicates,
			DomainDecompMPIBase* domainDecomp) = 0;

	virtual void exchangeMoleculesMPI(ParticleContainer* moleculeContainer, Domain* domain, MessageType msgType,
			bool removeRecvDuplicates, DomainDecompMPIBase* domainDecomp) = 0;

	void setCoverWholeDomain(unsigned int d, bool covers) {
		_coversWholeDomain[d] = covers;
	}

	virtual void initCommunicationPartners(double cutoffRadius, Domain * domain,
			DomainDecompMPIBase* domainDecomp,
			ParticleContainer* moleculeContainer) = 0;

	virtual std::vector<int> get3StageNeighbourRanks() = 0;

	virtual std::vector<int> getFullShellNeighbourRanks() {
		std::vector<int> neighbourRanks;
		for (unsigned int i = 0; i < _fullShellNeighbours.size(); i++) {
			neighbourRanks.push_back(_fullShellNeighbours[i].getRank());
		}
		return neighbourRanks;
	}


	virtual size_t getDynamicSize() {
		size_t totSize = 0;
		// _fullShellNeighbours
		totSize += sizeof(*this);
		//std::cout << "pre FSN:" << totSize;
		totSize += _fullShellNeighbours.capacity() * sizeof(CommunicationPartner);
		for (CommunicationPartner& neigh : _fullShellNeighbours) {
			totSize += neigh.getDynamicSize();
			//std::cout << "FSN:" << neigh.getDynamicSize();
		}
		//std::cout << "post FSN/pre neigh:" << totSize;
		totSize += (*_neighbours).capacity() * sizeof(CommunicationPartner);
		for (auto& neighList : (*_neighbours)) {
			for (auto& neigh : neighList) {
				totSize += neigh.getDynamicSize();
				//std::cout << "Neigh:" << neigh.getDynamicSize();
			}
		}
		//std::cout << "post Neigh:" << totSize;
		return totSize;
	}

	void printCommunicationPartners(std::string filename) const;

protected:

	//! vector of neighbours. The first dimension should be of size getCommDims().
	std::vector<std::vector<CommunicationPartner>> *_neighbours;
	
	// -------------------------------------------------------------------------
	std::vector<std::vector<CommunicationPartner>> *_haloExportForceImportNeighbours;
	std::vector<std::vector<CommunicationPartner>> *_haloImportForceExportNeighbours;
	std::vector<std::vector<CommunicationPartner>> *_leavingExportNeighbours;
	std::vector<std::vector<CommunicationPartner>> *_leavingImportNeighbours;
	
	void selectNeighbours(MessageType msgType, bool import);
	// -------------------------------------------------------------------------

	//! flag, which tells whether a processor covers the whole domain along a dimension
	//! if true, we will use the methods provided by the base class for handling the
	//! respective dimension, instead of packing and unpacking messages to self
	bool _coversWholeDomain[3];

	unsigned int _commDimms;

	//! zonal method (FullShell, HalfShell, ...)
	ZonalMethod* _zonalMethod;

	//! list of all neighbours (non-squeezed)
	std::vector<CommunicationPartner> _fullShellNeighbours;

	bool _pushPull;
};

class DirectNeighbourCommunicationScheme: public NeighbourCommunicationScheme {
	friend class NeighbourCommunicationSchemeTest;
public:
	DirectNeighbourCommunicationScheme(ZonalMethod* zonalMethod, bool pushPull) :
			NeighbourCommunicationScheme(1, zonalMethod, pushPull) {
	}
	virtual ~DirectNeighbourCommunicationScheme() {
	}
	virtual void initCommunicationPartners(double cutoffRadius, Domain * domain,
			DomainDecompMPIBase* domainDecomp,
			ParticleContainer* moleculeContainer) override;
	virtual std::vector<int> get3StageNeighbourRanks() override {
		std::vector<int> neighbourRanks;
		for (unsigned int i = 0; i < (*_neighbours)[0].size(); i++) {
			if ((*_neighbours)[0][i].isFaceCommunicator()) {
				neighbourRanks.push_back((*_neighbours)[0][i].getRank());
			}
		}
		return neighbourRanks;
	}

	virtual void prepareNonBlockingStageImpl(ParticleContainer* moleculeContainer, Domain* domain,
			unsigned int stageNumber, MessageType msgType, bool removeRecvDuplicates,
			DomainDecompMPIBase* domainDecomp);

	virtual void finishNonBlockingStageImpl(ParticleContainer* moleculeContainer, Domain* domain,
			unsigned int stageNumber, MessageType msgType, bool removeRecvDuplicates,
			DomainDecompMPIBase* domainDecomp);

	virtual void exchangeMoleculesMPI(ParticleContainer* moleculeContainer, Domain* domain, MessageType msgType,
			bool removeRecvDuplicates, DomainDecompMPIBase* domainDecomp) override;

protected:
	void finalizeExchangeMoleculesMPI(ParticleContainer* moleculeContainer, Domain* /*domain*/, MessageType /*msgType*/,
			bool removeRecvDuplicates, DomainDecompMPIBase* domainDecomp);
	void initExchangeMoleculesMPI(ParticleContainer* moleculeContainer, Domain* /*domain*/, MessageType msgType,
			bool /*removeRecvDuplicates*/, DomainDecompMPIBase* domainDecomp);

private:
	void doDirectFallBackExchange(const std::vector<HaloRegion>& haloRegions, MessageType msgType,
			DomainDecompMPIBase* domainDecomp, ParticleContainer*& moleculeContainer);
	
	void aquireNeighbours(Domain *domain, HaloRegion *haloRegion, std::vector<HaloRegion>& desiredRegions, std::vector<CommunicationPartner>& partners01, std::vector<CommunicationPartner>& partners02);
	void shiftIfNeccessary(double *domainLength, HaloRegion *region, double *shiftArray);
	void overlap(HaloRegion *myRegion, HaloRegion *inQuestion);
	bool iOwnThis(HaloRegion *myRegion, HaloRegion *inQuestion);
};

class IndirectNeighbourCommunicationScheme: public NeighbourCommunicationScheme {
	friend class NeighbourCommunicationSchemeTest;
public:

	IndirectNeighbourCommunicationScheme(ZonalMethod* zonalMethod) :
			NeighbourCommunicationScheme(3, zonalMethod, false) {
	}
	virtual ~IndirectNeighbourCommunicationScheme() {
	}
	void exchangeMoleculesMPI(ParticleContainer* moleculeContainer, Domain* domain, MessageType msgType,
			bool removeRecvDuplicates, DomainDecompMPIBase* domainDecomp) override;

	virtual void initCommunicationPartners(double cutoffRadius, Domain * domain,
			DomainDecompMPIBase* domainDecomp,
			ParticleContainer* moleculeContainer) override;
	virtual std::vector<int> get3StageNeighbourRanks() override {
		std::vector<int> neighbourRanks;
		for (unsigned int i = 0; i < _fullShellNeighbours.size(); i++) {
			if (_fullShellNeighbours[i].isFaceCommunicator()) {
				neighbourRanks.push_back(_fullShellNeighbours[i].getRank());
			}
		}
		return neighbourRanks;
	}

	virtual void prepareNonBlockingStageImpl(ParticleContainer* moleculeContainer, Domain* domain,
			unsigned int stageNumber, MessageType msgType, bool removeRecvDuplicates,
			DomainDecompMPIBase* domainDecomp);

	virtual void finishNonBlockingStageImpl(ParticleContainer* moleculeContainer, Domain* domain,
			unsigned int stageNumber, MessageType msgType, bool removeRecvDuplicates,
			DomainDecompMPIBase* domainDecomp);

protected:
	void initExchangeMoleculesMPI1D(ParticleContainer* moleculeContainer, Domain* domain, MessageType msgType,
			bool removeRecvDuplicates, unsigned short d, DomainDecompMPIBase* domainDecomp);

	void finalizeExchangeMoleculesMPI1D(ParticleContainer* moleculeContainer, Domain* domain, MessageType msgType,
			bool removeRecvDuplicates, unsigned short d, DomainDecompMPIBase* domainDecomp);
	void exchangeMoleculesMPI1D(ParticleContainer* moleculeContainer, Domain* domain, MessageType msgType,
			bool removeRecvDuplicates, unsigned short d, DomainDecompMPIBase* domainDecomp);
	void convert1StageTo3StageNeighbours(const std::vector<CommunicationPartner>& commPartners,
			std::vector<std::vector<CommunicationPartner>>& neighbours, HaloRegion& ownRegion, double cutoffRadius);

};
