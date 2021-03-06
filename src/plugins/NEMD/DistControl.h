/*
 * DistControl.h
 *
 *  Created on: 16.03.2015
 *      Author: mheinen
 */

#ifndef DISTCONTROL_H_
#define DISTCONTROL_H_

#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <cstdint>
#include "utils/ObserverBase.h"
#include "utils/Region.h"
#include "utils/CommVar.h"
#include "molecules/MoleculeForwardDeclaration.h"

using namespace std;

class XMLfileUnits;
class Domain;
class ParticleContainer;
class DomainDecompBase;

#define FORMAT_SCI_MAX_DIGITS std::setw(24) << std::scientific << std::setprecision(std::numeric_limits<double>::digits10)

enum DistControlUpdateMethods
{
	DCUM_UNKNOWN = 0,
	DCUM_DENSITY_PROFILE = 1,
	DCUM_DENSITY_PROFILE_DERIVATION = 2,
//	DCUM_FORCE_PROFILE = 3,
};

enum DistControlInitMethods
{
	DCIM_UNKNOWN = 0,
	DCIM_START_CONFIGURATION = 1,
	DCIM_MIDPOINT_VALUES = 2,
	DCIM_READ_FROM_FILE = 3,
};

enum DistControlCOM : uint16_t
{
	DCCOM_UNKNOWN = 0,
	DCCOM_DEFAULT = 1,
	DCCOM_INTERFACE_POSITIONS = 2
};

class DistControl : public ControlInstance, public SubjectBase
{
public:
	DistControl();
	virtual ~DistControl();

	std::string GetShortName() {return "DiC";}
	void readXML(XMLfileUnits& xmlconfig);

	// set subdivision
	void SetSubdivision(const uint32_t& numBins) {_binParams.count = numBins; _nSubdivisionOpt = SDOPT_BY_NUM_SLABS;}
	void SetSubdivision(const double& dBinWidth) {_binParams.width = dBinWidth; _nSubdivisionOpt = SDOPT_BY_SLAB_WIDTH;}
	void PrepareSubdivision();  // need to be called before PrepareDataStructures()

	// data structures
	void PrepareDataStructures();

	// init
	void InitPositions(double dInterfaceMidLeft, double dInterfaceMidRight);

	double GetInterfaceMidLeft() {return _dInterfaceMidLeft;}
	double GetInterfaceMidRight() {return _dInterfaceMidRight;}
	unsigned int GetUpdateFreq() {return _controlFreqs.update;}
	unsigned int GetWriteFreqProfiles() {return _controlFreqs.write.profiles;}

	// set update/init method
	void SetUpdateMethod(const int& nMethod, const unsigned short& nVal1, const unsigned short& nVal2, const unsigned short& nVal3, const double& dVal);
	void SetInitMethod(const int& nMethod, const double& dVal1, const double& dVal2, std::string strVal, const unsigned long& nVal);
	int GetInitMethod() {return _nMethodInit;}

	void Init(ParticleContainer* particleContainer);
	void WriteHeader();
	void WriteData(unsigned long simstep);
	void WriteDataProfiles(unsigned long simstep);

	// place method inside loop over molecule container
	void SampleProfiles(Molecule* mol);

	void UpdatePositionsInit(ParticleContainer* particleContainer);  // call in Simulation::prepare_start()
	void UpdatePositions(unsigned long simstep);
	void AlignSystemCenterOfMass(Molecule* mol, unsigned long simstep);
	void SampleCOM(Molecule* mol, unsigned long simstep);
	void AlignCOM(Molecule* mol, unsigned long simstep);
	void CalcGlobalValuesCOM(unsigned long simstep);

	// SubjectBase methods
	virtual void registerObserver(ObserverBase* observer);
	virtual void deregisterObserver(ObserverBase* observer);
	virtual void informObserver();

	// COM
	bool AlignCOMactivated() {return _COM.isActive;}
	uint16_t GetMethodCOM() {return _COM.method;}

private:
	// place methods after the loop
	void CalcProfiles();
	void EstimateInterfaceMidpoint();  // called by UpdatePositions
	void EstimateInterfaceMidpointsByForce();
	void ResetLocalValues();
	void ResetLocalValuesCOM();

	// data structures
	void InitDataStructurePointers();
	void AllocateDataStructures();
	void InitDataStructures();

	// processing profiles
	void SmoothProfile(double* dData, double* dSmoothData, const unsigned long& nNumVals, const unsigned int& nNeighbourVals);
	void SmoothProfiles(const unsigned int& nNeighbourVals);
	void DerivateProfile(double* dDataX, double* dDataY, double* dDerivDataY, const unsigned long& nNumVals, const unsigned int& nNeighbourVals);
	void DerivateProfiles(const unsigned int& nNeighbourVals);


private:
	double _dInterfaceMidLeft;
	double _dInterfaceMidRight;

	unsigned short _nNumComponents;
	unsigned short _nTargetCompID;
	unsigned long _nNumValuesScalar;
	unsigned long* _nOffsets;

	unsigned long* _nNumMoleculesLocal;
	unsigned long* _nNumMoleculesGlobal;
	double* _dMidpointPositions;
	double* _dForceSumLocal;
	double* _dForceSumGlobal;
	double* _dDensityProfile;
	double* _dDensityProfileSmoothed;
	double* _dDensityProfileSmoothedDerivation;
	double* _dForceProfile;
	double* _dForceProfileSmoothed;

	// update method
	int _nMethod;
	double _dVaporDensity;
	unsigned short _nNeighbourValsSmooth;
	unsigned short _nNeighbourValsDerivate;

	int _nMethodInit;
	std::string _strFilenameInit;
	unsigned long _nRestartTimestep;

	// write data
	std::string _strFilename;
	std::string _strFilenameProfilesPrefix;

	// observer
	std::vector<ObserverBase*> _observer;

	int _nSubdivisionOpt;

	// align COM
	struct COM_type
	{
		uint16_t method;
		bool isActive;
		CommVar <uint64_t*> numMolecules;
		CommVar <double*> posSum;
		uint16_t cidTarget;
		struct PositionType
		{
			double target[3];
			double actual[3];
			double addVec[3];
		} position;
	} _COM;

	struct BinParamsType
	{
		uint32_t count;
		double width;
		double invWidth;
		double volume;
	} _binParams;

	struct ControlFreqType
	{
		uint32_t update;
		struct WriteFreqType
		{
			uint32_t data;
			uint32_t profiles;
		} write;
	} _controlFreqs;

};  // class DistControl


#endif /* DISTCONTROL_H_ */
