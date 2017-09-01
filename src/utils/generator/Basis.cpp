/*
 * Copyright (c) 2013      Christoph Niethammer <christoph.niethammer@gmail.com>
 *
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER
 */

#include "utils/generator/Basis.h"
#include "utils/Logger.h"
#include "Simulation.h"
#include "ensemble/EnsembleBase.h"

using namespace std;
using Log::global_log;


void Basis::readXML(XMLfileUnits& xmlconfig) {
	XMLfile::Query query = xmlconfig.query("site");
	Ensemble* ensemble = _simulation.getEnsemble();
	string oldpath = xmlconfig.getcurrentnodepath();
	for(auto siteIter = query.begin(); siteIter; siteIter++) {
		Molecule molecule;
		xmlconfig.changecurrentnode(siteIter);
		int componentid;
		xmlconfig.getNodeValue("componentid", componentid);
		molecule.setComponent(ensemble->getComponent(componentid));
		double r[3];
		xmlconfig.getNodeValueReduced("coordinate/x", r[0]);
		xmlconfig.getNodeValueReduced("coordinate/y", r[1]);
		xmlconfig.getNodeValueReduced("coordinate/z", r[2]);
		molecule.setr(0, r[0]);
		molecule.setr(1, r[1]);
		molecule.setr(2, r[2]);
		global_log->info() << "[Basis] Adding molecule cid=" << componentid << ", (x,y,z)=(" << molecule.r(0) << "," << molecule.r(1) << "," << molecule.r(2) << ")" << endl;
		addMolecule(molecule);
	}
	xmlconfig.changecurrentnode(oldpath);
}

void Basis::addMolecule(Molecule molecule) {
	_molecules.push_back(molecule);
}

Molecule Basis::getMolecule(int i) {
	return _molecules[i];
}


int Basis::numMolecules(){
	return _molecules.size();
}
