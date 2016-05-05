#include "particleContainer/LinkedCells.h"

#include <cmath>

#include "Domain.h"
#include "ensemble/GrandCanonical.h"
#include "parallel/DomainDecompBase.h"
#include "particleContainer/adapter/ParticlePairs2PotForceAdapter.h"
#include "particleContainer/handlerInterfaces/ParticlePairsHandler.h"
#include "particleContainer/adapter/CellProcessor.h"
#include "ParticleCell.h"
#include "molecules/Molecule.h"
#include "utils/Logger.h"
#include "utils/UnorderedVector.h"

using namespace std;
using Log::global_log;

//################################################
//############ PUBLIC METHODS ####################
//################################################

LinkedCells::LinkedCells(double bBoxMin[3], double bBoxMax[3],
		double cutoffRadius, double LJCutoffRadius, double cellsInCutoffRadius) :
		ParticleContainer(bBoxMin, bBoxMax) {
	int numberOfCells = 1;
	_cutoffRadius = cutoffRadius;
	_LJCutoffRadius = LJCutoffRadius;

	global_log->debug() << "cutoff: " << cutoffRadius << endl;
	global_log->debug() << "LJ cutoff:" << LJCutoffRadius << endl;
	global_log->debug() << "# cells in cutoff: " << cellsInCutoffRadius << endl;

	_cellsInCutoff = ceil(cellsInCutoffRadius);

	if (_cellsInCutoff != 1) {
		global_log->error()
				<< "With the recent release only 1 cell per cutoff radius is supported,"
				<< " but the input file prescribes " << _cellsInCutoff
				<< " cells per cutoff radius." << endl
				<< "\tThe support has been dropped, since no speedup can be expected using"
				<< " multiple cells per cutoff radius." << endl
				<< "\tIf you can provide a case, where this is not true, please contact us."
				<< endl;
		global_simulation->exit(-1);
	}

	for (int d = 0; d < 3; d++) {
		/* first calculate the cell length for this dimension */
		_boxWidthInNumCells[d] = floor(
				(_boundingBoxMax[d] - _boundingBoxMin[d]) / cutoffRadius
						* cellsInCutoffRadius);
		// in each dimension at least one layer of (inner+boundary) cells is necessary
		if (_boxWidthInNumCells[d] == 0) {
			_boxWidthInNumCells[d] = 1;
		}
		_cellLength[d] = (_boundingBoxMax[d] - _boundingBoxMin[d])
				/ _boxWidthInNumCells[d];
		_haloWidthInNumCells[d] = ceil(cellsInCutoffRadius);
		_haloLength[d] = _haloWidthInNumCells[d] * _cellLength[d];
		_haloBoundingBoxMin[d] = _boundingBoxMin[d] - _haloLength[d];
		_haloBoundingBoxMax[d] = _boundingBoxMax[d] + _haloLength[d];

		_cellsPerDimension[d] = _boxWidthInNumCells[d]
				+ 2 * _haloWidthInNumCells[d];

		numberOfCells *= _cellsPerDimension[d];
		assert(numberOfCells > 0);
	}
	global_log->debug() << "Cell size (" << _cellLength[0] << ", "
			<< _cellLength[1] << ", " << _cellLength[2] << ")" << endl;
	global_log->debug() << "Cells per dimension (incl. halo): "
			<< _cellsPerDimension[0] << " x " << _cellsPerDimension[1] << " x "
			<< _cellsPerDimension[2] << endl;

	_cells.resize(numberOfCells);

	// If the width of the inner region is less than the width of the halo
	// region a parallelization is not possible (with the used algorithms).
	// If a particle leaves this box, it would need to be communicated to the two next neighbors.
	if (_boxWidthInNumCells[0] < 2 * _haloWidthInNumCells[0]
			|| _boxWidthInNumCells[1] < 2 * _haloWidthInNumCells[1]
			|| _boxWidthInNumCells[2] < 2 * _haloWidthInNumCells[2]) {
		global_log->error_always_output()
				<< "LinkedCells (constructor): bounding box too small for calculated cell length"
				<< endl;
		global_log->error_always_output() << "_cellsPerDimension: " << _cellsPerDimension[0]
				<< " / " << _cellsPerDimension[1] << " / "
				<< _cellsPerDimension[2] << endl;
		global_log->error_always_output() << "_haloWidthInNumCells: "
				<< _haloWidthInNumCells[0] << " / " << _haloWidthInNumCells[1]
				<< " / " << _haloWidthInNumCells[2] << endl;
		global_log->error_always_output() << "_boxWidthInNumCells: " << _boxWidthInNumCells[0]
				<< " / " << _boxWidthInNumCells[1] << " / "
				<< _boxWidthInNumCells[2] << endl;
		global_simulation->exit(5);
	}
	this->_localInsertionsMinusDeletions = 0;

	initializeCells();
	calculateNeighbourIndices();
	_cellsValid = false;
}

LinkedCells::~LinkedCells() {
	std::vector<ParticleCell>::iterator it;
	for (it = _cells.begin(); it != _cells.end(); ++it) {
		it->deallocateAllParticles();
	}
}

void LinkedCells::readXML(XMLfileUnits& xmlconfig) {
	xmlconfig.getNodeValue("cellsInCutoffRadius", _cellsInCutoff);
	if (_cellsInCutoff != 1) {
		global_log->error()
				<< "With the recent release only 1 cell per cutoff radius is supported,"
				<< " but the input file prescribes " << _cellsInCutoff
				<< " cells per cutoff radius." << endl
				<< "\tThe support has been dropped, since no speedup can be expected using"
				<< " multiple cells per cutoff radius." << endl
				<< "\tIf you can provide a case, where this is not true, please contact us."
				<< endl;
		global_simulation->exit(-1);
	}
	global_log->info() << "Cells in cut-off radius: " << _cellsInCutoff << endl;
}

void LinkedCells::rebuild(double bBoxMin[3], double bBoxMax[3]) {
	for (int i = 0; i < 3; i++) {
		this->_boundingBoxMin[i] = bBoxMin[i];
		this->_boundingBoxMax[i] = bBoxMax[i];
		_haloWidthInNumCells[i] = ::ceil(_cellsInCutoff); /* TODO: Single value?! */
	}
	int numberOfCells = 1;

	for (int dim = 0; dim < 3; dim++) {
		_boxWidthInNumCells[dim] = floor(
				(_boundingBoxMax[dim] - _boundingBoxMin[dim]) / _cutoffRadius
						* _haloWidthInNumCells[dim]);
		// in each dimension at least one layer of (inner+boundary) cells is necessary
		if (_boxWidthInNumCells[dim] == 0) {
			_boxWidthInNumCells[dim] = 1;
		}

		_cellsPerDimension[dim] = (int) floor(
				(_boundingBoxMax[dim] - _boundingBoxMin[dim])
						/ (_cutoffRadius / _haloWidthInNumCells[dim]))
				+ 2 * _haloWidthInNumCells[dim];
		// in each dimension at least one layer of (inner+boundary) cells necessary
		if (_cellsPerDimension[dim] == 2 * _haloWidthInNumCells[dim]) {
			global_log->error_always_output() << "LinkedCells::rebuild: region to small"
					<< endl;
			global_simulation->exit(1);
		}
		numberOfCells *= _cellsPerDimension[dim];
		_cellLength[dim] = (_boundingBoxMax[dim] - _boundingBoxMin[dim])
				/ (_cellsPerDimension[dim] - 2 * _haloWidthInNumCells[dim]);
		_haloBoundingBoxMin[dim] = this->_boundingBoxMin[dim]
				- _haloWidthInNumCells[dim] * _cellLength[dim];
		_haloBoundingBoxMax[dim] = this->_boundingBoxMax[dim]
				+ _haloWidthInNumCells[dim] * _cellLength[dim];
		_haloLength[dim] = _haloWidthInNumCells[dim] * _cellLength[dim];
	}

	_cells.resize(numberOfCells);

	// If the with of the inner region is less than the width of the halo region
	// a parallelisation isn't possible (with the used algorithms).
	// In this case, print an error message
	// _cellsPerDimension is 2 times the halo width + the inner width
	// so it has to be at least 3 times the halo width
	if (_boxWidthInNumCells[0] < 2 * _haloWidthInNumCells[0]
			|| _boxWidthInNumCells[1] < 2 * _haloWidthInNumCells[1]
			|| _boxWidthInNumCells[2] < 2 * _haloWidthInNumCells[2]) {
		global_log->error_always_output()
				<< "LinkedCells (rebuild): bounding box too small for calculated cell Length"
				<< endl;
		global_log->error_always_output() << "cellsPerDimension " << _cellsPerDimension[0]
				<< " / " << _cellsPerDimension[1] << " / "
				<< _cellsPerDimension[2] << endl;
		global_log->error_always_output() << "_haloWidthInNumCells" << _haloWidthInNumCells[0]
				<< " / " << _haloWidthInNumCells[1] << " / "
				<< _haloWidthInNumCells[2] << endl;
		global_simulation->exit(5);
	}

	initializeCells();
	calculateNeighbourIndices();

	// TODO: We loose particles here as they are not communicated to the new owner
	// delete all Particles which are outside of the halo region
	deleteParticlesOutsideBox(_haloBoundingBoxMin, _haloBoundingBoxMax);

	_cellsValid = false;
}

void LinkedCells::update() {
	std::vector<ParticleCell>::iterator celliter;

	for (celliter = _cells.begin(); celliter != _cells.end(); ++celliter) {

		std::vector<Molecule*> & molsToSort =
				celliter->filterLeavingMolecules();
		std::vector<Molecule*>::iterator it;

		for (it = molsToSort.begin(); it != molsToSort.end(); ++it) {
			bool wasInserted = addParticlePointer(*it);

			// lets stay on the safe side:
			if (wasInserted) {
				// all is good, nothing to do, just helping the branch predictor
			} else {
				delete *it;
			}
		}
		molsToSort.clear();
	}

	_cellsValid = true;
}

bool LinkedCells::addParticle(Molecule& particle) {
	const bool inBox = particle.inBox(_haloBoundingBoxMin, _haloBoundingBoxMax);

	if (inBox) {
		Molecule * mol = new Molecule(particle);
		addParticlePointer(mol, true, false);
	}

	return inBox;
}

bool LinkedCells::addParticlePointer(Molecule * particle,
		bool inBoxCheckedAlready, bool checkWhetherDuplicate) {
	const bool inBox = inBoxCheckedAlready
			or particle->inBox(_haloBoundingBoxMin, _haloBoundingBoxMax);

	bool wasInserted = false;

	if (inBox) {
		int cellIndex = getCellIndexOfMolecule(particle);
		wasInserted = _cells[cellIndex].addParticle(particle,
				checkWhetherDuplicate);
	}

	return wasInserted;
}

/**
 * @todo replace this by a call to component->getNumMolecules() !?
 */
unsigned LinkedCells::countParticles(unsigned int cid) {
	unsigned N = 0;
	std::vector<Molecule*>::iterator molIter1;
	for (unsigned i = 0; i < _cells.size(); i++) {
		ParticleCell& currentCell = _cells[i];
		if (!currentCell.isHaloCell()) {
			for (molIter1 = currentCell.getParticlePointers().begin();
					molIter1 != currentCell.getParticlePointers().end();
					molIter1++) {

				if ((*molIter1)->componentid() == cid)
					N++;
			}
		}
	}
	return N;
}

// @todo: couldn't this use getRegion?
unsigned LinkedCells::countParticles(unsigned int cid, double* cbottom,
		double* ctop) {
	int minIndex[3];
	int maxIndex[3];
	for (int d = 0; d < 3; d++) {
		if (cbottom[d] < this->_haloBoundingBoxMin[d])
			minIndex[d] = 0;
		else
			minIndex[d] = (int) floor(
					(cbottom[d] - this->_haloBoundingBoxMin[d])
							/ _cellLength[d]);

		if (ctop[d] > this->_haloBoundingBoxMax[d])
			maxIndex[d] = (int) floor(
					(this->_haloBoundingBoxMax[d] - _haloBoundingBoxMin[d])
							/ this->_cellLength[d]);
		else
			maxIndex[d] = (int) floor(
					(ctop[d] - this->_haloBoundingBoxMin[d]) / _cellLength[d]);

		if (minIndex[d] < 0)
			minIndex[d] = 0;
		if (maxIndex[d] >= _cellsPerDimension[d])
			maxIndex[d] = _cellsPerDimension[d] - 1;
	}

	unsigned N = 0;
	int cix[3];
	std::vector<Molecule*>::iterator molIter1;
	bool individualCheck;
	int cellid;

	for (cix[0] = minIndex[0]; maxIndex[0] >= cix[0]; (cix[0])++) {
		for (cix[1] = minIndex[1]; maxIndex[1] >= cix[1]; (cix[1])++) {
			for (cix[2] = minIndex[2]; maxIndex[2] >= cix[2]; (cix[2])++) {
				individualCheck = (cix[0] == minIndex[0])
						|| (cix[0] == minIndex[0] + 1)
						|| (cix[0] == maxIndex[0])
						|| (cix[0] == maxIndex[0] - 1)
						|| (cix[1] == minIndex[1])
						|| (cix[1] == minIndex[1] + 1)
						|| (cix[1] == maxIndex[1])
						|| (cix[1] == maxIndex[1] - 1)
						|| (cix[2] == minIndex[2])
						|| (cix[2] == minIndex[2] + 1)
						|| (cix[2] == maxIndex[2])
						|| (cix[2] == maxIndex[2] - 1);
				cellid = this->cellIndexOf3DIndex(cix[0], cix[1], cix[2]);
				ParticleCell& currentCell = _cells[cellid];
				if (currentCell.isHaloCell())
					continue;
				if (individualCheck) {
					for (molIter1 = currentCell.getParticlePointers().begin();
							molIter1 != currentCell.getParticlePointers().end();
							molIter1++) {
						if ((*molIter1)->inBox(cbottom, ctop)
								and ((*molIter1)->componentid() == cid)) {
							N++;
						}
					}
				} else {
					for (molIter1 = currentCell.getParticlePointers().begin();
							molIter1 != currentCell.getParticlePointers().end();
							molIter1++) {
						if ((*molIter1)->componentid() == cid)
							N++;
					}
				}
			}
		}
	}

	return N;
}

void LinkedCells::traverseNonInnermostCells(CellProcessor& cellProcessor){
	throw "Not yet implemented.";
}

void LinkedCells::traversePartialInnermostCells(CellProcessor& cellProcessor, unsigned int stage, int stageCount){
	throw "Not yet implemented.";
}

void LinkedCells::traverseCells(CellProcessor& cellProcessor) {
	if (_cellsValid == false) {
		global_log->error()
				<< "Cell structure in LinkedCells (traversePairs) invalid, call update first"
				<< endl;
		global_simulation->exit(1);
	}

	vector<long int>::iterator neighbourOffsetsIter;

#ifndef NDEBUG
	global_log->debug()
			<< "LinkedCells::traverseCells: Processing pairs."
			<< endl;
	global_log->debug() << "_minNeighbourOffset=" << _minNeighbourOffset
			<< "; _maxNeighbourOffset=" << _maxNeighbourOffset << endl;
#endif

	cellProcessor.initTraversal(_maxNeighbourOffset + _minNeighbourOffset + 1);
	// open the window of cells activated
	for (long int cellIndex = 0; cellIndex < _maxNeighbourOffset; cellIndex++) {
		cellProcessor.preprocessCell(_cells[cellIndex]);
	}

	// loop over all inner cells and calculate forces to forward neighbours
	for (long int cellIndex = 0; cellIndex < (long int) _cells.size();
			cellIndex++) {
		ParticleCell& currentCell = _cells[cellIndex];

		// extend the window of cells with cache activated
		if (cellIndex + _maxNeighbourOffset < (long int) _cells.size()) {
			cellProcessor.preprocessCell(
					_cells[cellIndex + _maxNeighbourOffset]);
		}

		if (currentCell.isInnerCell()) {
			cellProcessor.processCell(currentCell);
			// loop over all forward neighbours
			for (neighbourOffsetsIter = _forwardNeighbourOffsets.begin();
					neighbourOffsetsIter != _forwardNeighbourOffsets.end();
					neighbourOffsetsIter++) {
				ParticleCell& neighbourCell = _cells[cellIndex
						+ *neighbourOffsetsIter];
				cellProcessor.processCellPair(currentCell, neighbourCell);
			}
		}

		if (currentCell.isHaloCell()) {
			cellProcessor.processCell(currentCell);
			for (neighbourOffsetsIter = _forwardNeighbourOffsets.begin();
					neighbourOffsetsIter != _forwardNeighbourOffsets.end();
					neighbourOffsetsIter++) {
				long int neighbourCellIndex = cellIndex + *neighbourOffsetsIter;
				if ((neighbourCellIndex < 0)
						|| (neighbourCellIndex >= (int) (_cells.size())))
					continue;
				ParticleCell& neighbourCell = _cells[neighbourCellIndex];
				if (!neighbourCell.isHaloCell())
					continue;

				cellProcessor.processCellPair(currentCell, neighbourCell);
			}
		}

		// loop over all boundary cells and calculate forces to forward and backward neighbours
		if (currentCell.isBoundaryCell()) {
			cellProcessor.processCell(currentCell);

			// loop over all forward neighbours
			for (neighbourOffsetsIter = _forwardNeighbourOffsets.begin();
					neighbourOffsetsIter != _forwardNeighbourOffsets.end();
					neighbourOffsetsIter++) {
				ParticleCell& neighbourCell = _cells[cellIndex
						+ *neighbourOffsetsIter];
				cellProcessor.processCellPair(currentCell, neighbourCell);
			}

			// loop over all backward neighbours. calculate only forces
			// to neighbour cells in the halo region, all others already have been calculated
			for (neighbourOffsetsIter = _backwardNeighbourOffsets.begin();
					neighbourOffsetsIter != _backwardNeighbourOffsets.end();
					neighbourOffsetsIter++) {
				ParticleCell& neighbourCell = _cells[cellIndex
						- *neighbourOffsetsIter];  // minus oder plus?
				if (neighbourCell.isHaloCell()) {
					cellProcessor.processCellPair(currentCell, neighbourCell);
				}
			}
		} // if ( isBoundaryCell() )

		// narrow the window of cells activated
		if (cellIndex >= _minNeighbourOffset) {
			cellProcessor.postprocessCell(
					_cells[cellIndex - _minNeighbourOffset]);
		}
	} // loop over all cells

	// close the window of cells with cache activated
	for (unsigned int cellIndex = _cells.size() - _minNeighbourOffset;
			cellIndex < _cells.size(); cellIndex++) {
		cellProcessor.postprocessCell(_cells[cellIndex]);
	}
	cellProcessor.endTraversal();
}

unsigned long LinkedCells::getNumberOfParticles() {
	unsigned long N = 0;
	std::vector<ParticleCell>::iterator it;
	for (it = _cells.begin(); it != _cells.end(); ++it) {
		N += it->getMoleculeCount();
	}
	return N;
}

MoleculeIterator LinkedCells::nextNonEmptyCell() {
	MoleculeIterator ret = LinkedCells::end();

	const std::vector<ParticleCell>::const_iterator cellsEnd = _cells.end();

	do {
		++_cellIterator;
	} while (_cellIterator != cellsEnd and _cellIterator->isEmpty());

	if (_cellIterator != cellsEnd) {
		_particleIterator = _cellIterator->getParticlePointers().begin();
		ret = *_particleIterator;
	}

	return ret;
}

MoleculeIterator LinkedCells::begin() {
	MoleculeIterator ret = LinkedCells::end();

	_cellIterator = _cells.begin();

	if (_cellIterator->isEmpty()) {
		ret = nextNonEmptyCell();
	} else {
		_particleIterator = _cellIterator->getParticlePointers().begin();
		ret = *_particleIterator;
	}

	return ret;
}

MoleculeIterator LinkedCells::next() {
	Molecule* ret = LinkedCells::end();

	++_particleIterator;

	if (_particleIterator != _cellIterator->getParticlePointers().end()) {
		ret = *_particleIterator;
	} else {
		ret = nextNonEmptyCell();
	}

	return ret;
}

MoleculeIterator LinkedCells::end() {
	return NULL;
}

MoleculeIterator LinkedCells::deleteCurrent() {
	delete *_particleIterator; // free storage for molecule
	MoleculeIterator ret = LinkedCells::end();
	std::vector<Molecule *> & v = _cellIterator->getParticlePointers();

	UnorderedVector::fastRemove(v, _particleIterator);

	if (_particleIterator != v.end()) {
		ret = *_particleIterator;
	} else {
		ret = nextNonEmptyCell();
	}

	return ret;
}

void LinkedCells::clear() {
	vector<ParticleCell>::iterator cellIter;
	for (cellIter = _cells.begin(); cellIter != _cells.end(); cellIter++) {
		cellIter->deallocateAllParticles();
	}
}

void LinkedCells::deleteParticlesOutsideBox(double boxMin[3],
		double boxMax[3]) {
	Molecule * it;
	for (it = begin(); it != end();) {
		bool keepMolecule = it->inBox(boxMin, boxMax);
		if (keepMolecule) {
			it = next();
		} else {
			it = deleteCurrent();
		}
	}
}

void LinkedCells::deleteOuterParticles() {
	if (_cellsValid == false) {
		global_log->error()
				<< "Cell structure in LinkedCells (deleteOuterParticles) invalid, call update first"
				<< endl;
		global_simulation->exit(1);
	}

	vector<unsigned long>::iterator cellIndexIter;
	for (cellIndexIter = _haloCellIndices.begin();
			cellIndexIter != _haloCellIndices.end(); cellIndexIter++) {
		ParticleCell& currentCell = _cells[*cellIndexIter];
		currentCell.deallocateAllParticles();
	}
}

double LinkedCells::get_halo_L(int index) const {
	return _haloLength[index];
}

void LinkedCells::getHaloParticles(list<Molecule*> &haloParticlePtrs) {
	if (_cellsValid == false) {
		global_log->error()
				<< "Cell structure in LinkedCells (getHaloParticles) invalid, call update first"
				<< endl;
		global_simulation->exit(1);
	}

	std::vector<Molecule*>::iterator particleIter;
	vector<unsigned long>::iterator cellIndexIter;

	// loop over all halo cells
	for (cellIndexIter = _haloCellIndices.begin();
			cellIndexIter != _haloCellIndices.end(); cellIndexIter++) {
		ParticleCell& currentCell = _cells[*cellIndexIter];
		// loop over all molecules in the cell
		for (particleIter = currentCell.getParticlePointers().begin();
				particleIter != currentCell.getParticlePointers().end();
				particleIter++) {
			haloParticlePtrs.push_back(*particleIter);
		}
	}
}

void LinkedCells::getHaloParticlesDirection(int direction,
		std::vector<Molecule*>& v, bool removeFromContainer) {
	assert(direction != 0);

	int startIndex[3] = { 0, 0, 0 };
	int stopIndex[3] = { _cellsPerDimension[0] - 1, _cellsPerDimension[1] - 1,
			_cellsPerDimension[2] - 1 };

	// get dimension in 0, 1, 2 format from direction in +-1, +-2, +-3 format
	unsigned dim = abs(direction) - 1;
	if (direction < 0) {
		stopIndex[dim] = startIndex[dim] + (_haloWidthInNumCells[dim] - 1); // -1 needed for equality below
	} else {
		startIndex[dim] = stopIndex[dim] - (_haloWidthInNumCells[dim] - 1); // -1 needed for equality below
	}

	for (int iz = startIndex[2]; iz <= stopIndex[2]; iz++) {
		for (int iy = startIndex[1]; iy <= stopIndex[1]; iy++) {
			for (int ix = startIndex[0]; ix <= stopIndex[0]; ix++) {
				const int cellIndex = cellIndexOf3DIndex(ix, iy, iz);
				ParticleCell & cell = _cells[cellIndex];
				std::vector<Molecule *> & mols = cell.getParticlePointers();
				v.insert(v.end(), mols.begin(), mols.end());
				if (removeFromContainer == true) {
					cell.removeAllParticles();
				}
			}
		}
	}
}

void LinkedCells::getBoundaryParticlesDirection(int direction,
		std::vector<Molecule*>& v) const {
	assert(direction != 0);

	int startIndex[3] = { 0, 0, 0 };
	int stopIndex[3] = { _cellsPerDimension[0] - 1, _cellsPerDimension[1] - 1,
			_cellsPerDimension[2] - 1 };

	// get dimension in 0, 1, 2 format from direction in +-1, +-2, +-3 format
	unsigned dim = abs(direction) - 1;
	if (direction < 0) {
		startIndex[dim] = _haloWidthInNumCells[dim];
		stopIndex[dim] = startIndex[dim] + (_haloWidthInNumCells[dim] - 1); // -1 needed for equality below
	} else {
		stopIndex[dim] = _boxWidthInNumCells[dim];
		startIndex[dim] = stopIndex[dim] - (_haloWidthInNumCells[dim] - 1); // -1 needed for equality below
	}

	for (int iz = startIndex[2]; iz <= stopIndex[2]; iz++) {
		for (int iy = startIndex[1]; iy <= stopIndex[1]; iy++) {
			for (int ix = startIndex[0]; ix <= stopIndex[0]; ix++) {
				const int cellIndex = cellIndexOf3DIndex(ix, iy, iz);
				const ParticleCell & cell = _cells[cellIndex];
				const std::vector<Molecule *> & mols =
						cell.getParticlePointers();
				v.insert(v.end(), mols.begin(), mols.end());
			}
		}
	}
}

void LinkedCells::getRegionSimple(double lowCorner[3], double highCorner[3],
		std::vector<Molecule*> &particlePtrs, bool removeFromContainer) {
	if (_cellsValid == false) {
		global_log->error()
				<< "Cell structure in LinkedCells (getRegionSimple) invalid, call update first"
				<< endl;
		global_simulation->exit(1);
	}

	int startIndex[3];
	int stopIndex[3];
	int globalCellIndex;
	std::vector<Molecule*>::iterator particleIter;

	for (int dim = 0; dim < 3; dim++) {
		if (lowCorner[dim] <= this->_haloBoundingBoxMax[dim]
				&& highCorner[dim] >= this->_haloBoundingBoxMin[dim]) {
			startIndex[dim] = (int) floor(
					(lowCorner[dim] - _haloBoundingBoxMin[dim])
							/ _cellLength[dim]);
			stopIndex[dim] = (int) floor(
					(highCorner[dim] - _haloBoundingBoxMin[dim])
							/ _cellLength[dim]);
			if (startIndex[dim] < 0)
				startIndex[dim] = 0;
			if (stopIndex[dim] > _cellsPerDimension[dim] - 1)
				stopIndex[dim] = _cellsPerDimension[dim] - 1;
		} else {
			// No Part of the given region is owned by this process
			return;
		}
	}

	for (int iz = startIndex[2]; iz <= stopIndex[2]; iz++) {
		for (int iy = startIndex[1]; iy <= stopIndex[1]; iy++) {
			for (int ix = startIndex[0]; ix <= stopIndex[0]; ix++) {
				// globalCellIndex is the cellIndex of the molecule on the coarse Cell level.
				globalCellIndex = cellIndexOf3DIndex(ix, iy, iz);
				// loop over all subcells (either 1 or 8)
				// traverse all molecules in the current cell
				ParticleCell & currentCell = _cells[globalCellIndex];
				currentCell.getRegion(lowCorner, highCorner, particlePtrs,
						removeFromContainer);
			}
		}
	}
}

void LinkedCells::getRegion(double lowCorner[3], double highCorner[3],
		std::vector<Molecule*> &particlePtrs) {
	if (_cellsValid == false) {
		global_log->error()
				<< "Cell structure in LinkedCells (getRegion) invalid, call update first"
				<< endl;
		global_simulation->exit(1);
	}

	int startIndex[3];
	int stopIndex[3];
	int globalCellIndex;
	std::vector<Molecule*>::iterator particleIter;

	for (int dim = 0; dim < 3; dim++) {
		if (lowCorner[dim] < this->_boundingBoxMax[dim]
				&& highCorner[dim] > this->_boundingBoxMin[dim]) {
			startIndex[dim] = (int) floor(
					(lowCorner[dim] - _haloBoundingBoxMin[dim])
							/ _cellLength[dim]) - 1;
			stopIndex[dim] = (int) floor(
					(highCorner[dim] - _haloBoundingBoxMin[dim])
							/ _cellLength[dim]) + 1;
			if (startIndex[dim] < 0)
				startIndex[dim] = 0;
			if (stopIndex[dim] > _cellsPerDimension[dim] - 1)
				stopIndex[dim] = _cellsPerDimension[dim] - 1;
		} else {
			// No Part of the given region is owned by this process
			return;
		}
	}

	for (int iz = startIndex[2]; iz <= stopIndex[2]; iz++) {
		for (int iy = startIndex[1]; iy <= stopIndex[1]; iy++) {
			for (int ix = startIndex[0]; ix <= stopIndex[0]; ix++) {
				// globalCellIndex is the cellIndex of the molecule on the coarse Cell level.
				globalCellIndex = cellIndexOf3DIndex(ix, iy, iz);
				// loop over all subcells (either 1 or 8)
				// traverse all molecules in the current cell
				ParticleCell & currentCell = _cells[globalCellIndex];
				currentCell.getRegion(lowCorner, highCorner, particlePtrs);
			}
		}
	}
}

//################################################
//############ PRIVATE METHODS ###################
//################################################

void LinkedCells::initializeCells() {
	_innerCellIndices.clear();
	_boundaryCellIndices.clear();
	_haloCellIndices.clear();

	long int cellIndex;
	double cellBoxMin[3], cellBoxMax[3];

	for (int iz = 0; iz < _cellsPerDimension[2]; ++iz) {
		cellBoxMin[2] = iz * _cellLength[2] + _haloBoundingBoxMin[2];
		cellBoxMax[2] = (iz + 1) * _cellLength[2] + _haloBoundingBoxMin[2];

		for (int iy = 0; iy < _cellsPerDimension[1]; ++iy) {
			cellBoxMin[1] = iy * _cellLength[1] + _haloBoundingBoxMin[1];
			cellBoxMax[1] = (iy + 1) * _cellLength[1] + _haloBoundingBoxMin[1];

			for (int ix = 0; ix < _cellsPerDimension[0]; ++ix) {
				cellBoxMin[0] = ix * _cellLength[0] + _haloBoundingBoxMin[0];
				cellBoxMax[0] = (ix + 1) * _cellLength[0]
						+ _haloBoundingBoxMin[0];

				cellIndex = cellIndexOf3DIndex(ix, iy, iz);
				ParticleCell & cell = _cells[cellIndex];

				cell.skipCellFromHaloRegion();
				cell.skipCellFromBoundaryRegion();
				cell.skipCellFromInnerRegion();

				cell.setBoxMin(cellBoxMin);
				cell.setBoxMax(cellBoxMax);
				_cells[cellIndex].setCellIndex(cellIndex);//set the index of the cell to the index of it...
				if (ix < _haloWidthInNumCells[0] || iy < _haloWidthInNumCells[1]
						|| iz < _haloWidthInNumCells[2]
						|| ix >= _cellsPerDimension[0] - _haloWidthInNumCells[0]
						|| iy >= _cellsPerDimension[1] - _haloWidthInNumCells[1]
						|| iz
								>= _cellsPerDimension[2]
										- _haloWidthInNumCells[2]) {
					cell.assignCellToHaloRegion();
					_haloCellIndices.push_back(cellIndex);
				} else if (ix < 2 * _haloWidthInNumCells[0]
						|| iy < 2 * _haloWidthInNumCells[1]
						|| iz < 2 * _haloWidthInNumCells[2]
						|| ix
								>= _cellsPerDimension[0]
										- 2 * _haloWidthInNumCells[0]
						|| iy
								>= _cellsPerDimension[1]
										- 2 * _haloWidthInNumCells[1]
						|| iz
								>= _cellsPerDimension[2]
										- 2 * _haloWidthInNumCells[2]) {
					cell.assignCellToBoundaryRegion();
					_boundaryCellIndices.push_back(cellIndex);
				} else {
					cell.assignCellToInnerRegion();
					_innerCellIndices.push_back(cellIndex);
				}
			}
		}
	}

	/*********************  Compute Border indices *********************/
	for (int dim = 0; dim < 3; ++dim) {
		for (int dir = 0; dir < 2; ++dir) {
			for (int typ = 0; typ < 2; ++typ) {
				_borderCellIndices[dim][dir][typ].clear();
				int low[3] = { 0, 0, 0 };
				int high[3] = { _cellsPerDimension[0] - 1, _cellsPerDimension[1]
						- 1, _cellsPerDimension[2] - 1 };

				if (typ == 1) {
					if (dir == 0)
						low[dim]++;
					else
						high[dim]--;
				}

				if (dir == 0)
					high[dim] = low[dim];
				else
					low[dim] = high[dim];

				for (int iz = low[2]; iz <= high[2]; ++iz) {
					for (int iy = low[1]; iy <= high[1]; ++iy) {
						for (int ix = low[0]; ix <= high[0]; ++ix) {
							cellIndex = cellIndexOf3DIndex(ix, iy, iz);
#ifndef NDEBUG
							ParticleCell & cell = _cells[cellIndex];

							assert(not cell.isInnerCell());

							if (typ == 0)
								assert(cell.isHaloCell());
							else
								; /* assert(cell.isBoundaryCell()) is not always true, as we have some halo cells in there */
#endif

							_borderCellIndices[dim][dir][typ].push_back(
									cellIndex);
						}
					}
				}
			}
		}
	}
}

void LinkedCells::calculateNeighbourIndices() {
	global_log->debug() << "Setting up cell neighbour indice lists." << endl;
	_forwardNeighbourOffsets.clear();
	_backwardNeighbourOffsets.clear();
	_maxNeighbourOffset = 0;
	_minNeighbourOffset = 0;
	double xDistanceSquare;
	double yDistanceSquare;
	double zDistanceSquare;
	double cutoffRadiusSquare = pow(_cutoffRadius, 2);
	for (int zIndex = -_haloWidthInNumCells[2];
			zIndex <= _haloWidthInNumCells[2]; zIndex++) {
		// The distance in one dimension is the width of a cell multiplied with the number
		// of cells between the two cells (this is received by substracting one of the
		// absolute difference of the cells, if this difference is not zero)
		if (zIndex != 0) {
			zDistanceSquare = pow((abs(zIndex) - 1) * _cellLength[2], 2);
		} else {
			zDistanceSquare = 0;
		}
		for (int yIndex = -_haloWidthInNumCells[1];
				yIndex <= _haloWidthInNumCells[1]; yIndex++) {
			if (yIndex != 0) {
				yDistanceSquare = pow((abs(yIndex) - 1) * _cellLength[1], 2);
			} else {
				yDistanceSquare = 0;
			}
			for (int xIndex = -_haloWidthInNumCells[0];
					xIndex <= _haloWidthInNumCells[0]; xIndex++) {
				if (xIndex != 0) {
					xDistanceSquare = pow((abs(xIndex) - 1) * _cellLength[0],
							2);
				} else {
					xDistanceSquare = 0;
				}
				if (xDistanceSquare + yDistanceSquare + zDistanceSquare
						<= cutoffRadiusSquare) {
					long int offset = cellIndexOf3DIndex(xIndex, yIndex,
							zIndex);
					if (offset > 0) {
						_forwardNeighbourOffsets.push_back(offset);
						if (offset > _maxNeighbourOffset) {
							_maxNeighbourOffset = offset;
						}
					}
					if (offset < 0) {
						_backwardNeighbourOffsets.push_back(abs(offset));
						if (abs(offset) > _minNeighbourOffset) {
							_minNeighbourOffset = abs(offset);
						}
					}
				}
			}
		}
	}

	global_log->info() << "Neighbour offsets are bounded by "
			<< _minNeighbourOffset << ", " << _maxNeighbourOffset << endl;
}

unsigned long int LinkedCells::getCellIndexOfMolecule(
		Molecule* molecule) const {
	int cellIndex[3]; // 3D Cell index

	for (int dim = 0; dim < 3; dim++) {
#ifndef NDEBUG
		if (molecule->r(dim) < _haloBoundingBoxMin[dim]
				|| molecule->r(dim) >= _haloBoundingBoxMax[dim]) {
			global_log->error() << "Molecule is outside of bounding box"
					<< endl;
			global_log->debug() << "Molecule:\n" << *molecule << endl;
		}
#endif
//		this version is sensitive to roundoffs, if we have molecules (initialized) precisely at position 0.0:
//		cellIndex[dim] = (int) floor((molecule->r(dim) - _haloBoundingBoxMin[dim]) / _cellLength[dim]);
		cellIndex[dim] = ((int) floor(
				(molecule->r(dim) - _boundingBoxMin[dim]) / _cellLength[dim]))
				+ _haloWidthInNumCells[dim];

	}
	return this->cellIndexOf3DIndex(cellIndex[0], cellIndex[1], cellIndex[2]);
}

long int LinkedCells::cellIndexOf3DIndex(long int xIndex, long int yIndex,
		long int zIndex) const {
	return (zIndex * _cellsPerDimension[1] + yIndex) * _cellsPerDimension[0]
			+ xIndex;
}

void LinkedCells::deleteMolecule(unsigned long molid, double x, double y,
		double z) {

	int ix = (int) floor(
			(x - this->_haloBoundingBoxMin[0]) / this->_cellLength[0]);
	int iy = (int) floor(
			(y - this->_haloBoundingBoxMin[1]) / this->_cellLength[1]);
	int iz = (int) floor(
			(z - this->_haloBoundingBoxMin[2]) / this->_cellLength[2]);

	unsigned long hash = this->cellIndexOf3DIndex(ix, iy, iz);
	if (hash >= _cells.size()) {
		global_log->error_always_output()
				<< "coordinates for atom deletion lie outside bounding box."
				<< endl;
		global_simulation->exit(1);
	}

	bool found = this->_cells[hash].deleteMolecule(molid);

	if (!found) {
		global_log->error_always_output() << "could not delete molecule " << molid << "."
				<< endl;
		global_simulation->exit(1);
	}
}

double LinkedCells::getEnergy(ParticlePairsHandler* particlePairsHandler,
		Molecule* m1, CellProcessor& cellProcessor) {

	// TODO: this will modify the forces on the (real) molecules, no?
	// TODO: and the global Upot, Virial, ..., no?

	static ParticleCell dummyCell;
	{
		// (potentially re-) initialize dummyCell
		dummyCell.assignCellToInnerRegion();
		dummyCell.removeAllParticles();
		dummyCell.addParticle(m1, false);
		dummyCell.buildSoACaches();
		dummyCell.setCellIndex(_cells.back().getCellIndex() * 10);
	}

	vector<long int>::iterator neighbourOffsetsIter;

	unsigned long cellIndex = getCellIndexOfMolecule(m1);
	ParticleCell& currentCell = _cells[cellIndex];

	double oldEnergy = global_simulation->getDomain()->getLocalUpot();

	cellProcessor.initTraversal(_maxNeighbourOffset + _minNeighbourOffset + 1 + 1); // one more +1 for the dummyCell

	cellProcessor.preprocessCell(dummyCell);

	// extend the window of cells with cache activated
	for (unsigned int windowCellIndex = cellIndex - _minNeighbourOffset;
			windowCellIndex < cellIndex + _maxNeighbourOffset + 1;
			windowCellIndex++) {
		cellProcessor.preprocessCell(_cells[windowCellIndex]);
		// TODO: this will activate more cells than necessary?
	}

	cellProcessor.processCellPair(dummyCell, currentCell);

	// forward neighbours
	for (neighbourOffsetsIter = _forwardNeighbourOffsets.begin();
			neighbourOffsetsIter != _forwardNeighbourOffsets.end();
			neighbourOffsetsIter++) {
		ParticleCell& neighbourCell = _cells[cellIndex + *neighbourOffsetsIter];
		cellProcessor.processCellPair(dummyCell, neighbourCell);
	}
	// backward neighbours
	for (neighbourOffsetsIter = _backwardNeighbourOffsets.begin();
			neighbourOffsetsIter != _backwardNeighbourOffsets.end();
			neighbourOffsetsIter++) {
		ParticleCell& neighbourCell = _cells[cellIndex - *neighbourOffsetsIter]; // minus oder plus?
		cellProcessor.processCellPair(dummyCell, neighbourCell);
	}

	// close the window of cells activated
	for (unsigned int windowCellIndex = cellIndex - _minNeighbourOffset;
			windowCellIndex < cellIndex + _maxNeighbourOffset + 1;
			windowCellIndex++) {
		cellProcessor.postprocessCell(_cells[windowCellIndex]);
	}
	cellProcessor.postprocessCell(dummyCell);

	cellProcessor.endTraversal();
	double newEnergy = global_simulation->getDomain()->getLocalUpot();

	return newEnergy - oldEnergy;
}

int LinkedCells::grandcanonicalBalance(DomainDecompBase* comm) {
	comm->collCommInit(1);
	comm->collCommAppendInt(this->_localInsertionsMinusDeletions);
	comm->collCommAllreduceSum();
	int universalInsertionsMinusDeletions = comm->collCommGetInt();
	comm->collCommFinalize();
	return universalInsertionsMinusDeletions;
}

void LinkedCells::grandcanonicalStep(ChemicalPotential* mu, double T,
		Domain* domain, CellProcessor& cellProcessor) {
	bool accept = true;
	double DeltaUpot;
	Molecule* m;
	ParticlePairs2PotForceAdapter particlePairsHandler(*domain);

	this->_localInsertionsMinusDeletions = 0;

	mu->submitTemperature(T);
	double minco[3];
	double maxco[3];
	for (int d = 0; d < 3; d++) {
		minco[d] = this->getBoundingBoxMin(d);
		maxco[d] = this->getBoundingBoxMax(d);
	}

	bool hasDeletion = true;
	bool hasInsertion = true;
	double ins[3];
	unsigned nextid = 0;
	while (hasDeletion || hasInsertion) {
		if (hasDeletion)
			hasDeletion = mu->getDeletion(this, minco, maxco);
		if (hasDeletion) {
			m = *_particleIterator;
			DeltaUpot = -1.0
					* getEnergy(&particlePairsHandler, m, cellProcessor);

			accept = mu->decideDeletion(DeltaUpot / T);
#ifndef NDEBUG
			if (accept)
				global_log->debug() << "r" << mu->rank() << "d" << m->id()
						<< endl;
			else
				global_log->debug() << "   (r" << mu->rank() << "-d" << m->id()
						<< ")" << endl;
#endif
			if (accept) {
//				m->upd_cache(); TODO what to do here?
				// reset forces and momenta to zero
				{
					double zeroVec[3] = { 0.0, 0.0, 0.0 };
					m->setF(zeroVec);
					m->setM(zeroVec);
				}

				mu->storeMolecule(*m);

				this->deleteMolecule(m->id(), m->r(0), m->r(1), m->r(2));
				//this->_particleIterator = this->begin();
				this->begin(); // will set _particleIterator to the beginning
				this->_localInsertionsMinusDeletions--;
			}
		}

		if (!mu->hasSample()) {
			m = this->begin();
			//old: m = &(*(_particles.begin()));
			bool rightComponent = false;
			MoleculeIterator mit;
			if (m->componentid() != mu->getComponentID()) {
				for (mit = this->begin(); mit != this->end(); mit =
						this->next()) {
					if (mit->componentid() == mu->getComponentID()) {
						rightComponent = true;
						break;
					}
				}
			}
			if (rightComponent)
				m = &(*(mit));
			mu->storeMolecule(*m);
		}
		if (hasInsertion) {
			nextid = mu->getInsertion(ins);
			hasInsertion = (nextid > 0);
		}
		if (hasInsertion) {
			Molecule tmp = mu->loadMolecule();
			for (int d = 0; d < 3; d++)
				tmp.setr(d, ins[d]);
			tmp.setid(nextid);
			// TODO: I (tchipevn) am changing the former code
			// (add particle, compute energy and if rejected, delete particle)
			// to: add particle only if accepted
//			_particles.push_back(tmp);

//			std::list<Molecule>::iterator mit = _particles.end();
			Molecule * mit = &tmp;
			m = &(*mit);
//			m->upd_cache(); TODO: what to do here?
			// reset forces and torques to zero
			if (!mu->isWidom()) {
				double zeroVec[3] = { 0.0, 0.0, 0.0 };
				m->setF(zeroVec);
				m->setM(zeroVec);
			}
			m->check(nextid);
#ifndef NDEBUG
			global_log->debug() << "rank " << mu->rank() << ": insert "
					<< m->id() << " at the reduced position (" << ins[0] << "/"
					<< ins[1] << "/" << ins[2] << ")? " << endl;
#endif

//			unsigned long cellid = this->getCellIndexOfMolecule(m);
//			this->_cells[cellid].addParticle(m);
			DeltaUpot = getEnergy(&particlePairsHandler, m, cellProcessor);
			domain->submitDU(mu->getComponentID(), DeltaUpot, ins);
			accept = mu->decideInsertion(DeltaUpot / T);

#ifndef NDEBUG
			if (accept)
				global_log->debug() << "r" << mu->rank() << "i" << mit->id()
						<< ")" << endl;
			else
				global_log->debug() << "   (r" << mu->rank() << "-i"
						<< mit->id() << ")" << endl;
#endif
			if (accept) {
				this->_localInsertionsMinusDeletions++;
				addParticle(tmp);
			} else {
				// this->deleteMolecule(m->id(), m->r(0), m->r(1), m->r(2));
//				this->_cells[cellid].deleteMolecule(m->id());

				mit->check(m->id());
//				this->_particles.erase(mit);
			}
		}
	}
	for (m = this->begin(); m != this->end(); m = this->next()) {
#ifndef NDEBUG
		m->check(m->id());
#endif
	}
}


void LinkedCells::updateInnerMoleculeCaches(){
	for (ParticleCell& cell : _cells){
		if(cell.isInnerCell()){
			cell.buildSoACaches();
		}
	}
}

void LinkedCells::updateBoundaryAndHaloMoleculeCaches(){
	for (ParticleCell& cell : _cells) {
		if (cell.isHaloCell() or cell.isBoundaryCell()) {
			cell.buildSoACaches();
		}
	}
}

void LinkedCells::updateMoleculeCaches() {
	for (ParticleCell& cell : _cells) {
		cell.buildSoACaches();
	}
}
