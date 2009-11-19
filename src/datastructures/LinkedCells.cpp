#include "datastructures/LinkedCells.h"
#include "molecules/potforce.h"
#include "datastructures/handlerInterfaces/ParticlePairsHandler.h"
#include "Cell.h"
#include "molecules/Molecule.h"
#include "parallel/DomainDecompBase.h"
#include "ensemble/GrandCanonical.h"
#include "Domain.h"

#include <cmath>
#include <iostream>

using namespace std;

//################################################
//############ PUBLIC METHODS ####################
//################################################


LinkedCells::LinkedCells(
   double bBoxMin[3], double bBoxMax[3], double cutoffRadius,
   double tersoffCutoffRadius, double cellsInCutoffRadius,
   ParticlePairsHandler& partPairsHandler
): ParticleContainer(partPairsHandler, bBoxMin, bBoxMax)
{
  int numberOfCells = 1;
  _cutoffRadius = cutoffRadius;
  this->_tersoffCutoffRadius = tersoffCutoffRadius;
  for(int dim=0; dim<3; dim++){
    _haloWidthInNumCells[dim] = (int) ceil(cellsInCutoffRadius);
    _cellsPerDimension[dim] = (int) floor((this->_boundingBoxMax[dim]-this->_boundingBoxMin[dim])/(cutoffRadius/cellsInCutoffRadius))
      + 2 * _haloWidthInNumCells[dim];
    // in each dimension at least one layer of (inner+boundary) cells necessary
    if(_cellsPerDimension[dim] == 2 * _haloWidthInNumCells[dim]){
      _cellsPerDimension[dim]++;
    }
    numberOfCells *= _cellsPerDimension[dim];
    _cellLength[dim]=(this->_boundingBoxMax[dim]-this->_boundingBoxMin[dim])/(_cellsPerDimension[dim]-2*_haloWidthInNumCells[dim]);
    _haloBoundingBoxMin[dim] = this->_boundingBoxMin[dim]-_haloWidthInNumCells[dim]*_cellLength[dim];
    _haloBoundingBoxMax[dim] = this->_boundingBoxMax[dim]+_haloWidthInNumCells[dim]*_cellLength[dim];
    _haloLength[dim] = _haloWidthInNumCells[dim]*_cellLength[dim];
  }

  _cells.resize(numberOfCells);
 
  // If the with of the inner region is less than the width of the halo region
  // a parallelisation isn't possible (with the used algorithms).
  // In this case, print an error message
  // _cellsPerDimension is 2 times the halo width + the inner width
  // so it has to be at least 3 times the halo width
  if(_cellsPerDimension[0] < 3*_haloWidthInNumCells[0] ||
     _cellsPerDimension[1] < 3*_haloWidthInNumCells[1] ||
     _cellsPerDimension[2] < 3*_haloWidthInNumCells[2]){
    cerr << "Error in LinkedCells (constructor): bounding box too small for calculated cell length" << endl;
    cerr << "cellsPerDimension" << _cellsPerDimension[0] << " / " << _cellsPerDimension[1] << " / " << _cellsPerDimension[2] << endl;
    cerr << "_haloWidthInNumCells" << _haloWidthInNumCells[0] << " / " << _haloWidthInNumCells[1] << " / " << _haloWidthInNumCells[2] << endl;
    exit(5);
  }
  this->_localInsertionsMinusDeletions = 0;
 
  initializeCells();
  calculateNeighbourIndices();
  _cellsValid = false;
}


LinkedCells::~LinkedCells(){
}

void LinkedCells::rebuild(double bBoxMin[3], double bBoxMax[3]){
  for(int i=0; i<3; i++){
    this->_boundingBoxMin[i] = bBoxMin[i];
    this->_boundingBoxMax[i] = bBoxMax[i];
  }

  int numberOfCells = 1;

  for(int dim=0; dim<3; dim++){
    _cellsPerDimension[dim] = (int) floor((this->_boundingBoxMax[dim]-this->_boundingBoxMin[dim])/(_cutoffRadius/_haloWidthInNumCells[dim]))
      + 2 * _haloWidthInNumCells[dim];
    // in each dimension at least one layer of (inner+boundary) cells necessary
    if(_cellsPerDimension[dim] == 2 * _haloWidthInNumCells[dim]){
      cerr << "ERROR in LinkedCells::rebuild: region to small" << endl;
      exit(1);
    }
    numberOfCells *= _cellsPerDimension[dim];
    _cellLength[dim]=(this->_boundingBoxMax[dim]-this->_boundingBoxMin[dim])/(_cellsPerDimension[dim]-2*_haloWidthInNumCells[dim]);
    _haloBoundingBoxMin[dim] = this->_boundingBoxMin[dim]-_haloWidthInNumCells[dim]*_cellLength[dim];
    _haloBoundingBoxMax[dim] = this->_boundingBoxMax[dim]+_haloWidthInNumCells[dim]*_cellLength[dim];
    _haloLength[dim] = _haloWidthInNumCells[dim]*_cellLength[dim];
  }

  _cells.resize(numberOfCells);
 
  // If the with of the inner region is less than the width of the halo region
  // a parallelisation isn't possible (with the used algorithms).
  // In this case, print an error message
  // _cellsPerDimension is 2 times the halo width + the inner width
  // so it has to be at least 3 times the halo width
  if(_cellsPerDimension[0] < 3*_haloWidthInNumCells[0] ||
     _cellsPerDimension[1] < 3*_haloWidthInNumCells[1] ||
     _cellsPerDimension[2] < 3*_haloWidthInNumCells[2]){
    cerr << "Error in LinkedCells (rebuild): bounding box too small for calculated cell Length" << endl;
    cerr << "cellsPerDimension" << _cellsPerDimension[0] << " / " << _cellsPerDimension[1] << " / " << _cellsPerDimension[2] << endl;
    cerr << "_haloWidthInNumCells" << _haloWidthInNumCells[0] << " / " << _haloWidthInNumCells[1] << " / " << _haloWidthInNumCells[2] << endl;
    exit(5);
  }
 
  initializeCells();
  calculateNeighbourIndices();
  
  // delete all Particles which are outside of the halo region
  std::list<Molecule>::iterator particleIterator = _particles.begin();
  bool erase_mol;
  while(particleIterator!=_particles.end()){
    erase_mol = false;
    for(unsigned short d=0;d<3;++d){
      const double& rd=particleIterator->r(d);
      // The molecules has to be within the domain of the process
      // If it is outside in at least one dimension, it has to be
      // erased /
      if(rd<this->_haloBoundingBoxMin[d] || rd>=this->_haloBoundingBoxMax[d]) erase_mol = true;
    }
    if(erase_mol) {
      particleIterator=_particles.erase(particleIterator);
    }
    else{
      particleIterator++;
    }
  }
  _cellsValid = false;
}

void LinkedCells::update(){
  // clear all Cells
  std::vector<Cell>::iterator celliter;
  for(celliter=(_cells).begin();celliter!=(_cells).end();++celliter){
    (*celliter).removeAllParticles();
  }
  
  unsigned long index; // index of the cell into which the pointer has to be inserted
  std::list<Molecule>::iterator pos;
  for(pos=_particles.begin();pos!=_particles.end();++pos) {
    index=getCellIndexOfMolecule(&(*pos));
    if(index < 0 || index >= _cells.size())
    {
      cout << "\nSEVERE ERROR\n";
      cout << "ID " << pos->id() << "\nr:" << pos->r(0) << " / "  << pos->r(1) << " / "  << pos->r(2) << endl;
      cout << "v:" << pos->v(0) << " / "  << pos->v(1) << " / "  << pos->v(2) << endl;
      cout << "F:" << pos->F(0) << " / "  << pos->F(1) << " / "  << pos->F(2) << endl;
      cout << "Cell: " << index << endl;
      cout << "_cells.size(): " << _cells.size() << "\tindex: " << index << "\n";
      cout << "Length of the cell: " << _cellLength[0] << " " << _cellLength[1] << " " << _cellLength[2] << "\n";
      cout << "\nBounding box, including halo, from (" << _haloBoundingBoxMin[0] << "/"
           << _haloBoundingBoxMin[1] << "/" << _haloBoundingBoxMin[2] << ") to (" << _haloBoundingBoxMax[0]
           << "/" << _haloBoundingBoxMax[1] << "/" << _haloBoundingBoxMax[2] << ").\n";
      cout << "removing m" << pos->id() << " (internal component number " << pos->componentid()
           << ", Utrans=" << pos->Utrans() << ").\n";
      cout << "\n\n";
      exit(1);
    }
    (_cells[index]).addParticle(&(*pos));
  }
  _cellsValid = true;
}



void LinkedCells::addParticle(Molecule& particle){
 
  double x = particle.r(0);
  double y = particle.r(1);
  double z = particle.r(2);

  if(x>=this->_haloBoundingBoxMin[0] && x < this->_haloBoundingBoxMax[0] && 
     y>=this->_haloBoundingBoxMin[1] && y < this->_haloBoundingBoxMax[1] && 
     z>=this->_haloBoundingBoxMin[2] && z < this->_haloBoundingBoxMax[2]){
    _particles.push_front(particle);
    if(_cellsValid){
      int cellIndex=getCellIndexOfMolecule(&particle);
      if(cellIndex < (int) 0 || cellIndex >= (int) _cells.size()){
        cerr << "Error in LinkedCells::addParticle(): INDEX ERROR" << endl;
        exit(1);
      }
      (_cells[cellIndex]).addParticle(&(_particles.front()));
    }
  }
}

void LinkedCells::countParticles(Domain* d)
{
  vector<Cell>::iterator cellIter;
  std::list<Molecule*>::iterator molIter1;
  for(unsigned i=0; i < _cells.size(); i++)
  {
    Cell& currentCell = _cells[i];
    if(currentCell.isHaloCell()) continue;
    for(molIter1=currentCell.getParticlePointers().begin(); molIter1!=currentCell.getParticlePointers().end(); molIter1++)
    {
      Molecule& molecule1 = **molIter1;
      d->observeRDF(molecule1.componentid());
    }
  }
}
unsigned LinkedCells::countParticles(int cid) 
{
   unsigned N = 0;
   vector<Cell>::iterator cellIter;
   std::list<Molecule*>::iterator molIter1;
   for(unsigned i=0; i < _cells.size(); i++)
   {
      Cell& currentCell = _cells[i];
      if(currentCell.isHaloCell()) continue;
      for( molIter1=currentCell.getParticlePointers().begin();
           molIter1!=currentCell.getParticlePointers().end();
           molIter1++ )
      {
         if((*molIter1)->componentid() == cid) N++;
      }
   }
   return N;
}
unsigned LinkedCells::countParticles(
   int cid, double* cbottom, double* ctop
) {
   int minIndex[3];
   int maxIndex[3];
   for(int d=0; d < 3; d++)
   {
      if(cbottom[d] < this->_haloBoundingBoxMin[d]) minIndex[d] = 0;
      else minIndex[d] = (int)floor(
         (cbottom[d] - this->_haloBoundingBoxMin[d]) / _cellLength[d]
      );
      if(ctop[d] > this->_haloBoundingBoxMax[d])
	 maxIndex[d] = (int)floor(
	    (this->_haloBoundingBoxMax[d] - _haloBoundingBoxMin[d])
	       / this->_cellLength[d]
         );
      else maxIndex[d] = (int)floor(
         (ctop[d] - this->_haloBoundingBoxMin[d]) / _cellLength[d]
      );

      if(minIndex[d] < 0) minIndex[d] = 0;
      if(maxIndex[d] >= _cellsPerDimension[d]) maxIndex[d] = _cellsPerDimension[d] - 1;
   }
   
   unsigned N = 0;
   int cix[3];
   vector<Cell>::iterator cellIter;
   std::list<Molecule*>::iterator molIter1;
   bool individualCheck;
   int cellid;

   for(cix[0] = minIndex[0]; maxIndex[0] >= cix[0]; (cix[0])++)
      for(cix[1] = minIndex[1]; maxIndex[1] >= cix[1]; (cix[1])++)
	 for(cix[2] = minIndex[2]; maxIndex[2] >= cix[2]; (cix[2])++)
	 {
	    individualCheck = (cix[0] == minIndex[0]) || (cix[0] == minIndex[0] + 1) ||
	                      (cix[0] == maxIndex[0]) || (cix[0] == maxIndex[0] - 1) ||
	                      (cix[1] == minIndex[1]) || (cix[1] == minIndex[1] + 1) ||
	                      (cix[1] == maxIndex[1]) || (cix[1] == maxIndex[1] - 1) ||
	                      (cix[2] == minIndex[2]) || (cix[2] == minIndex[2] + 1) ||
	                      (cix[2] == maxIndex[2]) || (cix[2] == maxIndex[2] - 1);
	    cellid = this->cellIndexOf3DIndex(cix[0], cix[1], cix[2]);
            Cell& currentCell = _cells[cellid];
            if(currentCell.isHaloCell()) continue;
	    if(individualCheck)
	    {
               for(molIter1=currentCell.getParticlePointers().begin();
                   molIter1!=currentCell.getParticlePointers().end();
                   molIter1++)
	       {
                  if( ((*molIter1)->r(0) > cbottom[0]) &&
		      ((*molIter1)->r(1) > cbottom[1]) &&
		      ((*molIter1)->r(2) > cbottom[2]) &&
		      ((*molIter1)->r(0) < ctop[0]) &&
		      ((*molIter1)->r(1) < ctop[1]) &&
		      ((*molIter1)->r(2) < ctop[2]) &&
		      ((*molIter1)->componentid() == cid) )
		  {
		     N++;
		  }
	       }
	    }
	    else
	    {
               for(molIter1=currentCell.getParticlePointers().begin();
                   molIter1!=currentCell.getParticlePointers().end();
                   molIter1++)
	       {
                  if((*molIter1)->componentid() == cid) N++;
	       }
	    }
	 }
   
   return N;
}

void LinkedCells::traversePairs(){
  if(_cellsValid == false) {
    cerr << "Cell structure in LinkedCells (traversePairs) invalid, call update first" << endl;
    exit(1);
  }
  this->_particlePairsHandler.init();
  
  // XXX comment
  double distanceVector[3];
  // loop over all cells
  vector<Cell>::iterator cellIter;
  std::list<Molecule*>::iterator molIter1;
  std::list<Molecule*>::iterator molIter2;
  for(cellIter=_cells.begin(); cellIter!= _cells.end(); cellIter++){
    for(molIter1=cellIter->getParticlePointers().begin(); molIter1!=cellIter->getParticlePointers().end(); molIter1++){
      (*molIter1)->setFM(0,0,0,0,0,0);
    }
  } 


  vector<unsigned long>::iterator cellIndexIter;
  vector<unsigned long>::iterator neighbourOffsetsIter;
  
  // sqare of the cutoff radius
  double cutoffRadiusSquare = _cutoffRadius * _cutoffRadius; 
  double tersoffCutoffRadiusSquare = this->_tersoffCutoffRadius
                                   * this->_tersoffCutoffRadius;
				   
#ifndef NDEBUG
  cout << "disconnecting Tersoff pairs.\n";
#endif
  for(unsigned i=0; i < _cells.size(); i++)
  {
    Cell& currentCell = _cells[i];
    for( molIter1=currentCell.getParticlePointers().begin();
         molIter1!=currentCell.getParticlePointers().end();
         molIter1++ )
    {
      Molecule& molecule1 = **molIter1;
      molecule1.clearTersoffNeighbourList();
    }
  }

#ifndef NDEBUG 
  cout << "processing pairs and preprocessing Tersoff pairs.\n"; //X
#endif

  // loop over all inner cells and calculate forces to forward neighbours
  for(cellIndexIter=_innerCellIndices.begin(); cellIndexIter!=_innerCellIndices.end(); cellIndexIter++){
    Cell& currentCell = _cells[*cellIndexIter];
    // forces between molecules in the cell
    for(molIter1=currentCell.getParticlePointers().begin(); molIter1!=currentCell.getParticlePointers().end(); molIter1++){
      Molecule& molecule1 = **molIter1;
      if(molecule1.numTersoff() == 0)
      {
         for(molIter2=molIter1; molIter2!=currentCell.getParticlePointers().end(); molIter2++){
           Molecule& molecule2 = **molIter2;
	   double dd = molecule2.dist2(molecule1,distanceVector);
           if( (&molecule1 != &molecule2) &&
	        (dd < cutoffRadiusSquare) )
	   {
             this->_particlePairsHandler.processPair(molecule1,molecule2,distanceVector,0, dd);
           }
         }
      }
      else
      {
         for(molIter2=molIter1; molIter2!=currentCell.getParticlePointers().end(); molIter2++)
	 {
           Molecule& molecule2 = **molIter2;
           if(&molecule1 != &molecule2)
	   {
	      double dd = molecule2.dist2(molecule1,distanceVector);
	      if(dd < cutoffRadiusSquare)
	      {
                 this->_particlePairsHandler.processPair(molecule1,molecule2,distanceVector,0, dd);
		 if((molecule2.numTersoff() > 0) && (dd < tersoffCutoffRadiusSquare))
		 {
		    this->_particlePairsHandler.preprocessTersoffPair(molecule1,molecule2,false);
		 }
              }
	   }
         }
      }
    }
    // loop over all neighbours
    for(neighbourOffsetsIter=_forwardNeighbourOffsets.begin(); neighbourOffsetsIter!=_forwardNeighbourOffsets.end(); neighbourOffsetsIter++)
    {
       Cell& neighbourCell = _cells[*cellIndexIter+*neighbourOffsetsIter];
       // loop over all particles in the cell
       for( molIter1=currentCell.getParticlePointers().begin();
            molIter1!=currentCell.getParticlePointers().end();
            molIter1++ )
       {
          Molecule& molecule1 = **molIter1;
	  if(molecule1.numTersoff() == 0)
	  {
             for( molIter2=neighbourCell.getParticlePointers().begin();
	          molIter2!=neighbourCell.getParticlePointers().end();
	          molIter2++ )
             {
                Molecule& molecule2 = **molIter2;
		double dd = molecule2.dist2(molecule1,distanceVector);
                if(dd < cutoffRadiusSquare)
		{
                   this->_particlePairsHandler.processPair(molecule1,molecule2,distanceVector,0, dd);
                }
             }
	  }
	  else
	  {
             for( molIter2=neighbourCell.getParticlePointers().begin();
	          molIter2!=neighbourCell.getParticlePointers().end();
	          molIter2++ )
             {
                Molecule& molecule2 = **molIter2;
		double dd = molecule2.dist2(molecule1,distanceVector);
                if(dd < cutoffRadiusSquare)
		{
                   this->_particlePairsHandler.processPair(molecule1,molecule2,distanceVector,0, dd);
		   if((molecule2.numTersoff() > 0) && (dd < tersoffCutoffRadiusSquare))
		   {
		      this->_particlePairsHandler.preprocessTersoffPair(molecule1,molecule2,false);
		   }
                }
             }
	  }
       }
    }
  }

  // loop over halo cells and detect Tersoff neighbours within the halo
  // this is relevant for the angle summation
  for(unsigned i=0; i < _cells.size(); i++)
  {
    Cell& currentCell = _cells[i];
    if(!currentCell.isHaloCell()) continue;

    for( molIter1=currentCell.getParticlePointers().begin();
         molIter1!=currentCell.getParticlePointers().end();
         molIter1++ )
    {
      Molecule& molecule1 = **molIter1;
      if(molecule1.numTersoff() == 0) continue;
      for(molIter2=molIter1; molIter2!=currentCell.getParticlePointers().end(); molIter2++)
      {
        Molecule& molecule2 = **molIter2;
        if((&molecule1 != &molecule2) && (molecule2.numTersoff() > 0))
        { 
          double dd = molecule2.dist2(molecule1, distanceVector);
          if(dd < tersoffCutoffRadiusSquare)
            this->_particlePairsHandler.preprocessTersoffPair(molecule1, molecule2, true);
        }
      }

      for( neighbourOffsetsIter=_forwardNeighbourOffsets.begin();
           neighbourOffsetsIter!=_forwardNeighbourOffsets.end();
           neighbourOffsetsIter++ )
      {
        int j = i + *neighbourOffsetsIter;
        if((j < 0) || (j >= (int)(this->_cells.size()))) continue;
        Cell& neighbourCell = _cells[j];
        if(!neighbourCell.isHaloCell()) continue;
        for( molIter2=neighbourCell.getParticlePointers().begin();
             molIter2!=neighbourCell.getParticlePointers().end();
             molIter2++ )
        {
          Molecule& molecule2 = **molIter2;
          if(molecule2.numTersoff() == 0) continue;
          double dd = molecule2.dist2(molecule1, distanceVector);
          if(dd < tersoffCutoffRadiusSquare)
            this->_particlePairsHandler.preprocessTersoffPair(molecule1, molecule2, true);
        }
      }
    }
  }

  // loop over all boundary cells and calculate forces to forward and backward neighbours
  for( cellIndexIter=_boundaryCellIndices.begin();
       cellIndexIter!=_boundaryCellIndices.end();
       cellIndexIter++ )
  {
     Cell& currentCell = _cells[*cellIndexIter];
     // forces between molecules in the cell
     for( molIter1=currentCell.getParticlePointers().begin();
          molIter1!=currentCell.getParticlePointers().end();
          molIter1++ )
     {
        Molecule& molecule1 = **molIter1;
	if(molecule1.numTersoff() == 0)
	{
           for( molIter2=molIter1;
	        molIter2!=currentCell.getParticlePointers().end();
	        molIter2++ )
	   {
              Molecule& molecule2 = **molIter2;
	      double dd = molecule2.dist2(molecule1,distanceVector);
              if( (&molecule1 != &molecule2) &&
		  (dd < cutoffRadiusSquare) )
	      {
                 this->_particlePairsHandler.processPair(molecule1,molecule2,distanceVector,0, dd);      
              }
           }
	}
	else
	{
           for( molIter2=molIter1;
	        molIter2!=currentCell.getParticlePointers().end();
	        molIter2++ )
	   {
              Molecule& molecule2 = **molIter2;
	      if(&molecule1 != &molecule2)
	      {
                 double dd = molecule2.dist2(molecule1,distanceVector);
		 if(dd < cutoffRadiusSquare)
		 {
                    this->_particlePairsHandler.processPair(molecule1,molecule2,distanceVector,0, dd); 
		    if((molecule2.numTersoff() > 0) && (dd < tersoffCutoffRadiusSquare))
		    {
		       this->_particlePairsHandler.preprocessTersoffPair(molecule1,molecule2,false);
		    }
		 }
	      }
	   }
	}
     }

     // loop over all forward neighbours
     for( neighbourOffsetsIter=_forwardNeighbourOffsets.begin();
          neighbourOffsetsIter!=_forwardNeighbourOffsets.end(); 
          neighbourOffsetsIter++ )
     {
        Cell& neighbourCell = _cells[*cellIndexIter+*neighbourOffsetsIter];
        // loop over all particles in the cell
        for( molIter1=currentCell.getParticlePointers().begin();
             molIter1!=currentCell.getParticlePointers().end();
	     molIter1++ )
        {
           Molecule& molecule1 = **molIter1;
	   if(molecule1.numTersoff() == 0)
	   {
              for( molIter2=neighbourCell.getParticlePointers().begin();
	           molIter2!=neighbourCell.getParticlePointers().end();
	           molIter2++ )
	      {
                 Molecule& molecule2 = **molIter2;
		 double dd = molecule2.dist2(molecule1,distanceVector);
                 if(dd < cutoffRadiusSquare)
	         {
                    if(neighbourCell.isHaloCell() && !isFirstParticle(molecule1, molecule2))
		    {
                       this->_particlePairsHandler.processPair(molecule1,molecule2,distanceVector,1,dd);
                    }
                    else
		    {
                       this->_particlePairsHandler.processPair(molecule1,molecule2,distanceVector,0,dd);
                    }
                 }
              }
	   }
	   else
	   {
              for( molIter2=neighbourCell.getParticlePointers().begin();
	           molIter2!=neighbourCell.getParticlePointers().end();
	           molIter2++ )
	      {
                 Molecule& molecule2 = **molIter2;
		 double dd = molecule2.dist2(molecule1,distanceVector);
                 if(dd < cutoffRadiusSquare)
	         {
		    int cd = 0;
                    if(neighbourCell.isHaloCell() && !isFirstParticle(molecule1, molecule2))
		    {
                       cd = 1;
                    }
		    this->_particlePairsHandler.processPair(molecule1,molecule2,distanceVector,cd,dd);
		    if((molecule2.numTersoff() > 0) && (dd < tersoffCutoffRadiusSquare))
		    {
		       this->_particlePairsHandler.preprocessTersoffPair(
		          molecule1, molecule2, (cd == 1)
		       );
		    }
                 }
              }
	   }
        }
     }

     // loop over all backward neighbours. calculate only forces
     // to neighbour cells in the halo region, all others already have been calculated
     for( neighbourOffsetsIter=_backwardNeighbourOffsets.begin();
          neighbourOffsetsIter!=_backwardNeighbourOffsets.end();
          neighbourOffsetsIter++)
     {
        Cell& neighbourCell = _cells[*cellIndexIter+*neighbourOffsetsIter];
        if(neighbourCell.isHaloCell())
	{
           // loop over all particles in the cell
           for( molIter1=currentCell.getParticlePointers().begin();
	        molIter1!=currentCell.getParticlePointers().end();
	        molIter1++ )
	   {
              Molecule& molecule1 = **molIter1;
              if(molecule1.numTersoff() == 0)
	      {
                 for( molIter2=neighbourCell.getParticlePointers().begin();
	              molIter2!=neighbourCell.getParticlePointers().end();
	              molIter2++ )
	         {
                    Molecule& molecule2 = **molIter2;
		    double dd = molecule2.dist2(molecule1,distanceVector);
                    if(dd < cutoffRadiusSquare)
		    {
                       if(isFirstParticle(molecule1, molecule2))
		       {
                          this->_particlePairsHandler.processPair(molecule1,molecule2,distanceVector,0,dd);
                       }
                       else
		       {
                          this->_particlePairsHandler.processPair(molecule1,molecule2,distanceVector,1,dd);
                       }
                    }
                 }
	      }
	      else
	      {
                 for( molIter2=neighbourCell.getParticlePointers().begin();
	              molIter2!=neighbourCell.getParticlePointers().end();
	              molIter2++ )
	         {
                    Molecule& molecule2 = **molIter2;
		    double dd = molecule2.dist2(molecule1,distanceVector);
                    if(dd < cutoffRadiusSquare)
		    {
		       int cd = 0;
                       if(isFirstParticle(molecule1, molecule2))
		       {
			  cd = 1;
		       }
                       this->_particlePairsHandler.processPair(molecule1,molecule2,distanceVector,cd,dd);
		       if((molecule2.numTersoff() > 0) && (dd < tersoffCutoffRadiusSquare))
		       {
		          this->_particlePairsHandler.preprocessTersoffPair(
   		             molecule1, molecule2, (cd == 1)
		          );
		       }
                    }
                 }
	      }
           }
        }
     }
  }

#ifndef NDEBUG
  cout << "processing Tersoff potential.\n";  //X
#endif
  double params[15];
  double delta_r;
  bool knowparams = false;

  for( cellIndexIter=_innerCellIndices.begin();
       cellIndexIter!=_boundaryCellIndices.end(); 
       cellIndexIter++ )
  {
    if(cellIndexIter == this->_innerCellIndices.end())
      cellIndexIter = this->_boundaryCellIndices.begin();
    Cell& currentCell = _cells[*cellIndexIter];
    for( molIter1=currentCell.getParticlePointers().begin();
         molIter1!=currentCell.getParticlePointers().end();
         molIter1++ )
    {
      Molecule& molecule1 = **molIter1;
      if(molecule1.numTersoff() == 0) continue;
      if(!knowparams)
      {
         delta_r = molecule1.tersoffParameters(params);
         knowparams = true;
      }
      this->_particlePairsHandler.processTersoffAtom(molecule1, params, delta_r);
    }
  }

  this->_particlePairsHandler.finish();
}


unsigned long LinkedCells::getNumberOfParticles(){
  return _particles.size(); 
}      


Molecule* LinkedCells::begin(){
  _particleIter = _particles.begin();
  if(_particleIter != _particles.end()){
    return &(*_particleIter);
  }
  else {
    return NULL;
  }
}


Molecule* LinkedCells::next(){
  _particleIter++;
  if(_particleIter != _particles.end()){
    return &(*_particleIter);
  }
  else {
    return NULL;
  }
}


Molecule* LinkedCells::end(){
  return NULL;
}


void LinkedCells::deleteOuterParticles(){ 
  if(_cellsValid == false) {
    cerr << "Cell structure in LinkedCells (deleteOuterParticles) invalid, call update first" << endl;
    exit(1);
  }
  
  vector<unsigned long>::iterator cellIndexIter;
  //std::list<Molecule*>::iterator molIter1;
  for(cellIndexIter=_haloCellIndices.begin(); cellIndexIter!=_haloCellIndices.end(); cellIndexIter++){
    Cell& currentCell = _cells[*cellIndexIter];
    currentCell.removeAllParticles();
  }
  
  std::list<Molecule>::iterator particleIterator = _particles.begin();
  bool erase_mol;
  while(particleIterator!=_particles.end()){
    erase_mol = false;
    for(unsigned short d=0;d<3;++d){
      const double& rd=particleIterator->r(d);
      // The molecules has to be within the domain of the process
      // If it is outside in at least one dimension, it has to be
      // erased /
      if(rd<this->_boundingBoxMin[d] || rd>=this->_boundingBoxMax[d]) erase_mol = true;
    }
    if(erase_mol) {
      particleIterator=_particles.erase(particleIterator);
    }
    else{
      particleIterator++;
    }
  }
}


double LinkedCells::get_halo_L(int index){
  return _haloLength[index]; 
}


void LinkedCells::getBoundaryParticles(list<Molecule*> &boundaryParticlePtrs){
  if(_cellsValid == false) {
    cerr << "Cell structure in LinkedCells (getBoundaryParticles) invalid, call update first" << endl;
    exit(1);
  }

  std::list<Molecule*>::iterator particleIter;
  vector<unsigned long>::iterator cellIndexIter;
  
  // loop over all boundary cells
  for(cellIndexIter=_boundaryCellIndices.begin(); cellIndexIter!=_boundaryCellIndices.end(); cellIndexIter++){
    Cell& currentCell = _cells[*cellIndexIter];
    // loop over all molecules in the cell
    for(particleIter=currentCell.getParticlePointers().begin(); particleIter!=currentCell.getParticlePointers().end(); particleIter++){
      boundaryParticlePtrs.push_back(*particleIter);
    }
  }
}

void LinkedCells::getHaloParticles(list<Molecule*> &haloParticlePtrs){
  if(_cellsValid == false) {
    cerr << "Cell structure in LinkedCells (getHaloParticles) invalid, call update first" << endl;
    exit(1);
  }

  std::list<Molecule*>::iterator particleIter;
  vector<unsigned long>::iterator cellIndexIter;
  
  // loop over all halo cells
  for(cellIndexIter=_haloCellIndices.begin(); cellIndexIter!=_haloCellIndices.end(); cellIndexIter++){
    Cell& currentCell = _cells[*cellIndexIter];
    // loop over all molecules in the cell
    for(particleIter=currentCell.getParticlePointers().begin(); particleIter!=currentCell.getParticlePointers().end(); particleIter++){
      haloParticlePtrs.push_back(*particleIter);
    }
  }
}


void LinkedCells::getRegion(double lowCorner[3], double highCorner[3], list<Molecule*> &particlePtrs){
  if(_cellsValid == false) {
    cerr << "Cell structure in LinkedCells (getRegion) invalid, call update first" << endl;
    exit(1);
  }

  int startIndex[3];
  int stopIndex[3];
  int globalCellIndex;
  std::list<Molecule*>::iterator particleIter;
  
  for(int dim=0; dim<3; dim++){
    if(lowCorner[dim] < this->_boundingBoxMax[dim] && highCorner[dim] > this->_boundingBoxMin[dim]){
      startIndex[dim] = (int) floor((lowCorner[dim]-_haloBoundingBoxMin[dim])/_cellLength[dim]) - 1;
      stopIndex[dim] = (int) floor((highCorner[dim]-_haloBoundingBoxMin[dim])/_cellLength[dim]) + 1;
      if(startIndex[dim] < 0) startIndex[dim] = 0; 
      if(stopIndex[dim] > _cellsPerDimension[dim]-1) stopIndex[dim] = _cellsPerDimension[dim]-1;
    }
    else{
      // No Part of the given region is owned by this process
      // --> chose some startIndex which is higher than the stopIndex
      startIndex[dim] = 1;
      stopIndex[dim] = 0;
    }
  }

  for(int iz=startIndex[2]; iz <=stopIndex[2]; iz++){
    for(int iy=startIndex[1]; iy <=stopIndex[1]; iy++){
      for(int ix=startIndex[0]; ix <=stopIndex[0]; ix++){
        // globalCellIndex is the cellIndex of the molecule on the coarse Cell level.
        globalCellIndex=(iz*_cellsPerDimension[1]+iy) * _cellsPerDimension[0] + ix;
        // loop over all subcells (either 1 or 8)
        // traverse all molecules in the current cell
        for(particleIter=_cells[globalCellIndex].getParticlePointers().begin(); particleIter!=_cells[globalCellIndex].getParticlePointers().end(); particleIter++){
          if((*particleIter)->r(0) >= lowCorner[0] && (*particleIter)->r(0) < highCorner[0] &&
            (*particleIter)->r(1) >= lowCorner[1] && (*particleIter)->r(1) < highCorner[1] &&
            (*particleIter)->r(2) >= lowCorner[2] && (*particleIter)->r(2) < highCorner[2]) {
            particlePtrs.push_back(*particleIter);
          }
        }
      }
    }
  }
}

//################################################
//############ PRIVATE METHODS ###################
//################################################


void LinkedCells::initializeCells(){
  _innerCellIndices.clear();
  _boundaryCellIndices.clear();
  _haloCellIndices.clear();
  unsigned long cellIndex;
  for(int iz=0; iz<_cellsPerDimension[2]; ++iz) {
    for(int iy=0; iy<_cellsPerDimension[1]; ++iy) {
      for(int ix=0; ix<_cellsPerDimension[0]; ++ix) {
        cellIndex = cellIndexOf3DIndex(ix,iy,iz);
        if(ix < _haloWidthInNumCells[0] || iy < _haloWidthInNumCells[1] || iz < _haloWidthInNumCells[2] ||
           ix >= _cellsPerDimension[0]-_haloWidthInNumCells[0] ||
           iy >= _cellsPerDimension[1]-_haloWidthInNumCells[1] ||
           iz >= _cellsPerDimension[2]-_haloWidthInNumCells[2]){
          _cells[cellIndex].assingCellToHaloRegion();
          _haloCellIndices.push_back(cellIndex);
        }
        else if (ix < 2*_haloWidthInNumCells[0] || iy < 2*_haloWidthInNumCells[1] || iz < 2*_haloWidthInNumCells[2] ||
                 ix >= _cellsPerDimension[0]-2*_haloWidthInNumCells[0] ||
                 iy >= _cellsPerDimension[1]-2*_haloWidthInNumCells[1] ||
                 iz >= _cellsPerDimension[2]-2*_haloWidthInNumCells[2]){
          _cells[cellIndex].assignCellToBoundaryRegion();
          _boundaryCellIndices.push_back(cellIndex);
        }
        else {
          _cells[cellIndex].assignCellToInnerRegion();
          _innerCellIndices.push_back(cellIndex);
        }
      }
    }
  }
}


void LinkedCells::calculateNeighbourIndices(){ 
  _forwardNeighbourOffsets.clear();
  _backwardNeighbourOffsets.clear();
  double xDistanceSquare;
  double yDistanceSquare;
  double zDistanceSquare;
  double cutoffRadiusSquare = pow(_cutoffRadius,2);
  for(int zIndex=-_haloWidthInNumCells[2]; zIndex<=_haloWidthInNumCells[2]; zIndex++) {
    // The distance in one dimension is the width of a cell multiplied with the number 
    // of cells between the two cells (this is received by substracting one of the 
    // absolute difference of the cells, if this difference is not zero)
    if(zIndex != 0){
      zDistanceSquare = pow((abs(zIndex)-1) * _cellLength[2],2);
    }
    else {
      zDistanceSquare = 0;
    }
    for(int yIndex=-_haloWidthInNumCells[1]; yIndex<=_haloWidthInNumCells[1]; yIndex++) {
      if(yIndex != 0){
        yDistanceSquare = pow((abs(yIndex)-1) * _cellLength[1],2);
      }
      else {
        yDistanceSquare = 0;
      }
      for(int xIndex=-_haloWidthInNumCells[0]; xIndex<=_haloWidthInNumCells[0]; xIndex++) {
        if(xIndex != 0){
          xDistanceSquare = pow((abs(xIndex)-1) * _cellLength[0],2);
        }
        else {
          xDistanceSquare = 0;
        }
        if(xDistanceSquare+yDistanceSquare+zDistanceSquare <= cutoffRadiusSquare) {
          long offset = cellIndexOf3DIndex(xIndex, yIndex, zIndex);
          if(offset > 0){
            _forwardNeighbourOffsets.push_back(offset);
          }
          if(offset < 0){
            _backwardNeighbourOffsets.push_back(offset);
          }
        }
      }
    }
  }
}

unsigned long LinkedCells::getCellIndexOfMolecule(Molecule* molecule) {
  int cellIndex[3]; // 3D Cell index

  for(int dim=0; dim<3; dim++){
    if(molecule->r(dim) < _haloBoundingBoxMin[dim] || molecule->r(dim) >= _haloBoundingBoxMax[dim]){
      cerr << "Error in getCellIndexOfMolecule(Molecule* molecule): Molecule is outside of the bounding box" << endl;
    } 
    cellIndex[dim] = (int) floor((molecule->r(dim)-_haloBoundingBoxMin[dim])/_cellLength[dim]);

  }
  return (cellIndex[2]*_cellsPerDimension[1]+cellIndex[1]) * _cellsPerDimension[0] + cellIndex[0]; 
}


unsigned long LinkedCells::cellIndexOf3DIndex(int xIndex, int yIndex, int zIndex){
  return (zIndex * _cellsPerDimension[1] + yIndex) * _cellsPerDimension[0] + xIndex;

}

bool LinkedCells::isFirstParticle(Molecule& m1, Molecule& m2){
  if(m1.r(2) < m2.r(2)) return true;
  else if(m1.r(2) > m2.r(2)) return false;
  else {
    if(m1.r(1) < m2.r(1)) return true;
    else if(m1.r(1) > m2.r(1)) return false;
    else {
      if(m1.r(0) < m2.r(0)) return true;
      else if(m1.r(0) > m2.r(0)) return false;
      else {
        cerr << "Error in LinkedCells::isFirstParticle: both Particles have the same position" << endl;
        exit(1);
      }
    }
  }
}

void LinkedCells::deleteMolecule
(
   unsigned long molid, double x, double y, double z
)
{
   int ix = (int)floor((x - this->_haloBoundingBoxMin[0]) / this->_cellLength[0]);
   int iy = (int)floor((y - this->_haloBoundingBoxMin[1]) / this->_cellLength[1]);
   int iz = (int)floor((z - this->_haloBoundingBoxMin[2]) / this->_cellLength[2]);
   unsigned long hash = this->cellIndexOf3DIndex(ix, iy, iz);
   if((hash >= this->_cells.size()) || (hash < 0))
   {
      cout << "SEVERE ERROR: coordinates for atom deletion lie outside bounding box.\n";
      exit(1);
   }
   bool found = this->_cells[hash].deleteMolecule(molid);
   if(!found)
   {
      cout << "SEVERE ERROR: could not delete molecule " << molid << ".\n";
      exit(1);
   }
}

double LinkedCells::getEnergy(Molecule* m1)
{
   double u = 0.0;

   std::list<Molecule*>::iterator molIter2;
   vector<unsigned long>::iterator neighbourOffsetsIter;
  
   // sqare of the cutoffradius
   double cutoffRadiusSquare = _cutoffRadius * _cutoffRadius; 
   double dd;
   double distanceVector[3];

   unsigned long cellIndex = getCellIndexOfMolecule(m1);
   Cell& currentCell = _cells[cellIndex];

   if(m1->numTersoff() > 0)
   {
      cout << "The grand canonical ensemble is not implemented for solids.\n";
      exit(484);
   }
   // molecules in the cell
   for( molIter2 = currentCell.getParticlePointers().begin();
        molIter2 != currentCell.getParticlePointers().end();
        molIter2++ )
   {
      if(m1->id() == (*molIter2)->id()) continue;
      dd = (*molIter2)->dist2(*m1, distanceVector);
      if(dd > cutoffRadiusSquare) continue;
      u += this->_particlePairsHandler.processPair(*m1, **molIter2, distanceVector, 2, dd);
   }

   // backward and forward neighbours
   for( neighbourOffsetsIter = _backwardNeighbourOffsets.begin();
        neighbourOffsetsIter != _forwardNeighbourOffsets.end();
        neighbourOffsetsIter++ )
   {
      if(neighbourOffsetsIter == _backwardNeighbourOffsets.end())
         neighbourOffsetsIter = _forwardNeighbourOffsets.begin();
      Cell& neighbourCell = _cells[cellIndex + *neighbourOffsetsIter];
      for( molIter2 = neighbourCell.getParticlePointers().begin();
           molIter2 != neighbourCell.getParticlePointers().end();
           molIter2++ )
      {
         dd = (*molIter2)->dist2(*m1, distanceVector);
         if(dd > cutoffRadiusSquare) continue;
         u += this->_particlePairsHandler.processPair(*m1, **molIter2, distanceVector, 2, dd);
      }
   }
   return u;
}

int LinkedCells::grandcanonicalBalance(DomainDecompBase* comm)
{
   comm->collCommInit(1);
   comm->collCommAppendInt(this->_localInsertionsMinusDeletions);
   comm->collCommAllreduceSum();
   int universalInsertionsMinusDeletions = comm->collCommGetInt();
   comm->collCommFinalize();
   return universalInsertionsMinusDeletions;
}

void LinkedCells::grandcanonicalStep
(
   ChemicalPotential* mu, double T
)
{
   bool accept = true;
   double DeltaUpot;
   Molecule* m;

   this->_localInsertionsMinusDeletions = 0;

   mu->submitTemperature(T);
   double minco[3];
   double maxco[3];
   for(int d=0; d < 3; d++)
   {
      minco[d] = this->getBoundingBoxMin(d);
      maxco[d] = this->getBoundingBoxMax(d);
   }

   bool hasDeletion = true;
   bool hasInsertion = true;
   double ins[3];
   unsigned nextid = 0;
   while(hasDeletion || hasInsertion)
   {
      if(hasDeletion)
         hasDeletion = mu->getDeletion(this, minco, maxco);
      if(hasDeletion)
      {
         m = &(*(this->_particleIter));
         DeltaUpot = -1.0 * getEnergy(m);

         accept = mu->decideDeletion(DeltaUpot / T);
#ifndef NDEBUG
         if(accept) cout << "r" << mu->rank() << "d" << m->id() << "\n";
         else cout << "   (r" << mu->rank() << "-d" << m->id() << ")\n";
         cout.flush();
#endif
         if(accept)
         {
            m->upd_cache();
            // m->clearFM();
            m->setFM(0.0,0.0,0.0,0.0,0.0,0.0);
            mu->storeMolecule(*m);
            this->deleteMolecule(m->id(), m->r(0), m->r(1), m->r(2));
            this->_particles.erase(this->_particleIter);
            this->_particleIter = _particles.begin();
            this->_localInsertionsMinusDeletions--;
         }
      }

      if(hasInsertion)
      {
         nextid = mu->getInsertion(ins);
         hasInsertion = (nextid > 0);
      }
      if(hasInsertion)
      {
         // for(int d = 0; d < 3; d++)
         //    ins[d] = ins[d]-coords[d]*proc_domain_L[d]-m_rmin[d];
         Molecule tmp = mu->loadMolecule();
         for(int d=0; d < 3; d++) tmp.setr(d, ins[d]);
         tmp.setid(nextid);
         this->_particles.push_back(tmp);

         std::list<Molecule>::iterator mit = _particles.end();
         mit--;
         m = &(*mit);
         m->upd_cache();
         m->setFM(0.0,0.0,0.0,0.0,0.0,0.0);
         m->check(nextid); 
#ifndef NDEBUG
         cout << "rank " << mu->rank() << ": insert " << m->id() 
              << " at the reduced position (" << ins[0] << "/" << ins[1] << "/" << ins[2] << ")? ";
         cout.flush();
#endif

         unsigned long cellid = this->getCellIndexOfMolecule(m);
         this->_cells[cellid].addParticle(m);
         DeltaUpot = getEnergy(m);
         accept = mu->decideInsertion(DeltaUpot/T);

#ifndef NDEBUG
         if(accept) cout << "r" << mu->rank() << "i" << mit->id() << ")\n";
         else cout << "   (r" << mu->rank() << "-i" << mit->id() << ")\n";
         cout.flush();
#endif
         if(accept)
         {
            this->_localInsertionsMinusDeletions++;
         }
         else
         {
            // this->deleteMolecule(m->id(), m->r(0), m->r(1), m->r(2));
            this->_cells[cellid].deleteMolecule(m->id());

            mit->check(m->id());
            this->_particles.erase(mit);
         }
      }
   }
   for(m = this->begin(); m != this->end(); m = this->next())
   {
#ifndef NDEBUG
      m->check(m->id());
#endif
   }
}
