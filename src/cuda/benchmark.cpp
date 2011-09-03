/*
 * benchmark.cpp
 *
 *  Created on: Aug 17, 2011
 *      Author: andreas
 */
#include <stdio.h>

#include "benchmark.h"

void SimulationStats::writeFrameStats( const std::string &frameFile ) {
	FILE *file;

	file = fopen( frameFile.c_str(), "wt" );

	fprintf( file, "potential, virial\n" );

	for( int i = 0 ; i < potentials.getCount() ; i++ ) {
		fprintf( file, "%.18e, %.18e\n", potentials[i], virials[i] );
	}

	fclose( file );
}

void SimulationStats::writeRunStats( const std::string &buildFile ) {
	// write header?
	FILE *file;
	if( (file = fopen( buildFile.c_str(), "rt" )) != NULL ) {
		fclose(file);
		file = fopen( buildFile.c_str(), "at" );
	}
	else {
		file = fopen( buildFile.c_str(), "wt" );
		fprintf( file, "timeSteps, moleculeCount, totalTime" );
#ifndef NO_CUDA
		fprintf( file, ", numWarps, numLocalStorageWarps, maxRegisterCount, CUDA_totalTime, CUDA_preTime, CUDA_postTime, CUDA_singleTime, CUDA_pairTime, CUDA_processingTime" );
#endif
		fprintf( file, ", name\n" );
	}


	fprintf( file, "%i, %i, %e", timeSteps, moleculeCount, (double) totalTime );
#ifndef NO_CUDA
	fprintf( file, ", %i, %i, %i, %e, %e, %e, %e, %e, %e",
			NUM_WARPS, NUM_LOCAL_STORAGE_WARPS, MAX_REGISTER_COUNT,
			(double) CUDA_frameTime, (double) CUDA_preTime, (double) CUDA_postTime, (double) CUDA_singleTime, (double) CUDA_pairTime, (double) CUDA_processingTime
			);
#endif
	fprintf( file, ", %s\n", name.c_str() );
}

SimulationStats simulationStats;
