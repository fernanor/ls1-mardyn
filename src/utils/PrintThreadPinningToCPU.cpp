/*
 * PrintThreadPinningToCPU.cpp
 *
 *  Created on: 26 Sep 2017
 *      Author: tchipevn
 */

#include "PrintThreadPinningToCPU.h"
#include "../WrapOpenMP.h"
#include "utils/Logger.h"

using Log::global_log;
using std::endl;

#if !defined(__INTEL_COMPILER) and !defined(_SX)
#include <sched.h> /* int sched_getcpu(void); */

void PrintThreadPinningToCPU() {
	int mastercpu = sched_getcpu();

	global_log->info() << "	Thread pinning:" << std::endl;
	global_log->info() << "	Master thread running on " << mastercpu << std::endl;

	#if defined(_OPENMP)
	#pragma omp parallel
	#endif
	{
		int cpu;

		// protect the call to sched_getcpu, just in case. It is "MT-Safe" but let's be super safe.
		#if defined(_OPENMP)
		#pragma omp critical (sched_getcpu)
		#endif
		cpu = sched_getcpu();


		const int myID = mardyn_get_thread_num();
		const int numThreads = mardyn_get_num_threads();

		for (int i = 0; i < numThreads; ++i) {
			if (i == myID) {
				global_log->info() << "		Thread with id " << myID << " is running on " << cpu << "." << endl;
			}

			#if defined(_OPENMP)
			#pragma omp barrier
			#endif
		}
	}
}
#else
using Log::global_log;

void PrintThreadPinningToCPU() {
	global_log->warning() << "Thread pinning information cannot be obtained for this system/compiler by ls1. "
						  "Instead, please use OpenMP runtime system capabilities, e.g. KMP_AFFINITY=verbose for the Intel Compiler." << endl;
}

#endif
