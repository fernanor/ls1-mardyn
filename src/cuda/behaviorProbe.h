#ifndef DEBUGBEHAVIORPROBE_H__
#define DEBUGBEHAVIORPROBE_H__

#include <stdio.h>

struct BehaviorProbe {
	enum Type {
		BPT_CLEARED,
		BPT_USED,
		BPT_SET,
		BPT_NONE,
		BPT_COUNT
	};
	static int counter;
	static Type lastType;
	static const char *msg[BPT_COUNT];

	static void message(Type type) {
		if( lastType == type ) {
			counter++;
		}
		else {
			if( counter > 1 )
				printf( "!!BP!! done with %s - %i events\n", msg[lastType], counter );

			lastType = type;
			counter = 1;
			printf( "!!BP!! %s\n", msg[type] );
		}
	}

	static void FMcleared() {
		message( BPT_CLEARED );
	}

	static void FMused() {
		message( BPT_USED );
	}

	static void FMset() {
		message( BPT_SET );
	}
};

#endif
