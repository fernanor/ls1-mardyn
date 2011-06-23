/*
 * componentDescriptor.h
 *
 *  Created on: Jun 21, 2011
 *      Author: andreas
 */

#ifndef COMPONENTDESCRIPTOR_H_
#define COMPONENTDESCRIPTOR_H_

#include <vector>

#include "sharedDecls.h"

class ComponentDescriptor : public CUDAStaticDataComponent {
public:
	ComponentDescriptor( const CUDAComponent &component ) :
		CUDAForceCalculationComponent( component ), _componentDescriptors( module.getGlobal<ComponentDescriptor *>("componentDescriptors") ) {
	}

	virtual void upload() {
		std::vector<ComponentDescriptor> componentDescriptors;
		const std::vector<Component> &components = _domain.getComponents();

		for( int i = 0 ; i < components.size() ; i++ ) {
			ComponentDescriptor componentDescriptor;
			componentDescriptor.numLJCenters = components[i].numLJcenters();

			for( int ljCenterIndex = 0 ; ljCenterIndex < componentDescriptor.numLJCenters ; ljCenterIndex++ ) {
				//componentDescriptor.relativeLJCenter[ljCenterIndex] = components[i].ljcenter(i)._r
			}
		}
	}

protected:
	CUDA::Global<ComponentDescriptor> _componentDescriptors;
};


#endif /* COMPONENTDESCRIPTOR_H_ */
