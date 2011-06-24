/*
 * componentDescriptor.h
 *
 *  Created on: Jun 21, 2011
 *      Author: andreas
 */

#ifndef COMPONENTDESCRIPTOR_H_
#define COMPONENTDESCRIPTOR_H_

#include <vector>

#include "cutil_math.h"

#include "sharedDecls.h"

class ComponentDescriptorStorage : public CUDAStaticDataComponent {
public:
	ComponentDescriptorStorage( const CUDAComponent &component ) :
		CUDAStaticDataComponent( component ), _componentDescriptors( _module.getGlobal<ComponentDescriptor *>("componentDescriptors") ) {
	}

	virtual void upload() {
		std::vector<ComponentDescriptor> componentDescriptors;
		const std::vector<Component> &components = _domain.getComponents();

		for( int i = 0 ; i < components.size() ; i++ ) {
			const Component &component = components[i];
			ComponentDescriptor componentDescriptor;
			componentDescriptor.numLJCenters = component.numLJcenters();

			for( int ljCenterIndex = 0 ; ljCenterIndex < componentDescriptor.numLJCenters ; ljCenterIndex++ ) {
				ComponentDescriptor::LJCenter &ljCenter = componentDescriptor.ljCenters[ljCenterIndex];

				const LJcenter &cLjCenter = component.ljcenter(i);
				ljCenter.ljParameters.epsilon = cLjCenter.eps();
				ljCenter.ljParameters.sigma = cLjCenter.sigma();
				ljCenter.relativePosition = make_float3( cLjCenter.rx(), cLjCenter.ry(), cLjCenter.rz() );
			}

			componentDescriptors.push_back(componentDescriptor);
		}

		_componentDescriptorBuffer.copyToDevice(componentDescriptors);
		_componentDescriptors.set( _componentDescriptorBuffer );
	}

protected:
	CUDA::Global<ComponentDescriptor *> _componentDescriptors;

	CUDA::DeviceBuffer<ComponentDescriptor> _componentDescriptorBuffer;
};


#endif /* COMPONENTDESCRIPTOR_H_ */
