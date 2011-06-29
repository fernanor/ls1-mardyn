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
		CUDAStaticDataComponent( component ),

		_componentDescriptors( _module.getGlobal<ComponentDescriptor *>("componentDescriptors") ),

		_debugComponentDescriptors( _module.getFunction("debugComponentDescriptors") ) {
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

				const LJcenter &cLjCenter = component.ljcenter(ljCenterIndex);
				ljCenter.ljParameters.epsilon = cLjCenter.eps();
				ljCenter.ljParameters.sigma = cLjCenter.sigma();
				ljCenter.relativePosition = make_floatType3( cLjCenter.rx(), cLjCenter.ry(), cLjCenter.rz() );
			}

			componentDescriptors.push_back(componentDescriptor);
		}

		_componentDescriptorBuffer.copyToDevice(componentDescriptors);
		_componentDescriptors.set( _componentDescriptorBuffer );

#ifdef DEBUG_COMPONENT_DESCRIPTORS
		_debugComponentDescriptors.call().parameter( componentDescriptors.size() ).execute();
#endif
	}

protected:
	CUDA::Global<ComponentDescriptor *> _componentDescriptors;

	CUDA::DeviceBuffer<ComponentDescriptor> _componentDescriptorBuffer;

	CUDA::Function _debugComponentDescriptors;
};


#endif /* COMPONENTDESCRIPTOR_H_ */
