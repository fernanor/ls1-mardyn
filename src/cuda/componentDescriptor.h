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
		_componentMixXis( _module.getGlobal<floatType *>("componentMixXis") ),
		_componentMixEtas( _module.getGlobal<floatType *>("componentMixEtas") ),

		_debugComponentDescriptors( _module.getFunction("debugComponentDescriptors") ) {
	}

	virtual void upload() {
		floatType xis[MAX_NUM_COMPONENTS][MAX_NUM_COMPONENTS];
		floatType etas[MAX_NUM_COMPONENTS][MAX_NUM_COMPONENTS];
		std::vector<ComponentDescriptor> componentDescriptors;

		const std::vector<Component> &components = _domain.getComponents();

		for( int i = 0 ; i < components.size() ; i++ ) {
			const Component &component = components[i];
			ComponentDescriptor componentDescriptor;
			componentDescriptor.numLJCenters = component.numLJcenters();
			componentDescriptor.numCharges = component.numCharges();
			componentDescriptor.numDipoles = component.numDipoles();

			// TODO: use inheritance for relativePosition?
			for( int ljCenterIndex = 0 ; ljCenterIndex < componentDescriptor.numLJCenters ; ljCenterIndex++ ) {
				ComponentDescriptor::LJCenter &ljCenter = componentDescriptor.ljCenters[ljCenterIndex];

				const LJcenter &cLjCenter = component.ljcenter(ljCenterIndex);
				ljCenter.ljParameters.epsilon = cLjCenter.eps();
				ljCenter.ljParameters.sigma = cLjCenter.sigma();
				ljCenter.relativePosition = make_floatType3( cLjCenter.rx(), cLjCenter.ry(), cLjCenter.rz() );
			}

			for( int chargeIndex = 0 ; chargeIndex < componentDescriptor.numCharges ; chargeIndex++ ) {
				ComponentDescriptor::Charge &charge = componentDescriptor.charges[chargeIndex];

				const Charge &cCharge = component.charge(chargeIndex);
				charge.relativePosition = make_floatType3( cCharge.rx(), cCharge.ry(), cCharge.rz() );

				charge.q = cCharge.q();
			}

			for( int dipoleIndex = 0 ; dipoleIndex < componentDescriptor.numDipoles ; dipoleIndex++ ) {
				ComponentDescriptor::Dipole &dipole = componentDescriptor.dipoles[dipoleIndex];

				const Dipole &cDipole = component.dipole(dipoleIndex);
				dipole.relativePosition = make_floatType3( cDipole.rx(), cDipole.ry(), cDipole.rz() );
				dipole.relativeE = make_floatType3( cDipole.ex(), cDipole.ey(), cDipole.ez() );

				dipole.absMy = cDipole.absMy();
			}

			componentDescriptors.push_back(componentDescriptor);
		}

		// set mix coeff tables
		std::vector<double> &dmixcoeff = _domain.getmixcoeff();
		int index = 0;
		for( int i = 0 ; i < components.size() ; i++ ) {
			xis[i][i] = etas[i][i] = 1.0;
			for( int j = i + 1 ; j < components.size() ; j++ ) {
				floatType xi = dmixcoeff[index++];
				floatType eta = dmixcoeff[index++];

				xis[i][j] = xis[j][i] = xi;
				etas[i][j] = etas[j][i] = eta;
			}
		}

		_componentDescriptorBuffer.copyToDevice(componentDescriptors);
		_componentDescriptors.set( _componentDescriptorBuffer );

		_componentMixXiBuffer.copyToDevice( (floatType*) xis, MAX_NUM_COMPONENTS * MAX_NUM_COMPONENTS );
		_componentMixXis.set( _componentMixXiBuffer );

		_componentMixEtaBuffer.copyToDevice( (floatType*) etas, MAX_NUM_COMPONENTS * MAX_NUM_COMPONENTS );
		_componentMixEtas.set( _componentMixEtaBuffer );

#ifdef DEBUG_COMPONENT_DESCRIPTORS
		_debugComponentDescriptors.call().parameter( componentDescriptors.size() ).execute();
#endif
	}

protected:
	CUDA::Global<ComponentDescriptor *> _componentDescriptors;
	CUDA::Global<floatType *> _componentMixXis, _componentMixEtas;

	CUDA::DeviceBuffer<ComponentDescriptor> _componentDescriptorBuffer;
	CUDA::DeviceBuffer<floatType> _componentMixXiBuffer, _componentMixEtaBuffer;

	CUDA::Function _debugComponentDescriptors;
};


#endif /* COMPONENTDESCRIPTOR_H_ */
