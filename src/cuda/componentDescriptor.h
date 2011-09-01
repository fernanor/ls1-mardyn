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

		_componentDescriptors( _module.getGlobal<ComponentDescriptors>("componentDescriptors") ),
		_componentMixXis( _module.getGlobal<ComponentMixCoefficients>("componentMixXis") ),
		_componentMixEtas( _module.getGlobal<ComponentMixCoefficients>("componentMixEtas") ),

		_debugComponentDescriptors( _module.getFunction("debugComponentDescriptors") ) {
	}

	virtual void upload() {
		ComponentMixCoefficients xis;
		ComponentMixCoefficients etas;
		ComponentDescriptors componentDescriptors;

		const std::vector<Component> &components = _domain.getComponents();

		for( int i = 0 ; i < components.size() ; i++ ) {
			const Component &component = components[i];
			ComponentDescriptor &componentDescriptor = componentDescriptors[i];

			componentDescriptor.numLJCenters = component.numLJcenters();
			componentDescriptor.numCharges = component.numCharges();
			componentDescriptor.numDipoles = component.numDipoles();

			// TODO: use inheritance for relativePosition?
#if MAX_NUM_LJCENTERS > 0
			for( int ljCenterIndex = 0 ; ljCenterIndex < componentDescriptor.numLJCenters ; ljCenterIndex++ ) {
				ComponentDescriptor::LJCenter &ljCenter = componentDescriptor.ljCenters[ljCenterIndex];

				const LJcenter &cLjCenter = component.ljcenter(ljCenterIndex);
				ljCenter.ljParameters.epsilon = cLjCenter.eps();
				ljCenter.ljParameters.sigma = cLjCenter.sigma();
				ljCenter.relativePosition = make_floatType3( cLjCenter.rx(), cLjCenter.ry(), cLjCenter.rz() );
			}
#endif

#if MAX_NUM_CHARGES > 0
			for( int chargeIndex = 0 ; chargeIndex < componentDescriptor.numCharges ; chargeIndex++ ) {
				ComponentDescriptor::Charge &charge = componentDescriptor.charges[chargeIndex];

				const Charge &cCharge = component.charge(chargeIndex);
				charge.relativePosition = make_floatType3( cCharge.rx(), cCharge.ry(), cCharge.rz() );

				charge.q = cCharge.q();
			}
#endif

#if MAX_NUM_DIPOLES > 0
			for( int dipoleIndex = 0 ; dipoleIndex < componentDescriptor.numDipoles ; dipoleIndex++ ) {
				ComponentDescriptor::Dipole &dipole = componentDescriptor.dipoles[dipoleIndex];

				const Dipole &cDipole = component.dipole(dipoleIndex);
				dipole.relativePosition = make_floatType3( cDipole.rx(), cDipole.ry(), cDipole.rz() );
				dipole.relativeE = make_floatType3( cDipole.ex(), cDipole.ey(), cDipole.ez() );

				dipole.absMy = cDipole.absMy();
			}
#endif
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

		_componentDescriptors.set( componentDescriptors );

		_componentMixXis.set( xis );

		_componentMixEtas.set( etas );

#ifdef DEBUG_COMPONENT_DESCRIPTORS
		_debugComponentDescriptors.call().parameter( componentDescriptors.size() ).execute();
#endif
	}

protected:
	CUDA::Global<ComponentDescriptors> _componentDescriptors;
	CUDA::Global<ComponentMixCoefficients> _componentMixXis, _componentMixEtas;

	CUDA::Function _debugComponentDescriptors;
};


#endif /* COMPONENTDESCRIPTOR_H_ */
