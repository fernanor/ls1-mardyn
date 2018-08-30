//
// Created by Kruegener on 8/27/2018.
// Ported from Domain.cpp by Mheier.
//

#include "Velocity3dProfile.h"

void Velocity3dProfile::output(string prefix) {
    global_log->info() << "[Velocity3dProfile] output" << std::endl;
    _profilePrefix = prefix;
    _profilePrefix += "_kartesian.V3Dpr";

    // Need to get DensityProfile to calculate average velocities per bin independent of number density.
    std::map<unsigned, long double> dens = _kartProf->_densProfile->getProfile();

    ofstream outfile(_profilePrefix.c_str());
    outfile.precision(6);

    outfile << "//Segment volume: " << _kartProf->segmentVolume << "\n//Accumulated data sets: " << _kartProf->accumulatedDatasets << "\n//Local profile of X-Y-Z components of velocity. Output file generated by the \"Velocity3dProfile\" method, plugins/profiles. \n";
    outfile << "//local velocity component profile: Y - Z || X-projection\n";
    // TODO: more info
    outfile << "// \t dX \t dY \t dZ \n";
    outfile << "\t" << 1/_kartProf->universalInvProfileUnit[0] << "\t" << 1/_kartProf->universalInvProfileUnit[1] << "\t" << 1/_kartProf->universalInvProfileUnit[2]<< "\n";
    outfile << "0 \t";

    for(unsigned z = 0; z < _kartProf->universalProfileUnit[2]; z++){
        outfile << (z+0.5) / _kartProf->universalInvProfileUnit[2] <<"  \t\t\t"; // Eintragen der z Koordinaten in Header
    }
    outfile << "\n";
    long double vd;
    // Y - axis label
    for(unsigned y = 0; y < _kartProf->universalProfileUnit[1]; y++){
        double hval = (y + 0.5) / _kartProf->universalInvProfileUnit[1];
        outfile << hval << "\t";
        // number density values
        for(unsigned z = 0; z < _kartProf->universalProfileUnit[2]; z++){
            for(unsigned x = 0; x < _kartProf->universalProfileUnit[0]; x++){
                long unID = (long) (x * _kartProf->universalProfileUnit[0] * _kartProf->universalProfileUnit[2] + y * _kartProf->universalProfileUnit[1] + z);
                // X - Y - Z output
                for(unsigned d = 0; d < 3; d++){
                    // Check for division by 0
                    if(dens[unID] != 0){
                        vd = _global3dProfile[d][unID] / dens[unID];
                    }
                    else{
                        vd = 0;
                    }
                    outfile << vd << "\t";
                }
            }
        }
        outfile << "\n";
    }
    outfile.close();

}
