#include "hoCuNDArray_utils.h"
#include "radial_utilities.h"
#include "hoNDArray_fileio.h"
#include "cuNDArray.h"
#include "imageOperator.h"
#include "identityOperator.h"
#include "hoPartialDerivativeOperator.h"
#include "hoCuConebeamProjectionOperator.h"
#include "cuConvolutionOperator.h"
#include "hoCuIdentityOperator.h"
#include "hoCuNDArray_math.h"
#include "hoCuNDArray_blas.h"
#include "hoCuCgSolver.h"
#include "CBCT_acquisition.h"
#include "complext.h"
#include "encodingOperatorContainer.h"
#include "vector_td_io.h"
#include "hoCuPartialDerivativeOperator.h"
#include "GPUTimer.h"

#include <iostream>
#include <algorithm>
#include <sstream>
#include <math_constants.h>
#include <boost/program_options.hpp>
#include <boost/make_shared.hpp>

using namespace std;
using namespace Gadgetron;

namespace po = boost::program_options;

int 
main(int argc, char** argv)
{
    string acquisition_filename;
    string outputFile;
    uintd3 imageSize;
    floatd3 voxelSize;
    float reg_weight;
    int device;
    unsigned int dump;
    unsigned int downsamples;
    unsigned int iterations;
    floatd2 mot_X;
    floatd2 mot_Y;
    floatd2 mot_Z;
    int ffs;


    po::options_description desc("Allowed options");
    desc.add_options()
            ("help", "produce help message")
            ("acquisition,a", po::value<string>(&acquisition_filename)->default_value("acquisition.hdf5"), "Acquisition data")
            ("samples,n",po::value<unsigned int>(),"Number of samples per ray")
            ("output,f", po::value<string>(&outputFile)->default_value("reconstruction.real"), "Output filename")
            ("size,s",po::value<uintd3>(&imageSize)->default_value(uintd3(512,512,1)),"Image size in pixels")
            ("binning,b",po::value<string>(),"Binning file for 4d reconstruction")
            ("SAG","Use exact SAG correction if present")
            ("voxelSize,v",po::value<floatd3>(&voxelSize)->default_value(floatd3(0.488f,0.488f,1.0f)),"Voxel size in mm")
            ("dimensions,d",po::value<floatd3>(),"Image dimensions in mm. Overwrites voxelSize.")
            ("iterations,i",po::value<unsigned int>(&iterations)->default_value(10),"Number of iterations")
            ("weight,w",po::value<float>(&reg_weight)->default_value(float(0.0f)),"Regularization weight")
            ("device",po::value<int>(&device)->default_value(0),"Number of the device to use (0 indexed)")
            ("downsample,D",po::value<unsigned int>(&downsamples)->default_value(0),"Downsample projections this factor")
            ("dump",po::value<unsigned int>(&dump)->default_value(0),"Dump image every N iterations")
            ("motion_X,X",po::value<floatd2>(&mot_X)->default_value(floatd2(0.0f,0.0f)),"Motion in X direction in mm")
            ("motion_Y,Y",po::value<floatd2>(&mot_Y)->default_value(floatd2(0.0f,0.0f)),"Motion in Y direction in mm")
            ("motion_Z,Z",po::value<floatd2>(&mot_Z)->default_value(floatd2(0.0f,0.0f)),"Motion in Z direction in mm")
            ("initX,x", po::value<string>(), "Initial Recon Guess")
            ("ffs",po::value<int>(&ffs)->default_value(0),"Use XY flying focal spot (0 = no, 1 = yes)")
            ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        cout << desc << "\n";
        return 1;
    }
    std::cout << "Command line options:" << std::endl;
    for (po::variables_map::iterator it = vm.begin(); it != vm.end(); ++it){
        boost::any a = it->second.value();
        std::cout << it->first << ": ";
        if (a.type() == typeid(std::string)) std::cout << it->second.as<std::string>();
        else if (a.type() == typeid(int)) std::cout << it->second.as<int>();
        else if (a.type() == typeid(unsigned int)) std::cout << it->second.as<unsigned int>();
        else if (a.type() == typeid(float)) std::cout << it->second.as<float>();
        else if (a.type() == typeid(vector_td<float,3>)) std::cout << it->second.as<vector_td<float,3> >();
        else if (a.type() == typeid(vector_td<int,3>)) std::cout << it->second.as<vector_td<int,3> >();
        else if (a.type() == typeid(vector_td<unsigned int,3>)) std::cout << it->second.as<vector_td<unsigned int,3> >();
        else std::cout << "Unknown type" << std::endl;
        std::cout << std::endl;
    }
    cudaSetDevice(device);
    cudaDeviceReset();

    //Really weird stuff. Needed to initialize the device?? Should find real bug.
    cudaDeviceManager::Instance()->lockHandle();
    cudaDeviceManager::Instance()->unlockHandle();

    boost::shared_ptr<CBCT_acquisition> ps(new CBCT_acquisition());
    ps->load(acquisition_filename);
    ps->get_geometry()->print(std::cout);
    ps->downsample(downsamples);

    float SDD = ps->get_geometry()->get_SDD();
    float SAD = ps->get_geometry()->get_SAD();

    boost::shared_ptr<CBCT_binning> binning(new CBCT_binning());
    if (vm.count("binning")){
        std::cout << "Loading binning data" << std::endl;
        binning->load(vm["binning"].as<string>());
    } else
        binning->set_as_default_3d_bin(ps->get_projections()->get_size(2));

    binning->print(std::cout);

    floatd3 imageDimensions;
    if (vm.count("dimensions")){
        imageDimensions = vm["dimensions"].as<floatd3>();
        voxelSize = imageDimensions/imageSize;
    }
    else imageDimensions = voxelSize*imageSize;

    float lengthOfRay_in_mm = norm(imageDimensions);
    unsigned int numSamplesPerPixel = 3;
    float minSpacing = min(voxelSize)/numSamplesPerPixel;

    unsigned int numSamplesPerRay;
    if (vm.count("samples")) numSamplesPerRay = vm["samples"].as<unsigned int>();
    else numSamplesPerRay = ceil( lengthOfRay_in_mm / minSpacing );

    float step_size_in_mm = lengthOfRay_in_mm / numSamplesPerRay;
    size_t numProjs = ps->get_projections()->get_size(2);
    size_t needed_bytes = 2 * prod(imageSize) * sizeof(float);

    std::vector<size_t> is_dims = to_std_vector((uint64d3)imageSize);
    std::cout << "IS dimensions " << is_dims[0] << " " << is_dims[1] << " " << is_dims[2] << std::endl;
    std::cout << "Image size " << imageDimensions << std::endl;

    // FC get use_cyl_det from data
    bool use_cyl_det =bool( ps->get_geometry()->get_DetType());
    std::cout << "Use_cyl_det " << use_cyl_det << std::endl;

    is_dims.push_back(binning->get_number_of_bins());

    hoCuNDArray<float> projections(*ps->get_projections());

    // Define encoding operator
    boost::shared_ptr< hoCuConebeamProjectionOperator >
            E( new hoCuConebeamProjectionOperator() );

    E->setup(ps,binning,imageDimensions);
    E->set_domain_dimensions(&is_dims);
    E->set_codomain_dimensions(ps->get_projections()->get_dimensions().get());
    E->set_use_cylindrical_detector(use_cyl_det);
    E->set_use_filtered_backprojection(bool(0));
    E->set_use_flying_focal_spot(bool(ffs));

    if (E->get_use_offset_correction())
        E->offset_correct(&projections);

    std::size_t found = outputFile.find('.');
    string dump_name = outputFile.substr(0,found);
    //  std::size_t length = outputFile.copy(dump_name,found);
    std::cout << "Output Name " << outputFile << std::endl;
    std::cout << "Dump Name " << dump_name << std::endl;

    // FC create vector of dX, dY, and dZ over the acquisitions
    float mot_X_extent = mot_X[1] - mot_X[0];
    float mot_Y_extent = mot_Y[1] - mot_Y[0];
    float mot_Z_extent = mot_Z[1] - mot_Z[0];

    std::vector<floatd3> mot_XYZ;
    float mot_X_val;
    float mot_Y_val;
    float mot_Z_val;
    floatd3 mot_XYZ_val;
    std::cout << "CBCT_reconstruct_CG: Motion Vector " << std::endl;
    if (bool(ffs))
    {
        // For FFS, we have half as many projection positions but we sample them twice
        for( unsigned int i=0; i<numProjs/2; i++ )
        {
            mot_X_val = mot_X[0] + mot_X_extent*i/(numProjs/2);
            mot_Y_val = mot_Y[0] + mot_Y_extent*i/(numProjs/2);
            mot_Z_val = mot_Z[0] + mot_Z_extent*i/(numProjs/2);
            std::cout << "i =  " << i << ", x: " << mot_X_val << ", y: " << mot_Y_val<< ", z: " << mot_Z_val << std::endl;
            mot_XYZ_val = floatd3(mot_X_val,mot_Y_val,mot_Z_val);
            // Push the values twice
            mot_XYZ.push_back(mot_XYZ_val);
            mot_XYZ.push_back(mot_XYZ_val);
        }
    }
    else
    {
        for( unsigned int i=0; i<numProjs; i++ )
        {
            mot_X_val = mot_X[0] + mot_X_extent*i/numProjs;
            mot_Y_val = mot_Y[0] + mot_Y_extent*i/numProjs;
            mot_Z_val = mot_Z[0] + mot_Z_extent*i/numProjs;
            std::cout << "i =  " << i << ", x: " << mot_X_val << ", y: " << mot_Y_val<< ", z: " << mot_Z_val << std::endl;
            mot_XYZ_val = floatd3(mot_X_val,mot_Y_val,mot_Z_val);

            mot_XYZ.push_back(mot_XYZ_val);
        }
    }

    E->set_motionXYZ_vector(mot_XYZ);

    // Define regularization operator
    boost::shared_ptr< hoCuIdentityOperator<float> >
            I( new hoCuIdentityOperator<float>() );

    I->set_weight(reg_weight);

    hoCuCgSolver<float> solver;

    solver.set_encoding_operator(E);

    if( reg_weight>0.0f ) {
        std::cout << "Adding identity operator with weight " << reg_weight << std::endl;
        solver.add_regularization_operator(I);
    }

    solver.set_max_iterations(iterations);
    solver.set_tc_tolerance(1e-8);
    solver.set_output_mode(hoCuCgSolver<float>::OUTPUT_VERBOSE);
    solver.set_dump_frequency(dump);
    solver.set_dump_name(dump_name);

    string initial_guess;
    boost::shared_ptr< hoNDArray<float>> init_guess_vol;
    boost::shared_ptr< hoCuNDArray<float>>  cu_init_guess_vol;
    if (vm.count("initX")){
        std::cout << "Loading initial reconstruction result" << std::endl;
        initial_guess = vm["initX"].as<string>();
        init_guess_vol = read_nd_array<float>( initial_guess.c_str() );

        //cu_init_guess_vol = boost::make_shared<hoCuNDArray<float>>(init_guess_vol);
        cu_init_guess_vol = boost::make_shared<hoCuNDArray<float>>(init_guess_vol.get());

        std::cout << "Adding initial recon to solver_x0" << std::endl;
        //solver.set_x0(cu_init_guess_vol);
    }

    /*  if (vm.count("TV")){
    boost::shared_ptr<hoCuPartialDerivativeOperator<float,4> > dx (new hoCuPartialDerivativeOperator<float,4>(0) );
    boost::shared_ptr<hoCuPartialDerivativeOperator<float,4> > dy (new hoCuPartialDerivativeOperator<float,4>(1) );
    boost::shared_ptr<hoCuPartialDerivativeOperator<float,4> > dz (new hoCuPartialDerivativeOperator<float,4>(2) );
    boost::shared_ptr<hoCuPartialDerivativeOperator<float,4> > dt (new hoCuPartialDerivativeOperator<float,4>(3) );

    dx->set_codomain_dimensions(&is_dims);
    dy->set_codomain_dimensions(&is_dims);
    dz->set_codomain_dimensions(&is_dims);
    dt->set_codomain_dimensions(&is_dims);

    dx->set_domain_dimensions(&is_dims);
    dy->set_domain_dimensions(&is_dims);
    dz->set_domain_dimensions(&is_dims);
    dt->set_domain_dimensions(&is_dims);

    dx->set_weight(vm["TV"].as<float>());
    dy->set_weight(vm["TV"].as<float>());
    dz->set_weight(vm["TV"].as<float>());
    dt->set_weight(vm["TV"].as<float>());

    solver.add_regularization_group_operator(dx);
    solver.add_regularization_group_operator(dy);
    solver.add_regularization_group_operator(dz);
    solver.add_regularization_group_operator(dt);
    solver.add_group(1);
    }*/

    // Run solver
    //

    boost::shared_ptr< hoCuNDArray<float> > result;

    {
        GPUTimer timer("\nRunning conjugate gradient solver");
        if (vm.count("initX"))
        {
            result = solver.solve_from_rhs(cu_init_guess_vol.get());
        }
        else
        {
            result = solver.solve(&projections);
        }
    }

    write_nd_array<float>( result.get(), outputFile.c_str());
}
