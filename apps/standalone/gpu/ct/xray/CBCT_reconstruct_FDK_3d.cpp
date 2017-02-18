#include "parameterparser.h"
#include "CBCT_acquisition.h"
#include "CBCT_binning.h"
#include "hoCuConebeamProjectionOperator.h"
#include "hoNDArray_fileio.h"
#include "hoCuNDArray_math.h"
#include "vector_td_utilities.h"
#include "hoNDArray_utils.h"
#include "GPUTimer.h"

#include <iostream>
#include <algorithm>
#include <sstream>

using namespace Gadgetron;
using namespace std;

int main(int argc, char** argv) 
{ 
  // Parse command line
  //

  ParameterParser parms(1024);
  parms.add_parameter( 'd', COMMAND_LINE_STRING, 1, "Input acquisition filename (.hdf5)", true );
  parms.add_parameter( 'b', COMMAND_LINE_STRING, 1, "Binning filename (.hdf5)", false );
  parms.add_parameter( 'r', COMMAND_LINE_STRING, 1, "Output image filename (.real)", true, "reconstruction_FDK.real" );
  parms.add_parameter( 'm', COMMAND_LINE_INT, 3, "Matrix size (3d)", true, "256, 256, 144" );
  parms.add_parameter( 'f', COMMAND_LINE_FLOAT, 3, "FOV in mm (3d)", true, "448, 448, 252" );
  parms.add_parameter( 'F', COMMAND_LINE_INT, 1, "Use filtered backprojection (fbp)", true, "1" );
  parms.add_parameter( 'P', COMMAND_LINE_INT, 1, "Projections per batch", false );
  parms.add_parameter( 'D', COMMAND_LINE_INT, 1, "Number of downsamples of projection plate", true, "0" );
  parms.add_parameter( 'X', COMMAND_LINE_FLOAT, 2, "Motion in X direction in mm",true,"0,0");
  parms.add_parameter( 'Y', COMMAND_LINE_FLOAT, 2, "Motion in Y direction in mm",true,"0,0");
  parms.add_parameter( 'Z', COMMAND_LINE_FLOAT, 2, "Motion in Z direction in mm",true,"0,0");
  parms.add_parameter( 'S', COMMAND_LINE_INT, 1, "Use XY flying focal spot (0 = no, 1 = yes)",true,"0");

  parms.parse_parameter_list(argc, argv);
  if( parms.all_required_parameters_set() ) {
    parms.print_parameter_list();
  }
  else{
    parms.print_parameter_list();
    parms.print_usage();
    return 1;
  }
  
  std::string acquisition_filename = (char*)parms.get_parameter('d')->get_string_value();
  std::string image_filename = (char*)parms.get_parameter('r')->get_string_value();

  // Load acquisition data
  //

  boost::shared_ptr<CBCT_acquisition> acquisition( new CBCT_acquisition() );

  {
    GPUTimer timer("Loading projections");
    acquisition->load(acquisition_filename);
  }

	// Downsample projections if requested
	//

	{
		GPUTimer timer("Downsampling projections");
		unsigned int num_downsamples = parms.get_parameter('D')->get_int_value();    
		acquisition->downsample(num_downsamples);
	}
  
  // Load or generate binning data
  //
  
  boost::shared_ptr<CBCT_binning> binning( new CBCT_binning() );

  if (parms.get_parameter('b')->get_is_set()){
    std::string binningdata_filename = (char*)parms.get_parameter('b')->get_string_value();
    binning->load(binningdata_filename);
    binning = boost::shared_ptr<CBCT_binning>(new CBCT_binning(binning->get_3d_binning()));
    binning->print();
  } 
  else 
    binning->set_as_default_3d_bin(acquisition->get_projections()->get_size(2));

  // Configuring...
  //

  uintd2 ps_dims_in_pixels( acquisition->get_projections()->get_size(0),
			    acquisition->get_projections()->get_size(1) );
  
  floatd2 ps_dims_in_mm( acquisition->get_geometry()->get_FOV()[0],
			 acquisition->get_geometry()->get_FOV()[1] );

  float SDD = acquisition->get_geometry()->get_SDD();
  float SAD = acquisition->get_geometry()->get_SAD();

  uintd3 is_dims_in_pixels( parms.get_parameter('m')->get_int_value(0),
			    parms.get_parameter('m')->get_int_value(1),
			    parms.get_parameter('m')->get_int_value(2) );
  
  floatd3 is_dims_in_mm( parms.get_parameter('f')->get_float_value(0), 
			 parms.get_parameter('f')->get_float_value(1), 
			 parms.get_parameter('f')->get_float_value(2) );
  
  bool use_fbp = parms.get_parameter('F')->get_int_value();
  bool use_cyl_det =bool( acquisition->get_geometry()->get_DetType());
  std::cout << "FDK_3d.cpp Detector Type Boolean: " << use_cyl_det << std::endl;

  // Get motion values
  floatd2 mot_X( parms.get_parameter('X')->get_float_value(0),
                parms.get_parameter('X')->get_float_value(1));
  floatd2 mot_Y( parms.get_parameter('Y')->get_float_value(0),
                parms.get_parameter('Y')->get_float_value(1));
  floatd2 mot_Z( parms.get_parameter('Z')->get_float_value(0),
                parms.get_parameter('Z')->get_float_value(1));

  // Get FFS value
  int ffs = parms.get_parameter('S')->get_int_value();

  // Allocate array to hold the result
  //
  
  std::vector<size_t> is_dims;
  is_dims.push_back(is_dims_in_pixels[0]);
  is_dims.push_back(is_dims_in_pixels[1]);
  is_dims.push_back(is_dims_in_pixels[2]);
  
  hoCuNDArray<float> fdk_3d(&is_dims);
  hoCuNDArray<float> projections(*acquisition->get_projections());  

  // Define conebeam projection operator
  // - and configure based on input parameters
  //
  
  boost::shared_ptr< hoCuConebeamProjectionOperator > E( new hoCuConebeamProjectionOperator() );

  E->setup( acquisition, binning, is_dims_in_mm );
  E->set_use_filtered_backprojection(use_fbp);
  E->set_use_cylindrical_detector(use_cyl_det);
  E->set_use_flying_focal_spot(bool(ffs));

  CommandLineParameter *parm = parms.get_parameter('P');
  if( parm && parm->get_is_set() )
    E->set_num_projections_per_batch( parm->get_int_value() );
  
  //FC add motion vector
  // FC create vector of dX, dY, and dZ over the acquisitions
   float mot_X_extent = mot_X[1] - mot_X[0];
   float mot_Y_extent = mot_Y[1] - mot_Y[0];
   float mot_Z_extent = mot_Z[1] - mot_Z[0];

   std::vector<floatd3> mot_XYZ;
   float mot_X_val;
   float mot_Y_val;
   float mot_Z_val;
   floatd3 mot_XYZ_val;
   size_t numProjs = acquisition->get_projections()->get_size(2);

   if (bool(ffs))
   {
       // For FFS, we have half as many projection positions but we sample them twice
       for( unsigned int i=0; i<numProjs/2; i++ )
       {
           mot_X_val = mot_X[0] + mot_X_extent*i/(numProjs/2);
           mot_Y_val = mot_Y[0] + mot_Y_extent*i/(numProjs/2);
           mot_Z_val = mot_Z[0] + mot_Z_extent*i/(numProjs/2);
           //std::cout << "i =  " << i << ", x: " << mot_X_val << ", y: " << mot_Y_val<< ", z: " << mot_Z_val << std::endl;
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
           //std::cout << "i =  " << i << ", x: " << mot_X_val << ", y: " << mot_Y_val<< ", z: " << mot_Z_val << std::endl;
           mot_XYZ_val = floatd3(mot_X_val,mot_Y_val,mot_Z_val);

           mot_XYZ.push_back(mot_XYZ_val);
       }
   }
   E->set_motionXYZ_vector(mot_XYZ);


  // Initialize the device
  // - just to report more accurate timings
  //

  cudaThreadSynchronize();

  //
  // Standard 3D FDK reconstruction
  //

  {
    GPUTimer timer("Running 3D FDK reconstruction");
    E->mult_MH( &projections, &fdk_3d );
    cudaThreadSynchronize();
  }

  write_nd_array<float>( &fdk_3d, image_filename.c_str() );
  return 0;
}
