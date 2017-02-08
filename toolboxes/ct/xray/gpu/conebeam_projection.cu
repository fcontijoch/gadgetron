//
// This code performs 3D cone beam CT forwards and backwards projection
//

#include "conebeam_projection.h"
#include "float3x3.h"
#include "hoCuNDArray_math.h"
#include "vector_td.h"
#include "cuNDArray_elemwise.h"
#include "cuNDArray_operators.h"
#include "cuNDArray_utils.h"
#include "cuNFFT.h"
#include "check_CUDA.h"
#include "GPUTimer.h"
#include "cudaDeviceManager.h"
#include "hoNDArray_fileio.h"
#include "setup_grid.h"

#include <cuda_runtime_api.h>
#include <math_constants.h>
#include <cufft.h>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>

#include "cuPrintf.cu"
#include "cuPrintf.cuh"

#define PS_ORIGIN_CENTERING
#define IS_ORIGIN_CENTERING
//#define FLIP_Z_AXIS

// Read the projection/image data respectively as a texture (for input)
// - taking advantage of the cache and hardware interpolation
//

#define NORMALIZED_TC 1

static texture<float, 3, cudaReadModeElementType> 
image_tex( NORMALIZED_TC, cudaFilterModeLinear, cudaAddressModeBorder );

static texture<float, cudaTextureType2DLayered, cudaReadModeElementType> 
projections_tex( NORMALIZED_TC, cudaFilterModeLinear, cudaAddressModeBorder );

namespace Gadgetron 
{

// Utility to convert from degrees to radians
//

static inline __host__ __device__
float degrees2radians(float degree) {
    return degree * (CUDART_PI_F/180.0f);
}

// Utilities for filtering in frequency space
//

static boost::shared_ptr< cuNDArray<float_complext> > cb_fft( cuNDArray<float> *data )
{
    if( data == 0x0 )
        throw std::runtime_error("CB FFT : illegal input pointer provided");

    std::vector<size_t> in_dims = *data->get_dimensions();
    std::vector<size_t> out_dims;
    out_dims.push_back((in_dims[0]>>1)+1);
    out_dims.push_back(in_dims[1]);
    out_dims.push_back(in_dims[2]);

    boost::shared_ptr< cuNDArray<float_complext> > result( new cuNDArray<float_complext>(&out_dims) );
    cufftHandle plan;

    if( cufftPlanMany( &plan, 1, (int*)(&in_dims[0]), 0x0, 1, in_dims[0], 0x0, 1, out_dims[0], CUFFT_R2C, in_dims[1]*in_dims[2] ) != CUFFT_SUCCESS) {
        throw std::runtime_error("CB FFT plan failed");
    }

    if( cufftExecR2C( plan, data->get_data_ptr(), (cuFloatComplex*) result->get_data_ptr() ) != CUFFT_SUCCESS ) {
        throw std::runtime_error("CB FFT execute failed");;
    }

    if( cufftDestroy(plan) != CUFFT_SUCCESS) {
        throw std::runtime_error("CB FFT failed to destroy plan");
    }

    return result;
}

static void cb_ifft( cuNDArray<float_complext> *in_data, cuNDArray<float> *out_data )
{
    if( in_data == 0x0 || out_data == 0x0 )
        throw std::runtime_error("CB FFT : illegal input or output pointer provided");

    std::vector<size_t> in_dims = *in_data->get_dimensions();
    std::vector<size_t> out_dims = *out_data->get_dimensions();

    cufftHandle plan;

    if( cufftPlanMany( &plan, 1, (int*)(&out_dims[0]), 0x0, 1, in_dims[0], 0x0, 1, out_dims[0], CUFFT_C2R, in_dims[1]*in_dims[2] ) != CUFFT_SUCCESS) {
        throw std::runtime_error("CB iFFT plan failed");
    }

    if( cufftExecC2R( plan, (cuFloatComplex*) in_data->get_data_ptr(), out_data->get_data_ptr() ) != CUFFT_SUCCESS ) {
        throw std::runtime_error("CB iFFT execute failed");;
    }

    if( cufftDestroy(plan) != CUFFT_SUCCESS) {
        throw std::runtime_error("CB iFFT failed to destroy plan");
    }

    *out_data /= float(out_dims[0]);
}

//
// Redundancy correction for short scan mode
// - i.e. for less than a full rotation of data
//
// See "Optimal short scan convolution reconstruction for fanbeam CT", Dennis Parker, Med. Phys. 9(2) 1982
// and (for the implementation) "Parker weights revisited", Wesarg et al, Med. Phys. 29(3) 2002.
//

static __device__ const float epsilon = 0.001f;

static __inline__ __device__ float S( float beta )
{
    if( beta <= -0.5f ) return 0.0f;
    else if( beta > -0.5f && beta < 0.5f ) return 0.5f*(1.0f+sinf(CUDART_PI_F*beta));
    else /*if( beta >= 0.5f )*/ return 1.0f;
}

static __inline__ __device__ float B( float alpha, float delta )
{
    return 2.0f*(delta-alpha)+epsilon;
}

static __inline__ __device__ float b( float alpha, float delta )
{
    const float q = 0.1f; // with q=1 this formulae reduce to conventional Parker weights
    return q*B(alpha, delta);
}

__global__ void
redundancy_correct_kernel( float *projections,
                           const float * __restrict__ angles,
                           uintd3 dims, // Dimensions of the projections array
                           float delta  // The half-fan angle
                           )
{
    const unsigned int idx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned int num_elements = prod(dims);

    if( idx < num_elements ){

        const float in = projections[idx];
        const uintd3 co = idx_to_co<3>( idx, dims );
        const float tan_delta = tanf(delta);
        const float alpha = -atanf((float(co[0])/float(dims[0])-0.5f)*2.0f*tan_delta);
        const float beta = degrees2radians(angles[co[2]]);

        float omega = 0.5f*(S(beta/b(alpha, delta)-0.5f)+
                            S((beta+2.0f*(alpha-delta)-epsilon)/b(alpha, delta)+0.5f)-
                            S((beta-CUDART_PI_F+2.0f*alpha)/b(-alpha, delta)-0.5f)-
                            S((beta-CUDART_PI_F-2.0f*delta-epsilon)/b(-alpha, delta)+0.5f));

        projections[idx] = in*omega;
    }
}

void
redundancy_correct( cuNDArray<float> *projections,
                    float *angles_DevPtr,
                    float delta // The half-fan angle in radians
                    )
{
    //
    // Validate the input
    //

    if( projections == 0x0 ){
        throw std::runtime_error("Error: redundancy_correct: illegal array pointer provided");
    }

    if( projections->get_number_of_dimensions() != 3 ){
        throw std::runtime_error("Error: redundancy_correct: projections array must be three-dimensional");
    }

    const size_t projection_res_x = projections->get_size(0);
    const size_t projection_res_y = projections->get_size(1);
    const size_t num_projections = projections->get_size(2);
    uintd3 dims(projection_res_x, projection_res_y, num_projections);

    // Launch kernel
    //

    dim3 dimBlock, dimGrid;
    setup_grid( prod(dims), &dimBlock, &dimGrid );

    redundancy_correct_kernel<<< dimGrid, dimBlock >>>( projections->get_data_ptr(), angles_DevPtr, dims, delta );
    CHECK_FOR_CUDA_ERROR();
}


/***
 * Redundancy (or offset) correction from Wang. Med. Phys 2002, doi: 10.1118/1.1489043
 */
__global__ static void
offset_correct_kernel( float *projections,
                       const floatd2 * __restrict__ offsets,
                       uintd3 dims, // Dimensions of the projections array
                       floatd2 phys_dims, // Physical dimensions in mm
                       float SAD, // Source origin distance
                       float SDD // Source detector distance
                       )
{
    const unsigned int idx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned int num_elements = prod(dims);

    if( idx < num_elements ){

        const uintd3 co = idx_to_co<3>( idx, dims );
        const floatd2 offset = offsets[co[2]];
        const float t = phys_dims[0]*(float(co[0])/(float(dims[0]))-0.5f)+offset[0];
        const float omega = phys_dims[0]/2.0f-fabs(offset[0]);
        //const float omega = phys_dims[0]*float(co[0])/(2.0f*float(dims[0]));

        if( fabs(t) <= fabs(omega) ){
            //float w = 0.5*sinf(CUDART_PI_F*atanf(t/SDD)/(2.0f*atanf(omega/SDD)))+0.5;
            float sqrt_w = sinf(CUDART_PI_F*(t+omega)/(4.0f*omega));
            float w = sqrt_w*sqrt_w;
            projections[idx] *= w;
        }
    }
}

static void
offset_correct( cuNDArray<float> *projections,
                floatd2* offsets, // Ptr to cuda array
                floatd2 phys_dims,
                float SAD, // Source origin distance
                float SDD // Source detector distance
                )
{
    //
    // Validate the input
    //

    if( projections == 0x0 ){
        throw std::runtime_error("Error: offset_correct: illegal array pointer provided");
    }

    if( projections->get_number_of_dimensions() != 3 ){
        throw std::runtime_error("Error: offset_correct: projections array must be three-dimensional");
    }

    const size_t projection_res_x = projections->get_size(0);
    const size_t projection_res_y = projections->get_size(1);
    const size_t num_projections = projections->get_size(2);
    uintd3 dims(projection_res_x, projection_res_y, num_projections);

    // Launch kernel
    //

    dim3 dimBlock, dimGrid;
    setup_grid( prod(dims), &dimBlock, &dimGrid );

    offset_correct_kernel<<< dimGrid, dimBlock >>>( projections->get_data_ptr(), offsets, dims, phys_dims, SAD, SDD );
    CHECK_FOR_CUDA_ERROR();
}


/***
 * Redundancy (or offset) correction from Wang. Med. Phys 2002, doi: 10.1118/1.1489043
 */
__global__ static void
offset_correct_kernel_sqrt( float *projections,
                            const floatd2 * __restrict__ offsets,
                            uintd3 dims, // Dimensions of the projections array
                            floatd2 phys_dims, // Physical dimensions in mm
                            float SAD, // Source origin distance
                            float SDD // Source detector distance
                            )
{
    const unsigned int idx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned int num_elements = prod(dims);

    if( idx < num_elements ){

        const uintd3 co = idx_to_co<3>( idx, dims );
        const floatd2 offset = offsets[co[2]];
        const float t = phys_dims[0]*(float(co[0])/(float(dims[0]))-0.5f)+offset[0];
        const float omega = phys_dims[0]/2.0f-fabs(offset[0]);
        //const float omega = phys_dims[0]*float(co[0])/(2.0f*float(dims[0]));

        if( fabs(t) <= fabs(omega) ){
            //float w = 0.5*sinf(CUDART_PI_F*atanf(t/SDD)/(2.0f*atanf(omega/SDD)))+0.5;
            float sqrt_w = sinf(CUDART_PI_F*(t+omega)/(4.0f*omega));
            projections[idx] *= sqrt_w;
        }
    }
}

static void
offset_correct_sqrt( cuNDArray<float> *projections,
                     floatd2* offsets, // Ptr to cuda array
                     floatd2 phys_dims,
                     float SAD, // Source origin distance
                     float SDD // Source detector distance
                     )
{
    //
    // Validate the input
    //

    if( projections == 0x0 ){
        throw std::runtime_error("Error: offset_correct: illegal array pointer provided");
    }

    if( projections->get_number_of_dimensions() != 3 ){
        throw std::runtime_error("Error: offset_correct: projections array must be three-dimensional");
    }

    const size_t projection_res_x = projections->get_size(0);
    const size_t projection_res_y = projections->get_size(1);
    const size_t num_projections = projections->get_size(2);
    uintd3 dims(projection_res_x, projection_res_y, num_projections);

    // Launch kernel
    //

    dim3 dimBlock, dimGrid;
    setup_grid( prod(dims), &dimBlock, &dimGrid );

    offset_correct_kernel_sqrt<<< dimGrid, dimBlock >>>( projections->get_data_ptr(), offsets, dims, phys_dims, SAD, SDD );
    CHECK_FOR_CUDA_ERROR();
}


void apply_offset_correct(hoCuNDArray<float>* projections,std::vector<floatd2>& offsets,		floatd2 ps_dims_in_mm, float SDD,	float SAD){

    std::vector<size_t> dims = *projections->get_dimensions();
    size_t projection_size = dims[0]*dims[1];


    thrust::device_vector<floatd2> offsets_devVec(offsets);
    //Calculate number of projections we can fit on device, rounded to nearest MB
    size_t batch_size = (1024)*(cudaDeviceManager::Instance()->getFreeMemory()/(1024*projection_size*sizeof(float)));
    size_t remaining = dims[2];

    for (unsigned int i = 0; i < dims[2]/(batch_size+1)+1; i++){
        std::vector<size_t> projection_dims = dims;
        projection_dims[2] = std::min(remaining,batch_size);
        //Make a view of the batch of projections
        hoCuNDArray<float> projections_view(projection_dims,projections->get_data_ptr()+batch_size*i);
        cuNDArray<float> cu_projections(projections_view); //Copy to device
        floatd2* cu_offsets = thrust::raw_pointer_cast(&offsets_devVec[i*batch_size]);
        offset_correct_sqrt(&cu_projections,cu_offsets,ps_dims_in_mm,SAD,SDD);

        cudaMemcpy(projections_view.get_data_ptr(),cu_projections.get_data_ptr(),cu_projections.get_number_of_bytes(),cudaMemcpyDeviceToHost);
        remaining -= batch_size;
    }
}

//
// Forwards projection
//

__global__ void
conebeam_forwards_projection_kernel( float * __restrict__ projections,
                                     float * __restrict__ angles,
                                     floatd2 *offsets,
                                     floatd3 is_dims_in_pixels,
                                     floatd3 is_dims_in_mm,
                                     intd2 ps_dims_in_pixels_int,
                                     floatd2 ps_dims_in_mm,
                                     int num_projections,
                                     float SDD,
                                     float SAD,
                                     int num_samples_per_ray )
{
    const int idx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x+threadIdx.x;
    const int num_elements = prod(ps_dims_in_pixels_int)*num_projections;

    if( idx < num_elements){

        const intd3 co = idx_to_co<3>( idx, intd3(ps_dims_in_pixels_int[0], ps_dims_in_pixels_int[1], num_projections) );

        // Projection space dimensions and spacing
        //

        const floatd2 ps_dims_in_pixels = floatd2(ps_dims_in_pixels_int[0], ps_dims_in_pixels_int[1]);
        const floatd2 ps_spacing = ps_dims_in_mm / ps_dims_in_pixels;

        // Determine projection angle and rotation matrix
        //

        const float angle = angles[co[2]];
        const float3x3 rotation = calcRotationMatrixAroundZ(degrees2radians(angle));

        // Find start and end point for the line integral (image space)
        //

        floatd3 startPoint = floatd3(0.0f, -SAD, 0.0f);
        startPoint = mul(rotation, startPoint);

        // Projection plate indices
        //

#ifdef PS_ORIGIN_CENTERING
        const floatd2 ps_pc = floatd2(co[0], co[1]) + floatd2(0.5);
#else
        const floatd2 ps_pc = floatd2(co[0], co[1]);
#endif

        // Convert the projection plate coordinates into image space,
        // - local to the plate in metric units
        // - including half-fan and sag correction
        //

        const floatd2 proj_coords = (ps_pc / ps_dims_in_pixels - 0.5f) * ps_dims_in_mm + offsets[co[2]];

        // Define the end point for the line integrals
        //

        const float ADD = SDD - SAD; // in mm.
        floatd3 endPoint = floatd3(proj_coords[0], ADD, proj_coords[1]);
        endPoint = mul(rotation, endPoint);

        // Find direction vector of the line integral
        //

        floatd3 dir = endPoint-startPoint;

        // Perform integration only inside the bounding cylinder of the image volume
        //

        const floatd3 vec_over_dir = (is_dims_in_mm-startPoint)/dir;
        const floatd3 vecdiff_over_dir = (-is_dims_in_mm-startPoint)/dir;
        const floatd3 start = amin(vecdiff_over_dir, vec_over_dir);
        const floatd3 end   = amax(vecdiff_over_dir, vec_over_dir);

        float a1 = fmax(max(start),0.0f);
        float aend = fmin(min(end),1.0f);
        startPoint += a1*dir;

        const float sampling_distance = norm((aend-a1)*dir)/num_samples_per_ray;

        // Now perform conversion of the line integral start/end into voxel coordinates
        //

        startPoint /= is_dims_in_mm;
#ifdef FLIP_Z_AXIS
        startPoint[2] *= -1.0f;
#endif
        startPoint += 0.5f;
        dir /= is_dims_in_mm;
#ifdef FLIP_Z_AXIS
        dir[2] *= -1.0f;
#endif
        dir /= float(num_samples_per_ray); // now in step size units

        //
        // Perform line integration
        //

        float result = 0.0f;

        for ( int sampleIndex = 0; sampleIndex<num_samples_per_ray; sampleIndex++) {

#ifndef IS_ORIGIN_CENTERING
            floatd3 samplePoint = startPoint+dir*float(sampleIndex) + floatd3(0.5f)/is_dims_in_pixels;
#else
            floatd3 samplePoint = startPoint+dir*float(sampleIndex);
#endif

            // Accumulate result
            //

            result += tex3D( image_tex, samplePoint[0], samplePoint[1], samplePoint[2] );
        }

        // Output (normalized to the length of the ray)
        //

        projections[idx] = result*sampling_distance;
    }
}

//
// Forwards projection of a 3D volume onto a set of (binned) projections
//

void
conebeam_forwards_projection( hoCuNDArray<float> *projections,
                              hoCuNDArray<float> *image,
                              std::vector<float> angles,
                              std::vector<floatd2> offsets,
                              std::vector<unsigned int> indices,
                              int projections_per_batch,
                              float samples_per_pixel,
                              floatd3 is_dims_in_mm,
                              floatd2 ps_dims_in_mm,
                              float SDD,
                              float SAD)
{
    //
    // Validate the input
    //

    if( projections == 0x0 || image == 0x0 ){
        throw std::runtime_error("Error: conebeam_forwards_projection: illegal array pointer provided");
    }

    if( projections->get_number_of_dimensions() != 3 ){
        throw std::runtime_error("Error: conebeam_forwards_projection: projections array must be three-dimensional");
    }

    if( image->get_number_of_dimensions() != 3 ){
        throw std::runtime_error("Error: conebeam_forwards_projection: image array must be three-dimensional");
    }

    if( projections->get_size(2) != angles.size() || projections->get_size(2) != offsets.size() ) {
        throw std::runtime_error("Error: conebeam_forwards_projection: inconsistent sizes of input arrays/vectors");
    }

    int projection_res_x = projections->get_size(0);
    int projection_res_y = projections->get_size(1);

    int num_projections_in_bin = indices.size();
    int num_projections_in_all_bins = projections->get_size(2);

    int matrix_size_x = image->get_size(0);
    int matrix_size_y = image->get_size(1);
    int matrix_size_z = image->get_size(2);

    hoCuNDArray<float> *int_projections = projections;

    if( projections_per_batch > num_projections_in_bin )
        projections_per_batch = num_projections_in_bin;

    int num_batches = (num_projections_in_bin+projections_per_batch-1) / projections_per_batch;

    // Build texture from input image
    //

    cudaFuncSetCacheConfig(conebeam_forwards_projection_kernel, cudaFuncCachePreferL1);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaExtent extent;
    extent.width = matrix_size_x;
    extent.height = matrix_size_y;
    extent.depth = matrix_size_z;

    cudaMemcpy3DParms cpy_params = {0};
    cpy_params.kind = cudaMemcpyHostToDevice;
    cpy_params.extent = extent;

    cudaArray *image_array;
    cudaMalloc3DArray(&image_array, &channelDesc, extent);
    CHECK_FOR_CUDA_ERROR();

    cpy_params.dstArray = image_array;
    cpy_params.srcPtr = make_cudaPitchedPtr
            ((void*)image->get_data_ptr(), extent.width*sizeof(float), extent.width, extent.height);
    cudaMemcpy3D(&cpy_params);
    CHECK_FOR_CUDA_ERROR();

    cudaBindTextureToArray(image_tex, image_array, channelDesc);
    CHECK_FOR_CUDA_ERROR();

    // Allocate the angles, offsets and projections in device memory
    //

    float *projections_DevPtr, *projections_DevPtr2;
    cudaMalloc( (void**) &projections_DevPtr, projection_res_x*projection_res_y*projections_per_batch*sizeof(float));
    cudaMalloc( (void**) &projections_DevPtr2, projection_res_x*projection_res_y*projections_per_batch*sizeof(float));

    cudaStream_t mainStream, indyStream;
    cudaStreamCreate(&mainStream);
    cudaStreamCreate(&indyStream);

    std::vector<float> angles_vec;
    std::vector<floatd2> offsets_vec;

    for( int p=0; p<indices.size(); p++ ){

        int from_id = indices[p];

        if( from_id >= num_projections_in_all_bins ) {
            throw std::runtime_error("Error: conebeam_forwards_projection: illegal index in bin");
        }

        angles_vec.push_back(angles[from_id]);
        offsets_vec.push_back(offsets[from_id]);
    }

    thrust::device_vector<float> angles_devVec(angles_vec);
    thrust::device_vector<floatd2> offsets_devVec(offsets_vec);

    //
    // Iterate over the batches
    //

    for (unsigned int batch=0; batch<num_batches; batch++ ){

        int from_projection = batch * projections_per_batch;
        int to_projection = (batch+1) * projections_per_batch;

        if (to_projection > num_projections_in_bin)
            to_projection = num_projections_in_bin;

        int projections_in_batch = to_projection-from_projection;

        // Block/grid configuration
        //

        dim3 dimBlock, dimGrid;
        setup_grid( projection_res_x*projection_res_y*projections_in_batch, &dimBlock, &dimGrid );

        // Launch kernel
        //

        floatd3 is_dims_in_pixels(matrix_size_x, matrix_size_y, matrix_size_z);
        intd2 ps_dims_in_pixels(projection_res_x, projection_res_y);

        float* raw_angles = thrust::raw_pointer_cast(&angles_devVec[from_projection]);
        floatd2* raw_offsets = thrust::raw_pointer_cast(&offsets_devVec[from_projection]);

        conebeam_forwards_projection_kernel<<< dimGrid, dimBlock, 0, mainStream >>>
                                                                                  ( projections_DevPtr, raw_angles, raw_offsets,
                                                                                    is_dims_in_pixels, is_dims_in_mm, ps_dims_in_pixels, ps_dims_in_mm,
                                                                                    projections_in_batch, SDD, SAD, samples_per_pixel*float(matrix_size_x) );

        // If not initial batch, start copying the old stuff
        //

        int p = from_projection;
        while( p<to_projection) {

            int num_sequential_projections = 1;
            while( p+num_sequential_projections < to_projection &&
                   indices[p+num_sequential_projections]==(indices[p+num_sequential_projections-1]+1) ){
                num_sequential_projections++;
            }

            int to_id = indices[p];
            int size = projection_res_x*projection_res_y;

            cudaMemcpyAsync( int_projections->get_data_ptr()+to_id*size,
                             projections_DevPtr+(p-from_projection)*size,
                             size*num_sequential_projections*sizeof(float),
                             cudaMemcpyDeviceToHost, mainStream);

            p += num_sequential_projections;
        }

        std::swap(projections_DevPtr, projections_DevPtr2);
        std::swap(mainStream, indyStream);
    }

    cudaFree(projections_DevPtr2);
    cudaFree(projections_DevPtr);
    cudaFreeArray(image_array);

    CUDA_CALL(cudaStreamDestroy(indyStream));
    CUDA_CALL(cudaStreamDestroy(mainStream));
    CHECK_FOR_CUDA_ERROR();

}



//
// Forwards projection
//

__global__ void
conebeam_forwards_projection_kernel_cyl( float * __restrict__ projections,
                                         float * __restrict__ angles,
                                         floatd2 *offsets,
                                         floatd3 *mot_XYZ,
                                         floatd3 is_dims_in_pixels,
                                         floatd3 is_dims_in_mm,
                                         intd2 ps_dims_in_pixels_int,
                                         floatd2 ps_dims_in_mm,
                                         int num_projections,
                                         float SDD,
                                         float SAD,
                                         int num_samples_per_ray,
                                         bool ffs_)
{
    const int idx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x+threadIdx.x;
    const int num_elements = prod(ps_dims_in_pixels_int)*num_projections;

    if( idx < num_elements){

        const intd3 co = idx_to_co<3>( idx, intd3(ps_dims_in_pixels_int[0], ps_dims_in_pixels_int[1], num_projections) );

        // Projection space dimensions and spacing
        //

        const floatd2 ps_dims_in_pixels = floatd2(ps_dims_in_pixels_int[0], ps_dims_in_pixels_int[1]);
        const floatd2 ps_spacing = ps_dims_in_mm / ps_dims_in_pixels;

        // Determine projection angle and rotation matrix
        //

        const float angle = angles[co[2]];
        const float3x3 rotation = calcRotationMatrixAroundZ(degrees2radians(angle));

        // Find start and end point for the line integral (image space)
        //

        float ffsShift = 0.0f;
        float ffs_dir = 0.0f;
        float ffs_shift = 0.0f;
        if (idx == 31457380)
        {
            //printf("conbeam_projection.cu - ForwProject_Cyl Kernel \n");
        }
        if (ffs_) //This means flying focal spot is on
        {
            if (idx == 31457380)
            {
                //printf("FFS on \n");
            }
            ffs_shift = 0.25 * (2 * M_PI) * SDD / 800; // This currently hard codes view spacing of 0.45
            if ( co[2]%2 == 0)
            {
                ffs_dir = 1.0f;
            }
            else
            {
                ffs_dir = -1.0f;
            }
            ffsShift = ffs_shift*ffs_dir;
        }

        floatd3 startPoint = floatd3(ffsShift, -SAD, 0.0f);
        startPoint = mul(rotation, startPoint);

        // Projection plate indices
        //

#ifdef PS_ORIGIN_CENTERING
        const floatd2 ps_pc = floatd2(co[0], co[1]) + floatd2(0.5);
#else
        const floatd2 ps_pc = floatd2(co[0], co[1]);
#endif

        // Convert the projection plate coordinates into image space,
        // - local to the plate in metric units
        // - including half-fan and sag correction
        //

        const floatd2 proj_coords = (ps_pc / ps_dims_in_pixels - 0.5f) * ps_dims_in_mm + offsets[co[2]];

        // Define the end point for the line integrals
        //

        //const float ADD = SDD - SAD; // in mm.
        //		floatd3 endPoint = floatd3(proj_coords[0], ADD, proj_coords[1]);
        floatd3 endPoint = floatd3(SDD*std::sin(proj_coords[0]), SDD*std::cos(proj_coords[0]) - SAD, std::cos(proj_coords[0]) * proj_coords[1]);
        endPoint = mul(rotation, endPoint);

        // Find direction vector of the line integral
        //

        floatd3 dir = endPoint-startPoint;

        // Report out values
        /*if (idx == 31457380)
        {
            printf("conbeam_projection.cu - ForwProject_Cyl Kernel \n");
            printf("Values: \n");
            printf("ps_dims_in_pixels[0]: %f \n", ps_dims_in_pixels[0]);
            printf("ps_dims_in_pixels[1]: %f \n", ps_dims_in_pixels[1]);
            printf("ps_spacing[0]: %f \n", ps_spacing[0]);
            printf("ps_spacing[1]: %f \n", ps_spacing[1]);
            printf("Angle: %f \n", angle);
            printf("startPoint[0]: %f \n", startPoint[0]);
            printf("startPoint[1]: %f \n", startPoint[1]);
            printf("startPoint[2]: %f \n", startPoint[2]);
            printf("ps_pc[0]: %f \n", ps_pc[0]);
            printf("ps_pc[1]: %f \n", ps_pc[1]);
            printf("proj_coords[0]: %f \n", proj_coords[0]);
            printf("proj_coords[1]: %f \n", proj_coords[1]);
            printf("endPoint[0]: %f \n", endPoint[0]);
            printf("endPoint[1]: %f \n", endPoint[1]);
            printf("endPoint[2]: %f \n", endPoint[2]);
            printf("dir[0]: %f \n", dir[0]);
            printf("dir[1]: %f \n", dir[1]);
            printf("dir[2]: %f \n", dir[2]);

        }
        */


        // Perform integration only inside the bounding cylinder of the image volume

        const floatd3 vec_over_dir = (is_dims_in_mm-startPoint)/dir;
        const floatd3 vecdiff_over_dir = (-is_dims_in_mm-startPoint)/dir;
        const floatd3 start = amin(vecdiff_over_dir, vec_over_dir);
        const floatd3 end   = amax(vecdiff_over_dir, vec_over_dir);

        float a1 = fmax(max(start),0.0f);
        float aend = fmin(min(end),1.0f);



        // FC: Insert new box algorithm allowing for shift of box
        /*float tnear = -9999.0;
        float tfar = -9999.0;
        int flag_val = 0;
        float px = 0.0f;
        float t1a = 0.0f;
        float t2a = 0.0f;
        float t1 = 0.0f;
        float t2 = 0.0f;
        float a1 = 0.5f;
        float aend = 0.6f;

        floatd3 is_low_bounds = center_shift_in_mm - is_dims_in_mm/2;
        floatd3 is_upp_bounds = center_shift_in_mm + is_dims_in_mm/2;

        for (int dim_idx = 0; dim_idx<3; dim_idx++)
        {
            // For each dimension, lets figure out the bounds
            px = startPoint[dim_idx]; // Position of the starting point in this dimension
            t1a = is_low_bounds[dim_idx] - px / dir[dim_idx];
            t2a = is_upp_bounds[dim_idx] - px / dir[dim_idx];

            if (t1a>t2a) // If out of order, swap values
            {
                t1 = t2a;
                t2 = t1a;
            }
            else {
                t1 = t1a;
                t2 = t2a;
            }
            // Check if t1 is closer than tnear;
            if (t1 > tnear)
            {
                tnear = t1;
            }
            // Check if t2 is closer than tfar;
            if (t2 < tfar)
            {
                tfar = t2;
            }

            // Check for error
            if (tnear > tfar)
            {
                flag_val += 1; // If flag_val > 0, then it missed
            }

        }
        if (flag_val == 0)
        {
            a1=tnear;
            aend=tfar;
        }
        */


        startPoint += a1*dir;

        const float sampling_distance = norm((aend-a1)*dir)/num_samples_per_ray;

        // Shift start point based on motion vector
         // Define shift of object
         floatd3 mot_XYZ_val = mot_XYZ[co[2]];
         // Move points in opposite direction of backwards projection
         startPoint -= mot_XYZ_val;

        // Now perform conversion of the line integral start/end into voxel coordinates
        //

        startPoint /= is_dims_in_mm;
#ifdef FLIP_Z_AXIS
        startPoint[2] *= -1.0f;
#endif
        startPoint += 0.5f;
        dir /= is_dims_in_mm;
#ifdef FLIP_Z_AXIS
        dir[2] *= -1.0f;
#endif
        dir *= (aend-a1)/float(num_samples_per_ray); // now in step size units


        //
        // Perform line integration
        //


        float result = 0.0f;

        for ( int sampleIndex = 0; sampleIndex<num_samples_per_ray; sampleIndex++) {

#ifndef IS_ORIGIN_CENTERING
            floatd3 samplePoint = startPoint+dir*float(sampleIndex) + floatd3(0.5f)/is_dims_in_pixels;
#else
            floatd3 samplePoint = startPoint+dir*float(sampleIndex);
#endif

            // Accumulate result


            result += tex3D( image_tex, samplePoint[0], samplePoint[1], samplePoint[2] );
        }

        // Output (normalized to the length of the ray)
        //

        projections[idx] = result*sampling_distance;
    }
}

//
// Forwards projection of a 3D volume onto a set of (binned) projections
//

void
conebeam_forwards_projection_cyl(hoCuNDArray<float> *projections,
                                  hoCuNDArray<float> *image,
                                  std::vector<float> angles,
                                  std::vector<floatd2> offsets,
                                  std::vector<floatd3> mot_XYZ,
                                  std::vector<unsigned int> indices,
                                  int projections_per_batch,
                                  float samples_per_pixel,
                                  floatd3 is_dims_in_mm,
                                  floatd2 ps_dims_in_mm,
                                  float SDD,
                                  float SAD,
                                  bool ffs)
{
    //
    // Validate the input
    //

    if( projections == 0x0 || image == 0x0 ){
        throw std::runtime_error("Error: conebeam_forwards_projection: illegal array pointer provided");
    }

    if( projections->get_number_of_dimensions() != 3 ){
        throw std::runtime_error("Error: conebeam_forwards_projection: projections array must be three-dimensional");
    }

    if( image->get_number_of_dimensions() != 3 ){
        throw std::runtime_error("Error: conebeam_forwards_projection: image array must be three-dimensional");
    }

    if( projections->get_size(2) != angles.size() || projections->get_size(2) != offsets.size() ) {
        throw std::runtime_error("Error: conebeam_forwards_projection: inconsistent sizes of input arrays/vectors");
    }

    int projection_res_x = projections->get_size(0);
    int projection_res_y = projections->get_size(1);

    int num_projections_in_bin = indices.size();
    int num_projections_in_all_bins = projections->get_size(2);

    int matrix_size_x = image->get_size(0);
    int matrix_size_y = image->get_size(1);
    int matrix_size_z = image->get_size(2);

    hoCuNDArray<float> *int_projections = projections;

    if( projections_per_batch > num_projections_in_bin )
        projections_per_batch = num_projections_in_bin;

    int num_batches = (num_projections_in_bin+projections_per_batch-1) / projections_per_batch;

    // Build texture from input image
    //

    cudaFuncSetCacheConfig(conebeam_forwards_projection_kernel, cudaFuncCachePreferL1);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaExtent extent;
    extent.width = matrix_size_x;
    extent.height = matrix_size_y;
    extent.depth = matrix_size_z;

    cudaMemcpy3DParms cpy_params = {0};
    cpy_params.kind = cudaMemcpyHostToDevice;
    cpy_params.extent = extent;

    cudaArray *image_array;
    cudaMalloc3DArray(&image_array, &channelDesc, extent);
    CHECK_FOR_CUDA_ERROR();

    cpy_params.dstArray = image_array;
    cpy_params.srcPtr = make_cudaPitchedPtr
            ((void*)image->get_data_ptr(), extent.width*sizeof(float), extent.width, extent.height);
    cudaMemcpy3D(&cpy_params);
    CHECK_FOR_CUDA_ERROR();

    cudaBindTextureToArray(image_tex, image_array, channelDesc);
    CHECK_FOR_CUDA_ERROR();

    // Allocate the angles, offsets and projections in device memory
    //

    float *projections_DevPtr, *projections_DevPtr2;
    cudaMalloc( (void**) &projections_DevPtr, projection_res_x*projection_res_y*projections_per_batch*sizeof(float));
    cudaMalloc( (void**) &projections_DevPtr2, projection_res_x*projection_res_y*projections_per_batch*sizeof(float));

    cudaStream_t mainStream, indyStream;
    cudaStreamCreate(&mainStream);
    cudaStreamCreate(&indyStream);

    std::vector<float> angles_vec;
    std::vector<floatd2> offsets_vec;
    std::vector<floatd3> mot_XYZ_vec;

    for( int p=0; p<indices.size(); p++ ){

        int from_id = indices[p];

        if( from_id >= num_projections_in_all_bins ) {
            throw std::runtime_error("Error: conebeam_forwards_projection: illegal index in bin");
        }

        angles_vec.push_back(angles[from_id]);
        offsets_vec.push_back(offsets[from_id]);
        mot_XYZ_vec.push_back(mot_XYZ[from_id]);
    }

    thrust::device_vector<float> angles_devVec(angles_vec);
    thrust::device_vector<floatd2> offsets_devVec(offsets_vec);
    thrust::device_vector<floatd3> mot_XYZ_devVec(mot_XYZ_vec);

    //
    // Iterate over the batches
    //

    for (unsigned int batch=0; batch<num_batches; batch++ ){

        int from_projection = batch * projections_per_batch;
        int to_projection = (batch+1) * projections_per_batch;

        if (to_projection > num_projections_in_bin)
            to_projection = num_projections_in_bin;

        int projections_in_batch = to_projection-from_projection;

        // Block/grid configuration
        //

        dim3 dimBlock, dimGrid;
        setup_grid( projection_res_x*projection_res_y*projections_in_batch, &dimBlock, &dimGrid );

        // Launch kernel
        //

        floatd3 is_dims_in_pixels(matrix_size_x, matrix_size_y, matrix_size_z);
        intd2 ps_dims_in_pixels(projection_res_x, projection_res_y);

        float* raw_angles = thrust::raw_pointer_cast(&angles_devVec[from_projection]);
        floatd2* raw_offsets = thrust::raw_pointer_cast(&offsets_devVec[from_projection]);
        floatd3* raw_mot_XYZ = thrust::raw_pointer_cast(&mot_XYZ_devVec[from_projection]);

        conebeam_forwards_projection_kernel_cyl<<< dimGrid, dimBlock, 0, mainStream >>>
                ( projections_DevPtr, raw_angles, raw_offsets, raw_mot_XYZ,
                  is_dims_in_pixels, is_dims_in_mm, ps_dims_in_pixels, ps_dims_in_mm,
                  projections_in_batch, SDD, SAD, samples_per_pixel*float(matrix_size_x),ffs);

        // If not initial batch, start copying the old stuff
        //

        int p = from_projection;
        while( p<to_projection) {

            int num_sequential_projections = 1;
            while( p+num_sequential_projections < to_projection &&
                   indices[p+num_sequential_projections]==(indices[p+num_sequential_projections-1]+1) ){
                num_sequential_projections++;
            }

            int to_id = indices[p];
            int size = projection_res_x*projection_res_y;

            cudaMemcpyAsync( int_projections->get_data_ptr()+to_id*size,
                             projections_DevPtr+(p-from_projection)*size,
                             size*num_sequential_projections*sizeof(float),
                             cudaMemcpyDeviceToHost, mainStream);

            p += num_sequential_projections;
        }

        std::swap(projections_DevPtr, projections_DevPtr2);
        std::swap(mainStream, indyStream);
    }

    cudaFree(projections_DevPtr2);
    cudaFree(projections_DevPtr);
    cudaFreeArray(image_array);

    CUDA_CALL(cudaStreamDestroy(indyStream));
    CUDA_CALL(cudaStreamDestroy(mainStream));
    CHECK_FOR_CUDA_ERROR();

}
template <bool FBP> __global__ void
conebeam_backwards_projection_kernel( float * __restrict__ image,
                                      const float * __restrict__ angles,
                                      floatd2 *offsets,
                                      intd3 is_dims_in_pixels_int,
                                      floatd3 is_dims_in_mm,
                                      floatd2 ps_dims_in_pixels,
                                      floatd2 ps_dims_in_mm,
                                      int num_projections_in_batch,
                                      float num_projections_in_bin,
                                      float SDD,
                                      float SAD,
                                      bool accumulate )
{
    // Image voxel to backproject into (pixel coordinate and index)
    //

    const int idx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x+threadIdx.x;
    const int num_elements = prod(is_dims_in_pixels_int);

    if( idx < num_elements ){

        const intd3 co = idx_to_co<3>(idx, is_dims_in_pixels_int);

#ifdef IS_ORIGIN_CENTERING
        const floatd3 is_pc = floatd3(co[0], co[1], co[2]) + floatd3(0.5);
#else
        const floatd3 is_pc = floatd3(co[0], co[1], co[2]);
#endif

        // Normalized image space coordinate [-0.5, 0.5[
        //

        const floatd3 is_dims_in_pixels(is_dims_in_pixels_int[0],is_dims_in_pixels_int[1],is_dims_in_pixels_int[2]);

#ifdef FLIP_Z_AXIS
        floatd3 is_nc = is_pc / is_dims_in_pixels - floatd3(0.5f);
        is_nc[2] *= -1.0f;
#else
        const floatd3 is_nc = is_pc / is_dims_in_pixels - floatd3(0.5f);
#endif

        // Image space coordinate in metric units
        //

        if(idx == 0)
        {
            //    printf("conbeam_projection.cu - BackProject Kernel - Image Space Coordinates \n");
        }
        const floatd3 pos = is_nc * is_dims_in_mm;

        // Read the existing output value for accumulation at this point.
        // The cost of this fetch is hidden by the loop

        const float incoming = (accumulate) ? image[idx] : 0.0f;

        // Backprojection loop
        //

        float result = 0.0f;

        for( int projection = 0; projection < num_projections_in_batch; projection++ ) {

            // Projection angle
            //

            const float angle = degrees2radians(angles[projection]);

            // Projection rotation matrix
            //

            const float3x3 inverseRotation = calcRotationMatrixAroundZ(-angle);

            // Rotated image coordinate (local to the projection's coordinate system)
            //

            const floatd3 pos_proj = mul(inverseRotation, pos);

            // Project the image position onto the projection plate.
            // Account for half-fan and sag offsets.
            //

            const floatd3 startPoint = floatd3(0.0f, -SAD, 0.0f);
            floatd3 dir = pos_proj - startPoint;
            dir = dir / dir[1];
            const floatd3 endPoint = startPoint + dir * SDD;
            const floatd2 endPoint2d = floatd2(endPoint[0], endPoint[2]) - offsets[projection];

            // Convert metric projection coordinates into pixel coordinates
            //

#ifndef PS_ORIGIN_CENTERING
            floatd2 ps_pc = ((endPoint2d / ps_dims_in_mm) + floatd2(0.5f)) + floatd2(0.5f)/ps_dims_in_pixels;
            //floatd2 ps_pc = ((endPoint2d / ps_dims_in_mm) + floatd2(0.5f)) * ps_dims_in_pixels + floatd2(0.5f);
#else
            floatd2 ps_pc = ((endPoint2d / ps_dims_in_mm) + floatd2(0.5f));
#endif

            /*

            if (idx == 100 || idx == (num_elements-1) )
            {
                printf("Values: \n");
                printf("Projection: %u \n", projection);
                printf("Angle: %f \n", angle);
                printf("pos_proj[0]: %f \n", pos_proj[0]);
                printf("pos_proj[1]: %f \n", pos_proj[1]);
                printf("pos_proj[2]: %f \n", pos_proj[2]);
                printf("startPoint[0]: %f \n", startPoint[0]);
                printf("startPoint[1]: %f \n", startPoint[1]);
                printf("startPoint[2]: %f \n", startPoint[2]);
                printf("dir[0]: %f \n", dir[0]);
                printf("dir[1]: %f \n", dir[1]);
                printf("dir[2]: %f \n", dir[2]);
                printf("endPoint[0]: %f \n", endPoint[0]);
                printf("endPoint[1]: %f \n", endPoint[1]);
                printf("endPoint[2]: %f \n", endPoint[2]);
                printf("offsets[proj]: %f \n", offsets[projection]);

                printf("endPoint2d[0]: %f \n", endPoint2d[0]);
                printf("endPoint2d[1]: %f \n", endPoint2d[1]);
                printf("ps_pc[0]: %f \n", ps_pc[0]);
                printf("ps_pc[1]: %f \n", ps_pc[1]);
            }
            */
            // Apply filter (filtered backprojection mode only)
            //

            float weight = 1.0;

            if( FBP ){

                // Equation 3.59, page 96 and equation 10.2, page 386
                // in Computed Tomography 2nd edition, Jiang Hsieh
                //

                const float xx = pos[0];
                const float yy = pos[1];
                const float beta = angle;
                const float r = hypotf(xx,yy);
                const float phi = atan2f(yy,xx);
                const float D = SAD;
                const float ym = r*sinf(beta-phi);
                const float U = (D+ym)/D;
                weight = 1.0f/(U*U);
            }

            // Read the projection data (bilinear interpolation enabled) and accumulate
            //

            result +=  weight * tex2DLayered( projections_tex, ps_pc[0], ps_pc[1], projection );
        }

        // Output normalized image
        //

        image[idx] = incoming + result / num_projections_in_bin;
    }
}


template <bool FBP> __global__ void
conebeam_backwards_projection_cyl_kernel( float * __restrict__ image,
                                          const float * __restrict__ angles,
                                          floatd2 *offsets,
                                          floatd3 *mot_XYZ,
                                          intd3 is_dims_in_pixels_int,
                                          floatd3 is_dims_in_mm,
                                          floatd2 ps_dims_in_pixels,
                                          floatd2 ps_dims_in_mm,
                                          int num_projections_in_batch,
                                          float num_projections_in_bin,
                                          float SDD,
                                          float SAD,
                                          bool accumulate,
                                          bool ffs_)
{
    // Image voxel to backproject into (pixel coordinate and index)
    //

    const int idx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x+threadIdx.x;


    const int idx_middle = roundf(0.5*(gridDim.x*blockDim.x + gridDim.y*blockDim.y + gridDim.z*blockDim.z));
    //std::cout << "conbeam_projection.cu - BackProject_Cyl_Kernel Func. Detector  Boolean: " << use_cyl_det_ << std::endl;
    if (idx == idx_middle)
    {
         //printf("conbeam_projection.cu - Start BackProject_Cyl Kernel \n");
    }



    const int num_elements = prod(is_dims_in_pixels_int);

    if( idx < num_elements ){

        const intd3 co = idx_to_co<3>(idx, is_dims_in_pixels_int);

#ifdef IS_ORIGIN_CENTERING
        const floatd3 is_pc = floatd3(co[0], co[1], co[2]) + floatd3(0.5);
#else
        const floatd3 is_pc = floatd3(co[0], co[1], co[2]);
#endif

        // Normalized image space coordinate [-0.5, 0.5[
        //

        const floatd3 is_dims_in_pixels(is_dims_in_pixels_int[0],is_dims_in_pixels_int[1],is_dims_in_pixels_int[2]);

#ifdef FLIP_Z_AXIS
        floatd3 is_nc = is_pc / is_dims_in_pixels - floatd3(0.5f);
        is_nc[2] *= -1.0f;
#else
        const floatd3 is_nc = is_pc / is_dims_in_pixels - floatd3(0.5f);
#endif

        // FC added FOV shift to point location
        //is_nc += center_shift_in_mm;

        // Image space coordinate in metric units
        //

        if (idx == idx_middle)
        {
            //    printf("conbeam_projection.cu - BackProject_Cyl Kernel - Image Space Coordinates \n");
        }
        const floatd3 pos = is_nc * is_dims_in_mm;

        // Read the existing output value for accumulation at this point.
        // The cost of this fetch is hidden by the loop

        const float incoming = (accumulate) ? image[idx] : 0.0f;

        // Backprojection loop
        //

        float result = 0.0f;

        if (idx == idx_middle)
        {
            //   printf("conbeam_projection.cu - BackProject_Cyl Kernel - Start Backprojection Loop\n");
        }


        for( int projection = 0; projection < num_projections_in_batch; projection++ ) {

            // Projection angle
            //


            // FC: Lets move object pos given mot value
            floatd3  mot_XYZ_val = mot_XYZ[projection];
            floatd3  pos2 = pos + mot_XYZ_val;
            if (idx == idx_middle)
            {
                // printf("conbeam_projection.cu - BackProject_Cyl Kernel - BP Angle \n");
                // FC: Check that we have mot_XYZ_ vector
                //printf("proj =  %d, x: %f, y: %f, z: %f \n",projection, mot_XYZ_val[0], mot_XYZ_val[1], mot_XYZ_val[2]);
            }


            const float angle = degrees2radians(angles[projection]);

            // Projection rotation matrix
            //

            const float3x3 inverseRotation = calcRotationMatrixAroundZ(-angle);

            // Rotated image coordinate (local to the projection's coordinate system)
            //

            const floatd3 pos_proj = mul(inverseRotation, pos2);

            // Project the image position onto the projection plate.
            // Account for half-fan and sag offsets.
            //

            if (idx == idx_middle)
            {
                // printf("conbeam_projection.cu - BackProject_Cyl Kernel - BP Start and End Point \n");
            }


            float ffsShift = 0.0f;
            float ffs_dir = 0.0f;
            float ffs_shift = 0.0f;
            if (ffs_) //This means flying focal spot is on
            {
                ffs_shift = 0.25 * (2 * M_PI) * SDD / 800; // This currently hard codes view spacing of 0.45
                if ( projection%2 == 0) {
                    ffs_dir = 1.0f;
                }
                else {
                    ffs_dir = -1.0f;
                }
                ffsShift = ffs_shift*ffs_dir;
            }


            // X POSITION CAN BE SHIFTED BY FFS
            const floatd3 startPoint = floatd3(ffsShift, -SAD, 0.0f);
            floatd3 dir = pos_proj - startPoint;
            dir = dir / dir[1];

            // Find angle R from vector dir
            float Theta = atan2f(dir[0],dir[1]);
            float R = Theta + (M_PI)/2;
            float S = asin(ffsShift*sinf(R)/SDD);
            // This angle is the one from the detector focal spot that hits the same point on the detector as the shifted FFS
            float Psi = S + Theta;

            // Do we need to change epsi calculation?
            // Don't believe so
            const floatd3 startPoint_orig = floatd3(0, -SAD, 0.0f);
            floatd3 dir_orig = pos_proj - startPoint_orig;
            dir_orig = dir_orig / dir_orig[1];
            // This is the endpoint on the flat detector
            const floatd3 endPoint_orig = startPoint_orig + dir_orig * SDD;
            // This is the angle from the detector focal spot thorugh the pixel to the detector
            const float psi = atan2f(endPoint_orig[0],SDD);
            const float epsi = endPoint_orig[2] * std::cos(psi);

            if (idx == idx_middle)
            {
            //    printf("conbeam_projection.cu - BackProject_Cyl Kernel - Using Cyl Det \n");
                //printf("Orig psi: %f \n", psi);
                //printf("FFS psi: %f \n", Psi);
            }

            floatd2 endPoint2d;
            // Replace detector focal angle with flying focal spot adjusted angle
            if (ffs_) //This means flying focal spot is on
            {
                if (idx == idx_middle)
                {
                    //printf("conbeam_projection.cu - BackProject_Cyl Kernel - Using FFS \n");
                }
                endPoint2d = floatd2(Psi, epsi) - offsets[projection];
            }
            else
            {
                if (idx == idx_middle)
                {
                    //printf("conbeam_projection.cu - BackProject_Cyl Kernel - FFS OFF \n");
                }
                endPoint2d = floatd2(psi, epsi) - offsets[projection];
            }



                if (idx == idx_middle)
                {
                    //printf("Values: \n");
                    //printf("Projection: %u \n", projection);
                    //printf("Angle: %f \n", angle);
                    /*
                     * printf("pos_proj[0]: %f \n", pos_proj[0]);
                    printf("pos_proj[1]: %f \n", pos_proj[1]);
                    printf("pos_proj[2]: %f \n", pos_proj[2]);
                    printf("startPoint[0]: %f \n", startPoint[0]);
                    printf("startPoint[1]: %f \n", startPoint[1]);
                    printf("startPoint[2]: %f \n", startPoint[2]);
                    printf("dir[0]: %f \n", dir[0]);
                    printf("dir[1]: %f \n", dir[1]);
                    printf("dir[2]: %f \n", dir[2]);
                    printf("endPoint[0]: %f \n", endPoint[0]);
                    printf("endPoint[1]: %f \n", endPoint[1]);
                    printf("endPoint[2]: %f \n", endPoint[2]);
                    printf("offsets[proj]: %f \n", offsets[projection]);

                    printf("endPoint2d[0]: %f \n", endPoint2d[0]);
                    printf("endPoint2d[1]: %f \n", endPoint2d[1]);
                    printf("Psi: %f \n", psi);
                    printf("Epsi: %f \n", epsi);
                    */
                }



            // Convert metric projection coordinates into pixel coordinates
            //
            // FC change ps_dims_in_mm to be  angular for dim[0]
            floatd2 ps_dims_in_mm_cyl;
            float delta = std::atan(ps_dims_in_mm[0]/(2.0f*SDD)); // half-Fan angle in rad
            ps_dims_in_mm_cyl[0]=ps_dims_in_mm[0];
            //ps_dims_in_mm_cyl[0]=10.0*ps_dims_in_mm[0];
            ps_dims_in_mm_cyl[1]=ps_dims_in_mm[1];

#ifndef PS_ORIGIN_CENTERING
            floatd2 ps_pc = ((endPoint2d / ps_dims_in_mm_cyl) + floatd2(0.5f)) + floatd2(0.5f)/ps_dims_in_pixels;
            //floatd2 ps_pc = ((endPoint2d / ps_dims_in_mm) + floatd2(0.5f)) * ps_dims_in_pixels + floatd2(0.5f);
#else
            floatd2 ps_pc = ((endPoint2d / ps_dims_in_mm_cyl) + floatd2(0.5f));
#endif

            /*
            if (idx == idx_middle)
            {
                printf("Values: \n");
                printf("ps_dims_in_mm[0]: %f \n", ps_dims_in_mm[0]);
                printf("ps_dims_in_mm[1]: %f \n", ps_dims_in_mm[1]);

                printf("ps_dims_in_mm_cyl[0]: %f \n", ps_dims_in_mm_cyl[0]);
                printf("ps_dims_in_mm_cyl[1]: %f \n", ps_dims_in_mm_cyl[1]);

                printf("ps_pc[0]: %f \n", ps_pc[0]);
                printf("ps_pc[1]: %f \n", ps_pc[1]);
            }

            */

            // Apply filter (filtered backprojection mode only)
            //

            float weight = 1.0;

            if( FBP ){

                // Equation 3.59, page 96 and equation 10.2, page 386
                // in Computed Tomography 2nd edition, Jiang Hsieh
                //


                //FC: Do we need to edit this given flying focal spot?

                const float xx = pos[0];
                const float yy = pos[1];
                const float zz = pos[2];

                // Find distance from xx,yy to source
                const float beta = angle;
                const float r = hypotf(xx,yy);
                const float phi = atan2f(yy,xx);
                const float D = SAD;

                // Ym is the y distance of the pixel from the source
                const float ym = r*sinf(beta-phi);
                const float xm = r*cosf(beta-phi);

                // Lets find the distance in the x-y plane
                const float L1 = std::sqrt( (float) (D+ym)*(D+ym)+xm*xm );

                // Now find the 3D distance
                // This assumes Z is 0 for x-y plane at center of detector
                const float L = std::sqrt( L1*L1+zz*zz );

                weight = 1.0f*(D*D)/(L*L);

                /*
                    if (idx == idx_middle)
                    {
                        printf("conbeam_projection.cu/BP_Cyl_Kernel Calculated Cyl Weight \n");
                        printf("Weighting Values: \n");
                        printf("XX: %f \n", xx);
                        printf("YY: %f \n", yy);
                        printf("ZZ: %f \n", zz);
                        printf("Beta: %f \n", beta);
                        printf("r: %f \n", r);
                        printf("phi: %f \n", phi);
                        printf("D: %f \n", D);
                        printf("ym: %f \n", ym);
                        printf("xm: %f \n", xm);
                        printf("L1: %f \n", L1);
                        printf("L: %f \n", L);

                        const float xx = pos[0];
                        const float yy = pos[1];
                        const float beta = angle;
                        const float r = hypotf(xx,yy);
                        const float phi = atan2f(yy,xx);
                        const float D = SAD;
                        const float ym = r*sinf(beta-phi);
                        const float U = (D+ym)/D;

                        printf("U: %f \n", U);

                        printf("weight: %f \n", weight);


                    }
                    */

                //}
                /*
                 * else
                {

                const float xx = pos[0];
                const float yy = pos[1];
                const float beta = angle;
                const float r = hypotf(xx,yy);
                const float phi = atan2f(yy,xx);
                const float D = SAD;
                const float ym = r*sinf(beta-phi);
                const float U = (D+ym)/D;
                weight = 1.0f/(U*U);
                }
                */
            }

            // Read the projection data (bilinear interpolation enabled) and accumulate
            //

            result +=  weight * tex2DLayered( projections_tex, ps_pc[0], ps_pc[1], projection );
        }

        // Output normalized image
        //

        image[idx] = incoming + result / num_projections_in_bin;
    }
}


//
// Backprojection
//

template <bool FBP>
void conebeam_backwards_projection(hoCuNDArray<float> *projections,
                                   hoCuNDArray<float> *image,
                                   std::vector<float> angles,
                                   std::vector<floatd2> offsets,
                                   std::vector<unsigned int> indices,
                                   int projections_per_batch,
                                   intd3 is_dims_in_pixels,
                                   floatd3 is_dims_in_mm,
                                   floatd2 ps_dims_in_mm,
                                   float SDD,
                                   float SAD,
                                   bool short_scan,
                                   bool use_offset_correction,
                                   bool accumulate,
                                   cuNDArray<float> *cosine_weights,
                                   cuNDArray<float> *frequency_filter
                                   )
{
    printf("Input Parameters....\n");
    printf("Is_Dims_In_Pixels: X = %d , Y=%d, Z = %d\n",is_dims_in_pixels[0],is_dims_in_pixels[1],is_dims_in_pixels[2]);
    printf("Is_Dims_In_MM: X = %f , Y=%f, Z = %f\n",is_dims_in_mm[0],is_dims_in_mm[1],is_dims_in_mm[2]);
    printf("Ps_Dims_In_MM: X = %f , Y=%f\n",ps_dims_in_mm[0],ps_dims_in_mm[1]);
    printf("SDD: %f \n",SDD);
    printf("SAD: %f \n",SAD);

    //
    // Validate the input
    //
    printf("Validating input....\n");

    if( projections == 0x0 || image == 0x0 ){
        throw std::runtime_error("Error: conebeam_backwards_projection: illegal array pointer provided");
    }

    if( projections->get_number_of_dimensions() != 3 ){
        throw std::runtime_error("Error: conebeam_backwards_projection: projections array must be three-dimensional");
    }

    if( image->get_number_of_dimensions() != 3 ){
        throw std::runtime_error("Error: conebeam_backwards_projection: image array must be three-dimensional");
    }

    if( projections->get_size(2) != angles.size() || projections->get_size(2) != offsets.size() ) {
        throw std::runtime_error("Error: conebeam_backwards_projection: inconsistent sizes of input arrays/vectors");
    }

    if( FBP && !(cosine_weights && frequency_filter) ){
        throw std::runtime_error("Error: conebeam_backwards_projection: for _filtered_ backprojection both cosine weights and a filter must be provided");
    }

    printf("Validating input .... DONE\n");
    // Some utility variables
    //
    printf("Utility Variables....\n");

    int matrix_size_x = image->get_size(0);
    int matrix_size_y = image->get_size(1);
    int matrix_size_z = image->get_size(2);
    printf("Matrix Size: X = %d , Y=%d, Z = %d\n",matrix_size_x,matrix_size_y,matrix_size_z);

    floatd3 is_dims(matrix_size_x, matrix_size_y, matrix_size_z);
    int num_image_elements = matrix_size_x*matrix_size_y*matrix_size_z;
    printf("Num Image Elements: %d \n",num_image_elements);

    int projection_res_x = projections->get_size(0);
    int projection_res_y = projections->get_size(1);
    printf("Projection Res: X = %d , Y=%d\n",projection_res_x,projection_res_y);

    floatd2 ps_dims_in_pixels(projection_res_x, projection_res_y);

    int num_projections_in_all_bins = projections->get_size(2);
    int num_projections_in_bin = indices.size();
    printf("Num Proj in All Bins: %d \n",num_projections_in_all_bins);
    printf("Num Proj in Each Bins: %d \n",num_projections_in_bin);

    if( projections_per_batch > num_projections_in_bin )
        projections_per_batch = num_projections_in_bin;

    int num_batches = (num_projections_in_bin+projections_per_batch-1) / projections_per_batch;
    printf("Num Batches: %d \n",num_batches);


    printf("Utility Variables .... DONE\n");

    // Allocate device memory for the backprojection result
    //
    printf("Allocating Device Memory....\n");


    boost::shared_ptr< cuNDArray<float> > image_device;

    if( accumulate ){
        image_device = boost::shared_ptr< cuNDArray<float> >(new cuNDArray<float>(image));
    }
    else{
        image_device = boost::shared_ptr< cuNDArray<float> >(new cuNDArray<float>(image->get_dimensions().get()));
    }
    printf("Allocating Device Memory .... DONE\n");


    // Allocate the angles, offsets and projections in device memory
    //
    printf("Allocating Angles, Offsets, Projections ....\n");

    float *projections_DevPtr, *projections_DevPtr2;
    cudaMalloc( (void**) &projections_DevPtr, projection_res_x*projection_res_y*projections_per_batch*sizeof(float));
    cudaMalloc( (void**) &projections_DevPtr2, projection_res_x*projection_res_y*projections_per_batch*sizeof(float));

    cudaStream_t mainStream, indyStream;
    cudaStreamCreate(&mainStream);
    cudaStreamCreate(&indyStream);

    std::vector<float> angles_vec;
    std::vector<floatd2> offsets_vec;

    for( int p=0; p<indices.size(); p++ ){

        int from_id = indices[p];

        if( from_id >= num_projections_in_all_bins ) {
            throw std::runtime_error("Error: conebeam_backwards_projection: illegal index in bin");
        }

        angles_vec.push_back(angles[from_id]);
        offsets_vec.push_back(offsets[from_id]);
    }

    thrust::device_vector<float> angles_devVec(angles_vec);
    thrust::device_vector<floatd2> offsets_devVec(offsets_vec);

    printf("Allocating Angles, Offsets, Projections .... DONE\n");

    // From/to for the first batch
    // - to enable working streams...
    //
    printf("First Batch From/To .... \n");


    int from_projection = 0;
    int to_projection = projections_per_batch;

    if (to_projection > num_projections_in_bin )
        to_projection = num_projections_in_bin;

    int projections_in_batch = to_projection-from_projection;

    std::vector<size_t> dims;
    dims.push_back(projection_res_x);
    dims.push_back(projection_res_y);
    dims.push_back(projections_in_batch);

    std::vector<size_t> dims_next;

    cuNDArray<float> *projections_batch = new cuNDArray<float>(&dims, projections_DevPtr);

    // Upload first projections batch adhering to the binning.
    // Be sure to copy sequentially numbered projections in one copy operation.
    //

    {
        int p = from_projection;

        while( p<to_projection ) {

            int num_sequential_projections = 1;
            while( p+num_sequential_projections < to_projection &&
                   indices[p+num_sequential_projections]==(indices[p+num_sequential_projections-1]+1) ){
                num_sequential_projections++;
            }

            int from_id = indices[p];
            int size = projection_res_x*projection_res_y;

            cudaMemcpyAsync( projections_batch->get_data_ptr()+(p-from_projection)*size,
                             projections->get_data_ptr()+from_id*size,
                             size*num_sequential_projections*sizeof(float), cudaMemcpyHostToDevice, mainStream );

            CHECK_FOR_CUDA_ERROR();

            p += num_sequential_projections;
        }
    }
    printf("First Batch From/To .... DONE\n");


    //
    // Iterate over batches
    //
    printf("Iterate Batches .... \n");

    for( int batch = 0; batch < num_batches; batch++ ) {

        from_projection = batch * projections_per_batch;
        to_projection = (batch+1) * projections_per_batch;

        if (to_projection > num_projections_in_bin )
            to_projection = num_projections_in_bin;

        projections_in_batch = to_projection-from_projection;

        float* raw_angles = thrust::raw_pointer_cast(&angles_devVec[from_projection]);
        floatd2* raw_offsets = thrust::raw_pointer_cast(&offsets_devVec[from_projection]);


        if( FBP ){

            // Apply cosine weighting : "SDD / sqrt(SDD*SDD + u*u + v*v)"
            // - with (u,v) positions given in metric units on a virtual detector at the origin
            //

            printf("Apply Cosine Weights .... \n");
            *projections_batch *= *cosine_weights;
            printf("Apply Cosine Weights .... DONE\n");

            // Redundancy correct
            // - for short scan mode
            //

            if( short_scan ){
                printf("Short Scan Redundancy .... \n");
                float delta = std::atan(ps_dims_in_mm[0]/(2.0f*SDD));
                redundancy_correct( projections_batch, raw_angles, delta );
                printf("Short Scan Redundancy .... DONE\n");
            }

            // Apply frequency filter
            // - use zero padding to avoid the cyclic boundary conditions induced by the fft
            //
            printf("Apply Freq Filter .... \n");

            std::vector<size_t> batch_dims = *projections_batch->get_dimensions();
            uint64d3 pad_dims(batch_dims[0]<<1, batch_dims[1], batch_dims[2]);
            boost::shared_ptr< cuNDArray<float> > padded_projections = pad<float,3>( pad_dims, projections_batch );
            boost::shared_ptr< cuNDArray<complext<float> > > complex_projections = cb_fft( padded_projections.get() );
            *complex_projections *= *frequency_filter;
            cb_ifft( complex_projections.get(), padded_projections.get() );
            uint64d3 crop_offsets(batch_dims[0]>>1, 0, 0);
            crop<float,3>( crop_offsets, padded_projections.get(), projections_batch );
            printf("Apply Freq Filter .... DONE\n");

            // Apply offset correction
            // - for half fan mode, sag correction etc.
            //
            if (use_offset_correction)
            {
                printf("Apply Offset Correction .... \n");
                offset_correct( projections_batch, raw_offsets, ps_dims_in_mm, SAD, SDD );
                printf("Apply Offset Correction .... DONE\n");
            }


        } else if (use_offset_correction)
            offset_correct_sqrt( projections_batch, raw_offsets, ps_dims_in_mm, SAD, SDD );

        // Build array for input texture
        //
        printf("Build Cuda Texture .... \n");
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
        cudaExtent extent;
        extent.width = projection_res_x;
        extent.height = projection_res_y;
        extent.depth = projections_in_batch;

        cudaArray *projections_array;
        cudaMalloc3DArray( &projections_array, &channelDesc, extent, cudaArrayLayered );
        CHECK_FOR_CUDA_ERROR();

        cudaMemcpy3DParms cpy_params = {0};
        cpy_params.extent = extent;
        cpy_params.dstArray = projections_array;
        cpy_params.kind = cudaMemcpyDeviceToDevice;
        cpy_params.srcPtr =
                make_cudaPitchedPtr( (void*)projections_batch->get_data_ptr(), projection_res_x*sizeof(float),
                                     projection_res_x, projection_res_y );
        cudaMemcpy3DAsync( &cpy_params, mainStream );
        CHECK_FOR_CUDA_ERROR();

        cudaBindTextureToArray( projections_tex, projections_array, channelDesc );
        CHECK_FOR_CUDA_ERROR();
        printf("Build Cuda Texture .... DONE\n");


        // Upload projections for the next batch
        // - to enable streaming
        //
        printf("Upload next batch .... \n");

        if( batch < num_batches-1 ){ // for using multiple streams to hide the cost of the uploads

            int from_projection_next = (batch+1) * projections_per_batch;
            int to_projection_next = (batch+2) * projections_per_batch;

            if (to_projection_next > num_projections_in_bin )
                to_projection_next = num_projections_in_bin;

            int projections_in_batch_next = to_projection_next-from_projection_next;

            // printf("batch: %03i, handling projections: %03i - %03i, angles: %.2f - %.2f\n",
            //	 batch+1, from_projection_next, to_projection_next-1, angles[from_projection_next], angles[to_projection_next-1]);

            // Allocate device memory for projections and upload
            //

            dims_next.clear();
            dims_next.push_back(projection_res_x);
            dims_next.push_back(projection_res_y);
            dims_next.push_back(projections_in_batch_next);

            cuNDArray<float> projections_batch_next(&dims, projections_DevPtr2);

            // Upload projections adhering to the binning.
            // Be sure to copy sequentially numbered projections in one copy operation.
            //

            int p = from_projection_next;

            while( p<to_projection_next ) {

                int num_sequential_projections = 1;
                while( p+num_sequential_projections < to_projection_next &&
                       indices[p+num_sequential_projections]==(indices[p+num_sequential_projections-1]+1) ){
                    num_sequential_projections++;
                }

                int from_id = indices[p];
                int size = projection_res_x*projection_res_y;

                cudaMemcpyAsync( projections_batch_next.get_data_ptr()+(p-from_projection_next)*size,
                                 projections->get_data_ptr()+from_id*size,
                                 size*num_sequential_projections*sizeof(float), cudaMemcpyHostToDevice, indyStream );

                CHECK_FOR_CUDA_ERROR();

                p += num_sequential_projections;
            }
        }
        printf("Upload next batch .... DONE\n");


        // Define dimensions of grid/blocks.
        //

        dim3 dimBlock, dimGrid;
        setup_grid( matrix_size_x*matrix_size_y*matrix_size_z, &dimBlock, &dimGrid );

        // Invoke kernel
        //

        printf("Invoke Kernel .... \n");
        cudaFuncSetCacheConfig(conebeam_backwards_projection_kernel<FBP>, cudaFuncCachePreferL1);

        conebeam_backwards_projection_kernel<FBP><<< dimGrid, dimBlock, 0, mainStream >>>
                                                                                        ( image_device->get_data_ptr(), raw_angles, raw_offsets,
                                                                                          is_dims_in_pixels, is_dims_in_mm, ps_dims_in_pixels, ps_dims_in_mm,
                                                                                          projections_in_batch, num_projections_in_bin, SDD, SAD, (batch==0) ? accumulate : true );

        CHECK_FOR_CUDA_ERROR();
        printf("Invoke Kernel .... DONE\n");


        // Cleanup
        //
        printf("Cleanup .... \n");


        cudaUnbindTexture(projections_tex);
        cudaFreeArray(projections_array);
        CHECK_FOR_CUDA_ERROR();

        std::swap(projections_DevPtr, projections_DevPtr2);
        std::swap(mainStream, indyStream);

        delete projections_batch;
        if( batch < num_batches-1 )
            projections_batch = new cuNDArray<float>(&dims_next, projections_DevPtr);
        printf("Cleanup .... DONE\n");

    }

    // Copy result from device to host
    //

    cudaMemcpy( image->get_data_ptr(), image_device->get_data_ptr(),
                num_image_elements*sizeof(float), cudaMemcpyDeviceToHost );

    CHECK_FOR_CUDA_ERROR();

    cudaFree(projections_DevPtr2);
    cudaFree(projections_DevPtr);
    CUDA_CALL(cudaStreamDestroy(indyStream));
    CUDA_CALL(cudaStreamDestroy(mainStream));
    CHECK_FOR_CUDA_ERROR();
}

//
// Cylindrical Detector Backprojection
//

template <bool FBP>
void conebeam_backwards_projection_cyl( hoCuNDArray<float> *projections,
                                        hoCuNDArray<float> *image,
                                        std::vector<float> angles,
                                        std::vector<floatd2> offsets,
                                        std::vector<unsigned int> indices,
                                        int projections_per_batch,
                                        intd3 is_dims_in_pixels,
                                        floatd3 is_dims_in_mm,
                                        floatd2 ps_dims_in_mm,
                                        float SDD,
                                        float SAD,
                                        bool short_scan,
                                        bool use_offset_correction,
                                        bool accumulate,
                                        std::vector<floatd3> mot_XYZ,
                                        bool ffs,
                                        cuNDArray<float> *cosine_weights,
                                        cuNDArray<float> *frequency_filter
                                        )
{
    // printf("conbeam_projection.cu - Start BackProject_Cyl Kernel \n");

    printf("Input Parameters....\n");
    printf("Is_Dims_In_Pixels: X = %d , Y=%d, Z = %d\n",is_dims_in_pixels[0],is_dims_in_pixels[1],is_dims_in_pixels[2]);
    printf("Is_Dims_In_MM: X = %f , Y=%f, Z = %f\n",is_dims_in_mm[0],is_dims_in_mm[1],is_dims_in_mm[2]);
    printf("Ps_Dims_In_MM: X = %f , Y=%f\n",ps_dims_in_mm[0],ps_dims_in_mm[1]);
    printf("SDD: %f \n",SDD);
    printf("SAD: %f \n",SAD);

    //
    // Validate the input
    //
    // printf("Validating input....\n");

    if( projections == 0x0 || image == 0x0 ){
        throw std::runtime_error("Error: conebeam_backwards_projection_cyl: illegal array pointer provided");
    }

    if( projections->get_number_of_dimensions() != 3 ){
        throw std::runtime_error("Error: conebeam_backwards_projection_cyl: projections array must be three-dimensional");
    }

    if( image->get_number_of_dimensions() != 3 ){
        throw std::runtime_error("Error: conebeam_backwards_projection_cyl: image array must be three-dimensional");
    }

    if( projections->get_size(2) != angles.size() || projections->get_size(2) != offsets.size() ) {
        throw std::runtime_error("Error: conebeam_backwards_projection_cyl: inconsistent sizes of input arrays/vectors");
    }

    if( FBP && !(cosine_weights && frequency_filter) ){
        throw std::runtime_error("Error: conebeam_backwards_projection_cyl: for _filtered_ backprojection both cosine weights and a filter must be provided");
    }

    // printf("Validating input .... DONE\n");
    // Some utility variables
    //
    // printf("Utility Variables....\n");

    int matrix_size_x = image->get_size(0);
    int matrix_size_y = image->get_size(1);
    int matrix_size_z = image->get_size(2);
    printf("Matrix Size: X = %d , Y=%d, Z = %d\n",matrix_size_x,matrix_size_y,matrix_size_z);

    floatd3 is_dims(matrix_size_x, matrix_size_y, matrix_size_z);
    int num_image_elements = matrix_size_x*matrix_size_y*matrix_size_z;
    printf("Num Image Elements: %d \n",num_image_elements);

    int projection_res_x = projections->get_size(0);
    int projection_res_y = projections->get_size(1);
    printf("Projection Res: X = %d , Y=%d\n",projection_res_x,projection_res_y);

    floatd2 ps_dims_in_pixels(projection_res_x, projection_res_y);

    int num_projections_in_all_bins = projections->get_size(2);
    int num_projections_in_bin = indices.size();
    printf("Num Proj in All Bins: %d \n",num_projections_in_all_bins);
    printf("Num Proj in Each Bins: %d \n",num_projections_in_bin);

    if( projections_per_batch > num_projections_in_bin )
        projections_per_batch = num_projections_in_bin;

    int num_batches = (num_projections_in_bin+projections_per_batch-1) / projections_per_batch;
    printf("Num Batches: %d \n",num_batches);

    // FC: Check that we have mot_XYZ_ vector
    /*floatd3 mot_XYZ_val;
    for( unsigned int i=0; i<projections->get_size(2); i++ )
    {
        mot_XYZ_val = mot_XYZ[i];
        std::cout << "i =  " << i << ", x: " << mot_XYZ_val[0] << ", y: " << mot_XYZ_val[1] << ", z: " << mot_XYZ_val[2] << std::endl;
    }
    */


    // printf("Utility Variables .... DONE\n");

    // Allocate device memory for the backprojection result
    //
    // printf("Allocating Device Memory....\n");


    boost::shared_ptr< cuNDArray<float> > image_device;

    if( accumulate ){
        image_device = boost::shared_ptr< cuNDArray<float> >(new cuNDArray<float>(image));
    }
    else{
        image_device = boost::shared_ptr< cuNDArray<float> >(new cuNDArray<float>(image->get_dimensions().get()));
    }
    // printf("Allocating Device Memory .... DONE\n");


    // Allocate the angles, offsets and projections in device memory
    //
    // printf("Allocating Angles, Offsets, Projections ....\n");

    float *projections_DevPtr, *projections_DevPtr2;
    cudaMalloc( (void**) &projections_DevPtr, projection_res_x*projection_res_y*projections_per_batch*sizeof(float));
    cudaMalloc( (void**) &projections_DevPtr2, projection_res_x*projection_res_y*projections_per_batch*sizeof(float));

    cudaStream_t mainStream, indyStream;
    cudaStreamCreate(&mainStream);
    cudaStreamCreate(&indyStream);

    std::vector<float> angles_vec;
    std::vector<floatd2> offsets_vec;
    std::vector<floatd3> mot_XYZ_vec;

    for( int p=0; p<indices.size(); p++ ){

        int from_id = indices[p];

        if( from_id >= num_projections_in_all_bins ) {
            throw std::runtime_error("Error: conebeam_backwards_projection: illegal index in bin");
        }

        angles_vec.push_back(angles[from_id]);
        offsets_vec.push_back(offsets[from_id]);
        mot_XYZ_vec.push_back(mot_XYZ[from_id]);
    }

    thrust::device_vector<float> angles_devVec(angles_vec);
    thrust::device_vector<floatd2> offsets_devVec(offsets_vec);
    thrust::device_vector<floatd3> mot_XYZ_devVec(mot_XYZ_vec);

    // printf("Allocating Angles, Offsets, Projections .... DONE\n");

    // From/to for the first batch
    // - to enable working streams...
    //
    // printf("First Batch From/To .... \n");


    int from_projection = 0;
    int to_projection = projections_per_batch;

    if (to_projection > num_projections_in_bin )
        to_projection = num_projections_in_bin;

    int projections_in_batch = to_projection-from_projection;

    std::vector<size_t> dims;
    dims.push_back(projection_res_x);
    dims.push_back(projection_res_y);
    dims.push_back(projections_in_batch);

    std::vector<size_t> dims_next;

    cuNDArray<float> *projections_batch = new cuNDArray<float>(&dims, projections_DevPtr);

    // Upload first projections batch adhering to the binning.
    // Be sure to copy sequentially numbered projections in one copy operation.
    //

    {
        int p = from_projection;

        while( p<to_projection ) {

            int num_sequential_projections = 1;
            while( p+num_sequential_projections < to_projection &&
                   indices[p+num_sequential_projections]==(indices[p+num_sequential_projections-1]+1) ){
                num_sequential_projections++;
            }

            int from_id = indices[p];
            int size = projection_res_x*projection_res_y;

            cudaMemcpyAsync( projections_batch->get_data_ptr()+(p-from_projection)*size,
                             projections->get_data_ptr()+from_id*size,
                             size*num_sequential_projections*sizeof(float), cudaMemcpyHostToDevice, mainStream );

            CHECK_FOR_CUDA_ERROR();

            p += num_sequential_projections;
        }
    }
    // printf("First Batch From/To .... DONE\n");


    //
    // Iterate over batches
    //
    // printf("Iterate Batches .... \n");

    for( int batch = 0; batch < num_batches; batch++ ) {

        from_projection = batch * projections_per_batch;
        to_projection = (batch+1) * projections_per_batch;

        if (to_projection > num_projections_in_bin )
            to_projection = num_projections_in_bin;

        projections_in_batch = to_projection-from_projection;

        float* raw_angles = thrust::raw_pointer_cast(&angles_devVec[from_projection]);
        floatd2* raw_offsets = thrust::raw_pointer_cast(&offsets_devVec[from_projection]);
        floatd3* raw_mot_XYZ = thrust::raw_pointer_cast(&mot_XYZ_devVec[from_projection]);


        if( FBP )
        {

            // Apply cosine weighting : "SDD / sqrt(SDD*SDD + u*u + v*v)"
            // - with (u,v) positions given in metric units on a virtual detector at the origin
            //

            // printf("Apply Cosine Weights .... \n");
            *projections_batch *= *cosine_weights;
            // printf("Apply Cosine Weights .... DONE\n");

            // Redundancy correct
            // - for short scan mode
            //
            printf("About to Apply Short Scan Redundancy .... \n");


            if( short_scan )
            {
                if (ffs)
                {
                    printf("Skipped Short Scan Redundancy .... \n");
                }
                else
                {
                printf("Short Scan Redundancy .... \n");
                float delta = std::atan(ps_dims_in_mm[0]/(2.0f*SDD));
                redundancy_correct( projections_batch, raw_angles, delta );
                printf("Short Scan Redundancy .... DONE\n");
                }
            }

            // Apply frequency filter
            // - use zero padding to avoid the cyclic boundary conditions induced by the fft
            //
            // printf("Apply Freq Filter .... \n");

            std::vector<size_t> batch_dims = *projections_batch->get_dimensions();
            uint64d3 pad_dims(batch_dims[0]<<1, batch_dims[1], batch_dims[2]);
            boost::shared_ptr< cuNDArray<float> > padded_projections = pad<float,3>( pad_dims, projections_batch );
            boost::shared_ptr< cuNDArray<complext<float> > > complex_projections = cb_fft( padded_projections.get() );
            *complex_projections *= *frequency_filter;
            cb_ifft( complex_projections.get(), padded_projections.get() );
            uint64d3 crop_offsets(batch_dims[0]>>1, 0, 0);
            crop<float,3>( crop_offsets, padded_projections.get(), projections_batch );
            // printf("Apply Freq Filter .... DONE\n");

            // Apply offset correction
            // - for half fan mode, sag correction etc.
            //
            if (use_offset_correction)
            {
                // printf("Apply Offset Correction .... \n");
                offset_correct( projections_batch, raw_offsets, ps_dims_in_mm, SAD, SDD );
                // printf("Apply Offset Correction .... DONE\n");
            }


        } else if (use_offset_correction)
        {
            printf("Not doing FBP. Using Offset_SQRT correction .... \n");
            offset_correct_sqrt( projections_batch, raw_offsets, ps_dims_in_mm, SAD, SDD );
        }

        // Build array for input texture
        //
        // printf("Build Cuda Texture .... \n");
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
        cudaExtent extent;
        extent.width = projection_res_x;
        extent.height = projection_res_y;
        extent.depth = projections_in_batch;

        cudaArray *projections_array;
        cudaMalloc3DArray( &projections_array, &channelDesc, extent, cudaArrayLayered );
        CHECK_FOR_CUDA_ERROR();

        cudaMemcpy3DParms cpy_params = {0};
        cpy_params.extent = extent;
        cpy_params.dstArray = projections_array;
        cpy_params.kind = cudaMemcpyDeviceToDevice;
        cpy_params.srcPtr =
                make_cudaPitchedPtr( (void*)projections_batch->get_data_ptr(), projection_res_x*sizeof(float),
                                     projection_res_x, projection_res_y );
        cudaMemcpy3DAsync( &cpy_params, mainStream );
        CHECK_FOR_CUDA_ERROR();

        cudaBindTextureToArray( projections_tex, projections_array, channelDesc );
        CHECK_FOR_CUDA_ERROR();
        // printf("Build Cuda Texture .... DONE\n");


        // Upload projections for the next batch
        // - to enable streaming
        //
        // printf("Upload next batch .... \n");

        if( batch < num_batches-1 ){ // for using multiple streams to hide the cost of the uploads

            int from_projection_next = (batch+1) * projections_per_batch;
            int to_projection_next = (batch+2) * projections_per_batch;

            if (to_projection_next > num_projections_in_bin )
                to_projection_next = num_projections_in_bin;

            int projections_in_batch_next = to_projection_next-from_projection_next;

            // printf("batch: %03i, handling projections: %03i - %03i, angles: %.2f - %.2f\n",
            //	 batch+1, from_projection_next, to_projection_next-1, angles[from_projection_next], angles[to_projection_next-1]);

            // Allocate device memory for projections and upload
            //

            dims_next.clear();
            dims_next.push_back(projection_res_x);
            dims_next.push_back(projection_res_y);
            dims_next.push_back(projections_in_batch_next);

            cuNDArray<float> projections_batch_next(&dims, projections_DevPtr2);

            // Upload projections adhering to the binning.
            // Be sure to copy sequentially numbered projections in one copy operation.
            //

            int p = from_projection_next;

            while( p<to_projection_next ) {

                int num_sequential_projections = 1;
                while( p+num_sequential_projections < to_projection_next &&
                       indices[p+num_sequential_projections]==(indices[p+num_sequential_projections-1]+1) ){
                    num_sequential_projections++;
                }

                int from_id = indices[p];
                int size = projection_res_x*projection_res_y;

                cudaMemcpyAsync( projections_batch_next.get_data_ptr()+(p-from_projection_next)*size,
                                 projections->get_data_ptr()+from_id*size,
                                 size*num_sequential_projections*sizeof(float), cudaMemcpyHostToDevice, indyStream );

                CHECK_FOR_CUDA_ERROR();

                p += num_sequential_projections;
            }
        }
        // printf("Upload next batch .... DONE\n");


        // Define dimensions of grid/blocks.
        //

        dim3 dimBlock, dimGrid;
        setup_grid( matrix_size_x*matrix_size_y*matrix_size_z, &dimBlock, &dimGrid );

        // Invoke kernel
        //

        // printf("Invoke Kernel .... \n");
        cudaFuncSetCacheConfig(conebeam_backwards_projection_kernel<FBP>, cudaFuncCachePreferL1);

        conebeam_backwards_projection_cyl_kernel<FBP><<< dimGrid, dimBlock, 0, mainStream >>>
             ( image_device->get_data_ptr(), raw_angles, raw_offsets, raw_mot_XYZ,
               is_dims_in_pixels, is_dims_in_mm, ps_dims_in_pixels, ps_dims_in_mm,
               projections_in_batch, num_projections_in_bin, SDD, SAD, (batch==0) ? accumulate : true, ffs);

        CHECK_FOR_CUDA_ERROR();
        // printf("Invoke Kernel .... DONE\n");


        // Cleanup
        //
        // printf("Cleanup .... \n");


        cudaUnbindTexture(projections_tex);
        cudaFreeArray(projections_array);
        CHECK_FOR_CUDA_ERROR();

        std::swap(projections_DevPtr, projections_DevPtr2);
        std::swap(mainStream, indyStream);

        delete projections_batch;
        if( batch < num_batches-1 )
            projections_batch = new cuNDArray<float>(&dims_next, projections_DevPtr);
        // printf("Cleanup .... DONE\n");

    }

    // Copy result from device to host
    //

    cudaMemcpy( image->get_data_ptr(), image_device->get_data_ptr(),
                num_image_elements*sizeof(float), cudaMemcpyDeviceToHost );

    CHECK_FOR_CUDA_ERROR();

    cudaFree(projections_DevPtr2);
    cudaFree(projections_DevPtr);
    CUDA_CALL(cudaStreamDestroy(indyStream));
    CUDA_CALL(cudaStreamDestroy(mainStream));
    CHECK_FOR_CUDA_ERROR();
}

// Template instantiations
//

template void conebeam_backwards_projection<false>
( hoCuNDArray<float>*, hoCuNDArray<float>*, std::vector<float>, std::vector<floatd2>, std::vector<unsigned int>,
int, intd3, floatd3, floatd2, float, float, bool, bool, bool, cuNDArray<float>*, cuNDArray<float>* );

template void conebeam_backwards_projection<true>
( hoCuNDArray<float>*, hoCuNDArray<float>*, std::vector<float>, std::vector<floatd2>, std::vector<unsigned int>,
int, intd3, floatd3, floatd2, float, float, bool, bool, bool, cuNDArray<float>*, cuNDArray<float>* );


template void conebeam_backwards_projection_cyl<false>
( hoCuNDArray<float>*, hoCuNDArray<float>*, std::vector<float>, std::vector<floatd2>, std::vector<unsigned int>,
int, intd3, floatd3, floatd2, float, float, bool, bool, bool, std::vector<floatd3>, bool, floatd3, cuNDArray<float>*, cuNDArray<float>*);

template void conebeam_backwards_projection_cyl<true>
( hoCuNDArray<float>*, hoCuNDArray<float>*, std::vector<float>, std::vector<floatd2>, std::vector<unsigned int>,
int, intd3, floatd3, floatd2, float, float, bool, bool, bool, std::vector<floatd3>, bool, floatd3, cuNDArray<float>*, cuNDArray<float>*);
}
