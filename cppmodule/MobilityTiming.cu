/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2014 The Regents of
the University of Michigan All rights reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Maintainer: joaander
// Modified by Gang Wang
// Modified by Andrew Fiore


#include "Mobility.cuh"
#include "Helper.cuh"
#include "saruprngCUDA.h"
#include <stdio.h>
#include "TextureTools.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <thrust/version.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

//! command to convert floats or doubles to integers
#ifdef SINGLE_PRECISION
#define __scalar2int_rd __float2int_rd
#else
#define __scalar2int_rd __double2int_rd
#endif


/*! \file Stokes.cu
    \brief Defines GPU kernel code for integration considering hydrodynamic interactions on the GPU. Used by Stokes.cc.
*/

//! Shared memory array for gpu_stokes_step_one_kernel()
// extern __shared__ Scalar s_gammas[];
// We will use diameter dependent gamma in the future.

//! Shared memory array for partial sum of dot product kernel
extern __shared__ Scalar partial_sum[];
extern __shared__ Scalar4 shared_Fpos[];

//! Texture for reading table values
scalar4_tex_t tables1_tex;
//! Texture for reading particle positions
scalar4_tex_t pos_tex;

// Define addition of float4
inline __host__ __device__ float4 operator+(float4 a, float4 b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}

/*!
	Compute wave space part of Mobility ( Same Size Particles, timing )

*/
/*! \param d_pos            positions of the particles, actually they are fetched on texture memory
    \param d_vel            particle velocity
    \param d_net_force      net forces on the particles
    \param group_size       size of the group, i.e. number of particles
    \param d_group_members  index array to global HOOMD tag on each particle
    \param box              array containing box dimensions
    \param xi               Ewald splitting parameter
    \param eta              Spectral splitting parameter
    \param ewald_cut        Cut-off distance for real-space interaction
    \param ewald_dr         Distance spacing using in computing the pre-tabulated tables
    \param ewald_n          Number of entries in the Ewald tables
    \param d_ewaldC         Pre-tabulated form of the real-space Ewald sum for the Velocity-Force coupling
    \param d_gridX          x-component of force moments projected onto grid
    \param d_gridY          y-component of force moments projected onto grid
    \param d_gridZ          z-component of force moments projected onto grid
    \param d_gridk          wave vector and scaling factor associated with each reciprocal grid node
    \param plan             Plan for cufft
    \param Nx               Number of grid/FFT nodes in x-direction
    \param Ny               Number of grid/FFT nodes in y-direction
    \param Nz               Number of grid/FFT nodes in z-direction
    \param d_n_neigh        list containing number of neighbors for each particle
    \param d_nlist          list containing neighbors of each particle
    \param nli              index into nlist
    \param NxNyNz           total number of grid/FFT nodes
    \param grid             block grid to use when launching kernels
    \param threads          number of threads per block for kernels
    \param gridBlockSize    number of threads per block
    \param gridNBlock       number of blocks
    \param P                number of nodes in support of each gaussian for k-space sum
    \param gridh            distance between grid nodes
    \param d_diameter       array of particle diameters
*/
void gpu_stokes_Mwave_Timing_wrap( Scalar4 *d_pos,
                            Scalar4 *d_vel,
                            Scalar4 *d_net_force,
			    unsigned int *d_group_members,
			    unsigned int group_size,
                            const BoxDim& box,
			    Scalar xi,
			    Scalar eta,
			    Scalar4 *d_gridk,
			    CUFFTCOMPLEX *d_gridX,
			    CUFFTCOMPLEX *d_gridY,
			    CUFFTCOMPLEX *d_gridZ,
			    cufftHandle plan,
			    const int Nx,
			    const int Ny,
			    const int Nz,
			    unsigned int NxNyNz,
			    dim3 grid,
			    dim3 threads,
			    int gridBlockSize,
			    int gridNBlock,
			    const int P,
			    Scalar3 gridh,
			    const Scalar *d_diameter ){
    
    // Stuff for spreading and contraction
    dim3 Cgrid( group_size, 1, 1);
    dim3 Cthreads(P, P, P);
    
    Scalar quadW = gridh.x * gridh.y * gridh.z;
    Scalar xisq = xi * xi;
    Scalar prefac = ( 2.0 * xisq / 3.1415926536 / eta ) * sqrtf( 2.0 * xisq / 3.1415926536 / eta );
    Scalar expfac = 2.0 * xisq / eta;
    
    cudaEvent_t start1, start2, start3, start4, start5;
    cudaEvent_t stop1, stop2, stop3, stop4, stop5;
    cudaEventCreate( &start1 );
    cudaEventCreate( &start2 );
    cudaEventCreate( &start3 );
    cudaEventCreate( &start4 );
    cudaEventCreate( &start5 );
    cudaEventCreate( &stop1 );
    cudaEventCreate( &stop2 );
    cudaEventCreate( &stop3 );
    cudaEventCreate( &stop4 );
    cudaEventCreate( &stop5 );
    
    cudaEventRecord( start1 );
    //
    // Reset the grid ( remove any previously distributed forces )
    gpu_stokes_ZeroGrid_kernel<<<gridNBlock,gridBlockSize>>>(d_gridX,NxNyNz);
    gpu_stokes_ZeroGrid_kernel<<<gridNBlock,gridBlockSize>>>(d_gridY,NxNyNz);
    gpu_stokes_ZeroGrid_kernel<<<gridNBlock,gridBlockSize>>>(d_gridZ,NxNyNz);
    //
    // Spread forces onto grid
    gpu_stokes_Spread_kernel<<<Cgrid, Cthreads>>>( d_pos, d_diameter, d_net_force, d_gridX, d_gridY, d_gridZ, group_size, Nx, Ny, Nz, d_group_members, box, P, gridh, xi, eta, prefac, expfac );
    //
    cudaEventRecord( stop1 );
    cudaEventSynchronize( stop1 );

    cudaEventRecord( start2 );
    //
    // Perform FFT on gridded forces
    cufftExecC2C(plan, d_gridX, d_gridX, CUFFT_FORWARD);
    cufftExecC2C(plan, d_gridY, d_gridY, CUFFT_FORWARD);
    cufftExecC2C(plan, d_gridZ, d_gridZ, CUFFT_FORWARD);
    //
    cudaEventRecord( stop2 );
    cudaEventSynchronize( stop2 );

    cudaEventRecord( start3 );
    //
    // Apply wave space scaling to FFT'd forces
    gpu_stokes_Green_kernel<<<gridNBlock,gridBlockSize>>>( d_gridX, d_gridY, d_gridZ, d_gridk, NxNyNz);
    //
    cudaEventRecord( stop3 );
    cudaEventSynchronize( stop3 );

    cudaEventRecord( start4 );
    //
    // Return rescaled forces to real space
    cufftExecC2C(plan, d_gridX, d_gridX, CUFFT_INVERSE);
    cufftExecC2C(plan, d_gridY, d_gridY, CUFFT_INVERSE);
    cufftExecC2C(plan, d_gridZ, d_gridZ, CUFFT_INVERSE);
    //
    cudaEventRecord( stop4 );
    cudaEventSynchronize( stop4 );
    
    cudaEventRecord( start5 );
    //
    // Evaluate contribution of grid velocities at particle centers
    gpu_stokes_Contract_kernel<<<Cgrid, Cthreads, (P*P*P+1)*sizeof(float3)>>>( d_pos, d_vel, d_gridX, d_gridY, d_gridZ, group_size, Nx, Ny, Nz, xi, eta, d_group_members, box, P, gridh, d_diameter, quadW*prefac, expfac );
    //
    cudaEventRecord( stop5 );
    cudaEventSynchronize( stop5 );


    float t1=0,t2=0,t3=0,t4=0,t5=0;

    cudaEventElapsedTime(&t1, start1, stop1);
    cudaEventElapsedTime(&t2, start2, stop2);
    cudaEventElapsedTime(&t3, start3, stop3);
    cudaEventElapsedTime(&t4, start4, stop4);
    cudaEventElapsedTime(&t5, start5, stop5);

    printf("Mobility Spread Time: %f \n", t1);
    printf("Mobility fFFT Time: %f \n", t2);
    printf("Mobility WSmult Time: %f \n", t3);
    printf("Mobility iFFT Time: %f \n", t4);
    printf("Mobility Contract Time: %f \n", t5);

    cudaEventDestroy( start1 );
    cudaEventDestroy( start2 );
    cudaEventDestroy( start3 );
    cudaEventDestroy( start4 );
    cudaEventDestroy( start5 );
    cudaEventDestroy( stop1 );
    cudaEventDestroy( stop2 );
    cudaEventDestroy( stop3 );
    cudaEventDestroy( stop4 );
    cudaEventDestroy( stop5 );
 
}


/*!
	Wrap all the functions to compute U = M * F ( SAME SIZE PARTICLES ) -- Version with timing involved
	Drive GPU kernel functions
	\param d_vel array of particle velocities
	\param d_net_force array of net forces

	d_vel = M * d_net_force

*/
/*! \param d_pos            positions of the particles, actually they are fetched on texture memory
    \param d_vel            particle velocity
    \param d_net_force      net forces on the particles
    \param group_size       size of the group, i.e. number of particles
    \param d_group_members  index array to global HOOMD tag on each particle
    \param box              array containing box dimensions
    \param xi               Ewald splitting parameter
    \param eta              Spectral splitting parameter
    \param ewald_cut        Cut-off distance for real-space interaction
    \param ewald_dr         Distance spacing using in computing the pre-tabulated tables
    \param ewald_n          Number of entries in the Ewald tables
    \param d_ewaldC         Pre-tabulated form of the real-space Ewald sum for the Velocity-Force coupling
    \param d_gridX          x-component of force moments projected onto grid
    \param d_gridY          y-component of force moments projected onto grid
    \param d_gridZ          z-component of force moments projected onto grid
    \param d_gridk          wave vector and scaling factor associated with each reciprocal grid node
    \param plan             Plan for cufft
    \param Nx               Number of grid/FFT nodes in x-direction
    \param Ny               Number of grid/FFT nodes in y-direction
    \param Nz               Number of grid/FFT nodes in z-direction
    \param d_n_neigh        list containing number of neighbors for each particle
    \param d_nlist          list containing neighbors of each particle
    \param nli              index into nlist
    \param NxNyNz           total number of grid/FFT nodes
    \param grid             block grid to use when launching kernels
    \param threads          number of threads per block for kernels
    \param gridBlockSize    number of threads per block
    \param gridNBlock       number of blocks
    \param P                number of nodes in support of each gaussian for k-space sum
    \param gridh            distance between grid nodes
    \param d_diameter       array of particle diameters
*/
void gpu_stokes_Mobility_Timing_wrap( Scalar4 *d_pos,
                               Scalar4 *d_vel,
                               Scalar4 *d_net_force,
			       unsigned int *d_group_members,
			       unsigned int group_size,
                               const BoxDim& box,
			       Scalar xi,
			       Scalar eta,
			       Scalar ewald_cut,
			       Scalar ewald_dr,
			       int ewald_n,
			       Scalar4 *d_ewaldC1, 
			       Scalar self,
			       Scalar4 *d_gridk,
			       CUFFTCOMPLEX *d_gridX,
			       CUFFTCOMPLEX *d_gridY,
			       CUFFTCOMPLEX *d_gridZ,
			       cufftHandle plan,
			       const int Nx,
			       const int Ny,
			       const int Nz,
			       const unsigned int *d_n_neigh,
                               const unsigned int *d_nlist,
                               const unsigned int *d_headlist,
			       unsigned int NxNyNz,
			       dim3 grid,
			       dim3 threads,
			       int gridBlockSize,
			       int gridNBlock,
			       const int P,
			       Scalar3 gridh,
			       const Scalar *d_diameter ){

    // Real and wave space velocity
    Scalar4 *d_vel1, *d_vel2;
    cudaMalloc( &d_vel1, group_size*sizeof(Scalar4) );
    cudaMalloc( &d_vel2, group_size*sizeof(Scalar4) );

    // Add the wave space contribution to the velocity
    gpu_stokes_Mwave_Timing_wrap( d_pos, d_vel1, d_net_force, d_group_members, group_size, box, xi, eta, d_gridk, d_gridX, d_gridY, d_gridZ, plan, Nx, Ny, Nz, NxNyNz, grid, threads, gridBlockSize, gridNBlock, P, gridh, d_diameter );

    cudaEvent_t start1, stop1;
    cudaEventCreate( &start1 );
    cudaEventCreate( &stop1 );
    cudaEventRecord( start1 );
    //
    // Add the real space contribution to the velocity
    //
    // Real space calculation takes care of self contributions
    gpu_stokes_Mreal_kernel<<<grid, threads>>>(d_pos, d_vel2, d_net_force, group_size, xi, d_ewaldC1, self, ewald_cut, ewald_n, ewald_dr, d_group_members, box, d_n_neigh, d_nlist, d_headlist, d_diameter );
    //
    cudaEventRecord( stop1 );
    cudaEventSynchronize( stop1 );

    float t1=0;

    cudaEventElapsedTime(&t1, start1, stop1);

    printf("Mobility RS Time: %f \n", t1);

    // Add real and wave space parts together
    gpu_stokes_LinearCombination_kernel<<<grid, threads>>>(d_vel1, d_vel2, d_vel, 1.0, 1.0, group_size, d_group_members);

    // Free memory
    cudaFree(d_vel1);
    cudaFree(d_vel2);

    cudaEventDestroy( start1 );
    cudaEventDestroy( stop1 );
 
}

