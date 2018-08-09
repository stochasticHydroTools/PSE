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


#include "Stokes.cuh"
#include "Mobility.cuh"
#include "Brownian.cuh"
#include "Helper.cuh"

#include "hoomd/Saru.h"
#include "hoomd/TextureTools.h"

#include <stdio.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

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

#ifndef __ERRCHK_CUH__
#define __ERRCHK_CUH__
//! Function to check for errors
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
/*!
    \param code   returned error code
    \param file   which file the error occured in
    \param line   which line error check was tripped
    \param abort  whether to kill code upon error trigger
*/
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#endif

/*! \file Stokes.cu
    \brief Defines GPU kernel code for integration considering hydrodynamic interactions on the GPU. Used by Stokes.cc.
*/


//! Shared memory array for partial sum of dot product kernel
extern __shared__ Scalar partial_sum[];
extern __shared__ Scalar4 shared_Fpos[];

//! Texture for reading table values
scalar4_tex_t tables1_tex;
//! Texture for reading particle positions
scalar4_tex_t pos_tex;

//! Takes the integration on a group of particles
/*! \param d_pos            array of particle positions
    \param d_vel            array of particle velocities
    \param d_delu1          first 4 components of gradient of particle velocity
    \param d_delu2          second 4 components of gradient of particle velocity
    \param d_accel          array of particle "accelerations" (This is an overdamped integrator, so accelerations don't have physical meaning)
    \param d_image          array of particle images
    \param d_group_members  Device array listing the indicies of the mebers of the group to integrate
    \param group_size       Number of members in the group
    \param box Box          dimensions for periodic boundary condition handling
    \param deltaT           timestep
    \param d_net_force      net force on each particle, only used to set "accelerations"

    This kernel must be executed with a 1D grid of any block size such that the number of threads is greater than or
    equal to the number of members in the group. The kernel's implementation simply reads one particle in each thread
    and updates that particle. (Not necessary true for Stokesian Dynamics simulation)

    <b>Performance notes:</b>
    Particle properties are read via the texture cache to optimize the bandwidth obtained with sparse groups. The writes
    in sparse groups will not be coalesced. However, because ParticleGroup sorts the index list the writes will be as
    contiguous as possible leading to fewer memory transactions on compute 1.3 hardware and more cache hits on Fermi. (Not sure about this..)
*/
extern "C" __global__
void gpu_stokes_step_one_kernel(
				Scalar4 *d_pos,
				Scalar4 *d_vel,
				Scalar3 *d_accel,
				int3 *d_image,
				unsigned int *d_group_members,
				unsigned int group_size,
				BoxDim box,
				Scalar deltaT,
				Scalar4 *d_net_force,
				Scalar shear_rate
				){

    // determine which particle this thread works on (MEM TRANSFER: 4 bytes)
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (group_idx < group_size){

        unsigned int idx = d_group_members[group_idx];

        // read the particle's posision (MEM TRANSFER: 16 bytes)
        Scalar4 postype = d_pos[idx];
        Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);

        // read the particle's velocity and acceleration (MEM TRANSFER: 32 bytes)
        Scalar4 velmass = d_vel[idx];
	Scalar mass = velmass.w;
        Scalar3 vel = make_scalar3(velmass.x, velmass.y, velmass.z);

	// Add the shear
        vel.x += shear_rate * pos.y;

	Scalar4 net_force = d_net_force[idx];
        Scalar3 accel = make_scalar3(net_force.x, net_force.y, net_force.z);

        // update the position
        Scalar3 dx = vel * deltaT;

        // FLOPS: 3
        pos += dx;

	accel = accel/mass;

        // read in the particle's image (MEM TRANSFER: 16 bytes)
        int3 image = d_image[idx];

        // fix the periodic boundary conditions (FLOPS: 15)
        box.wrap(pos, image);

        // write out the results (MEM_TRANSFER: 48 bytes)
	d_accel[idx] = accel;
        d_pos[idx] = make_scalar4(pos.x, pos.y, pos.z, postype.w);
        d_image[idx] = image;
        }
    }

/*! \param d_pos              array of particle positions
    \param d_vel              array of particle velocities
    \param d_accel            array of particle accelerations
    \param d_image            array of particle images
    \param d_group_members    Device array listing the indicies of the mebers of the group to integrate
    \param group_size         Number of members in the group ( i.e. number of particles to consider )
    \param box                Box dimensions for periodic boundary condition handling
    \param dt                 timestep
    \param block_size         optimum block size returned by an autotuner
    \param d_net_force        net force on the particles
    \param T                  temperature
    \param timestep           time step
    \param seed               seed for random number generation
    \param xi                 splitting coefficient for Ewald summation
    \param eta                Spectral splitting parameter
    \param P                  number of nodes in support of each gaussian for k-space sum
    \param ewald_cut          cut off radius for Ewald summation
    \param ewald_dr           discretization of look up tables
    \param ewald_n            number of elements in look up tables
    \param d_ewaldC           Ewald coefficients for real space sum
    \param d_gridk            reciprocal lattice vectors and parameters for Ewald reciprocal space sum
    \param d_gridX            x-component of force moment projection onto the grid
    \param d_gridY            y-component of force moment projection onto the grid
    \param d_gridZ            z-component of force moment projection onto the grid
    \param plan cudaFFT       plan
    \param Nx 		      number of grid nodes in the x-direction
    \param Ny                 number of grid nodes in the y-direction
    \param Nz                 number of grid nodes in the z-direction
    \param d_n_neigh          Number of neighbors for every particle
    \param d_nlist            Neighbor list of every particle, 2D array, can be accessed by nli
    \param nli                Index lookup helper for d_nlist
    \param cheb_an            Chebychev coefficients
    \param n_cheb             Order of Chebyshev approximation
    \param N_total            total number of particles ( should be same as group_size )
    \param gridh              Spacing between grid ndoes
    \param cheb_recompute     whether to recompute chebyshev approximation
    \param eig_recompute      whether to recompute eigenvalues of matrix approximation
    \param stored_eigenvalue  previous max eigenvalue
    \param cheb_error         error tolerance in chebyshev approximation
*/
cudaError_t gpu_stokes_step_one(
				Scalar4 *d_pos,
				Scalar4 *d_vel,
				Scalar3 *d_accel,
				int3 *d_image,
				unsigned int *d_group_members,
				unsigned int group_size,
				const BoxDim& box,
				Scalar dt,
				unsigned int block_size,
				Scalar4 *d_net_force,
				const Scalar T,
				const unsigned int timestep,
				const unsigned int seed,
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
				int& m_Lanczos,
				const unsigned int N_total,
				const int P,
				Scalar3 gridh,
				Scalar cheb_error,
				Scalar shear_rate
				){

	// Total number of grid points
	unsigned int NxNyNz = Nx*Ny*Nz;

	// setup the grid to run the kernel
	// block for particle calculation
	dim3 grid( (group_size/block_size) + 1, 1, 1);
	dim3 threads(block_size, 1, 1);
	
	// block for grid calculation
	int gridBlockSize = ( NxNyNz > block_size ) ? block_size : NxNyNz;
	int gridNBlock = ( NxNyNz + gridBlockSize - 1 ) / gridBlockSize ; 
	
	// Get the textured tables for real space Ewald sum tabulation
	tables1_tex.normalized = false; // Not normalized
	tables1_tex.filterMode = cudaFilterModeLinear; // Filter mode: floor of the index
	// One dimension, Read mode: ElementType(Get what we write)
	cudaBindTexture(0, tables1_tex, d_ewaldC1, sizeof(Scalar4) * (ewald_n+1)); // This was a bug in former versions!
	
	// Same for the positions and forces
	pos_tex.normalized = false; // Not normalized
	pos_tex.filterMode = cudaFilterModePoint; // Filter mode: floor of the index
	cudaBindTexture(0, pos_tex, d_pos, sizeof(Scalar4) * N_total);

	// Get sheared grid vectors
    	gpu_stokes_SetGridk_kernel<<<gridNBlock,gridBlockSize>>>(d_gridk,Nx,Ny,Nz,NxNyNz,box,xi,eta);

	// Do Mobility and Brownian Calculations (compute the velocity from the forces)
	gpu_stokes_CombinedMobilityBrownian_wrap(  	
							d_pos,
							d_net_force,
                                			d_group_members,
                                			group_size,
                                			box,
                                			dt,
			        			d_vel, // output
			        			T,
			        			timestep,
			        			seed,
			        			xi,
							eta,
							P,
			        			ewald_cut,
			        			ewald_dr,
			        			ewald_n,
			        			d_ewaldC1, 
			        			d_gridk,
			        			d_gridX,
			        			d_gridY,
			        			d_gridZ,
			        			plan,
			        			Nx,
			        			Ny,
			        			Nz,
			        			d_n_neigh,
                                			d_nlist,
                                			d_headlist,
			        			m_Lanczos,
			        			N_total,
			        			NxNyNz,
			        			grid,
			        			threads,
			        			gridBlockSize,
			        			gridNBlock,
							gridh,
			        			cheb_error,
							self );


	// Use forward Euler integration to move the particles according the velocity
	// computed from the Mobility and Brownian calculations
	gpu_stokes_step_one_kernel<<< grid, threads >>>(
							d_pos, 
							d_vel, 
							d_accel, 
							d_image, 
							d_group_members, 
							group_size, 
							box, 
							dt, 
							d_net_force, 
							shear_rate
							);

	// Quick error check
	gpuErrchk(cudaPeekAtLastError());
	
	// Cleanup
	cudaUnbindTexture(tables1_tex);
	cudaUnbindTexture(pos_tex);
	
	return cudaSuccess;
}
