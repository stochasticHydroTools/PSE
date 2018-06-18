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

//! Spread particle quantities to the grid ( ALL PARTICLES SAME SIZE ) -- give one block per particle
/*! \param d_pos            positions of the particles, actually they are fetched on texture memory
    \param d_diameter       array of particle diameters
    \param d_net_force      net forces on the particles
    \param gridX            x-component of force moments projected onto grid
    \param gridY            y-component of force moments projected onto grid
    \param gridZ            z-component of force moments projected onto grid
    \param group_size       size of the group, i.e. number of particles
    \param Nx               number of grid nodes in x direction
    \param Ny               number of grid nodes in y direction
    \param Nz               number of grid nodes in z direction
    \param d_group_members  index array to global HOOMD tag on each particle
    \param box              array containing box dimensions
    \param P                number of grid nodes in support of spreading Gaussians
    \param gridh            space between grid nodes in each dimension
    \param xi               Ewald splitting parameter
    \param eta              Spectral splitting parameter
*/
__global__
void gpu_stokes_Spread_kernel( 	Scalar4 *d_pos,
			    	const Scalar  *d_diameter,
			    	Scalar4 *d_net_force,
			    	CUFFTCOMPLEX *gridX,
			    	CUFFTCOMPLEX *gridY,
			    	CUFFTCOMPLEX *gridZ,
			    	int group_size,
			    	int Nx,
			    	int Ny,
			    	int Nz,
			    	unsigned int *d_group_members,
			    	BoxDim box,
			    	const int P,
			    	Scalar3 gridh,
			    	Scalar xi,
			    	Scalar eta,
				Scalar prefac,
				Scalar expfac ) {

	__shared__ float3 shared[2]; // 16 kb max
	
	float3 *force_shared = shared;
	float3 *pos_shared = &shared[1];
	
	int group_idx = blockIdx.x;
	int thread_offset = threadIdx.z + threadIdx.y * blockDim.z + threadIdx.x * blockDim.z*blockDim.y;
	
	// Global particle ID
	unsigned int idx = d_group_members[group_idx];
	
	// Initialize shared memory and get particle position
	if ( thread_offset == 0 ){
		Scalar4 tpos = texFetchScalar4(d_pos, pos_tex, idx);
		pos_shared[0].x = tpos.x; 
		pos_shared[0].y = tpos.y; 
		pos_shared[0].z = tpos.z;
		
		Scalar4 tforce = d_net_force[idx];
		force_shared[0].x = tforce.x;
		force_shared[0].y = tforce.y;
		force_shared[0].z = tforce.z;
	}
	__syncthreads();
	
	// Box dimension
	Scalar3 L = box.getL();
	Scalar3 Ld2 = L / 2.0;
	
	// Retrieve position from shared memory
	Scalar3 pos = pos_shared[0];
	Scalar3 force = force_shared[0];
	
	// Fractional position within box 
	Scalar3 pos_frac = box.makeFraction(pos);
	
	pos_frac.x *= (Scalar)Nx;
	pos_frac.y *= (Scalar)Ny;
	pos_frac.z *= (Scalar)Nz;
	
	// Grid index of floor of fractional position
	int x = int( pos_frac.x );
	int y = int( pos_frac.y );
	int z = int( pos_frac.z );

	// Amount of work needed for each thread to cover support
	int3 n, t;
        n.x = ( P + blockDim.x - 1 ) / blockDim.x; // ceiling
        n.y = ( P + blockDim.y - 1 ) / blockDim.y;
        n.z = ( P + blockDim.z - 1 ) / blockDim.z;

	// Grid point associated with current thread
	int Pd2 = P/2; // integer division does floor

	for ( int ii = 0; ii < n.x; ++ii ){

		t.x = threadIdx.x + ii*blockDim.x;

		for ( int jj = 0; jj < n.y; ++jj ){

			t.y = threadIdx.y + jj*blockDim.y;

			for ( int kk = 0; kk < n.z; ++kk ){

				t.z = threadIdx.z + kk*blockDim.z;

				if ( ( t.x < P ) && ( t.y < P ) && ( t.z < P ) ){

					int x_inp = x + t.x - Pd2 + 1 - (P % 2) * ( pos_frac.x - Scalar( x ) < 0.5  );
					int y_inp = y + t.y - Pd2 + 1 - (P % 2) * ( pos_frac.y - Scalar( y ) < 0.5  );
					int z_inp = z + t.z - Pd2 + 1 - (P % 2) * ( pos_frac.z - Scalar( z ) < 0.5  );

					x_inp = (x_inp<0) ? x_inp+Nx : ( (x_inp>Nx-1) ? x_inp-Nx : x_inp );
					y_inp = (y_inp<0) ? y_inp+Ny : ( (y_inp>Ny-1) ? y_inp-Ny : y_inp );
					z_inp = (z_inp<0) ? z_inp+Nz : ( (z_inp>Nz-1) ? z_inp-Nz : z_inp );
					
					Scalar3 pos_grid;
					pos_grid.x = gridh.x*x_inp - Ld2.x;
					pos_grid.y = gridh.y*y_inp - Ld2.y;
					pos_grid.z = gridh.z*z_inp - Ld2.z;

					pos_grid.x = pos_grid.x + box.getTiltFactorXY() * pos_grid.y; // shear lattic position
					
					int grid_idx = x_inp * Ny * Nz + y_inp * Nz + z_inp;
					
					// Distance from particle to grid node
					Scalar3 r = pos_grid - pos;
					r = box.minImage(r);
					Scalar rsq = r.x*r.x + r.y*r.y + r.z*r.z;
					
					// Magnitude of the force contribution to the current grid node
					Scalar3 force_inp = prefac * expf( -expfac * rsq ) * force;
					
					// Add force to the grid
					atomicAdd( &(gridX[grid_idx].x), force_inp.x);
					atomicAdd( &(gridY[grid_idx].x), force_inp.y);
					atomicAdd( &(gridZ[grid_idx].x), force_inp.z);
				}
			}//kk
		}//jj
	}//ii

}

//! Compute the velocity from the force moments on the grid (Same Size Particles)
/*! \param gridX            x-component of force moments projected onto grid
    \param gridY            y-component of force moments projected onto grid
    \param gridZ            z-component of force moments projected onto grid
    \param gridk            wave vector and scaling factor associated with each reciprocal grid node
    \param NxNyNz           total number of grid nodes
*/
__global__
void gpu_stokes_Green_kernel(CUFFTCOMPLEX *gridX, CUFFTCOMPLEX *gridY, CUFFTCOMPLEX *gridZ, Scalar4 *gridk, unsigned int NxNyNz) {

	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	if ( tid < NxNyNz ) {
	
	  // Read the FFT force from global memory
	  Scalar2 fX = gridX[tid];  
	  Scalar2 fY = gridY[tid];
	  Scalar2 fZ = gridZ[tid];
	
	  // Current wave-space vector 
	  Scalar4 tk = gridk[tid];
	  Scalar ksq = tk.x*tk.x + tk.y*tk.y + tk.z*tk.z;
	  Scalar k = sqrtf( ksq );
	
	  // Dot product of the wave-vector with the force 
	  Scalar2 kdF = (tid==0) ? make_scalar2(0.0,0.0) : make_scalar2( ( tk.x*fX.x + tk.y*fY.x + tk.z*fZ.x ) / ksq,  ( tk.x*fX.y + tk.y*fY.y + tk.z*fZ.y ) / ksq );
	
	  // Scaling factor
	  Scalar B = (tid==0) ? 0.0 : tk.w * ( sinf( k ) / k ) * ( sinf( k ) / k );
	
	  //Write the velocity to global memory
	  gridX[tid] = make_scalar2( ( fX.x - tk.x * kdF.x ) * B, ( fX.y - tk.x * kdF.y ) * B );
	  gridY[tid] = make_scalar2( ( fY.x - tk.y * kdF.x ) * B, ( fY.y - tk.y * kdF.y ) * B );
	  gridZ[tid] = make_scalar2( ( fZ.x - tk.z * kdF.x ) * B, ( fZ.y - tk.z * kdF.y ) * B );
	
	
	}
}

//! Add velocity from grid to particles ( Same Size Particles, Block Per Particle (support) )
/*! \param d_pos            positions of the particles, actually they are fetched on texture memory
    \param d_net_force      net forces on the particles
    \param d_vel            particle velocity
    \param gridX            x-component of force moments projected onto grid
    \param gridY            y-component of force moments projected onto grid
    \param gridZ            z-component of force moments projected onto grid
    \param group_size       size of the group, i.e. number of particles
    \param Nx               number of grid nodes in x direction
    \param Ny               number of grid nodes in y direction
    \param Nz               number of grid nodes in z direction
    \param xi               Ewald splitting parameter
    \param eta              Spectral splitting parameter
    \param d_group_members  index array to global HOOMD tag on each particle
    \param box              array containing box dimensions
    \param P                number of grid nodes in support of spreading Gaussians
    \param gridh            space between grid nodes in each dimension
    \param d_diameter       array of particle diameters
*/
__global__
void gpu_stokes_Contract_kernel( 	Scalar4 *d_pos,
				 	Scalar4 *d_vel,
				 	CUFFTCOMPLEX *gridX,
				 	CUFFTCOMPLEX *gridY,
				 	CUFFTCOMPLEX *gridZ,
				 	int group_size,
				 	int Nx,
				 	int Ny,
				 	int Nz,
				 	Scalar xi,
				 	Scalar eta,
				 	unsigned int *d_group_members,
				 	BoxDim box,
				 	const int P,
				 	Scalar3 gridh,
				 	const Scalar *d_diameter,
				 	Scalar prefac,
				 	Scalar expfac ){

	extern __shared__ float3 shared[];
	
	float3 *velocity = shared;
	float3 *pos_shared = &shared[blockDim.x*blockDim.y*blockDim.z];
	
	int group_idx = blockIdx.x;
	int thread_offset = threadIdx.z + threadIdx.y * blockDim.z + threadIdx.x * blockDim.z*blockDim.y;
	int block_size = blockDim.x * blockDim.y * blockDim.z;
	
	// Global particle ID
	unsigned int idx = d_group_members[group_idx];
	
	// Initialize shared memory and get particle position
	velocity[thread_offset] = make_scalar3(0.0,0.0,0.0);
	if ( thread_offset == 0 ){
		Scalar4 tpos = texFetchScalar4(d_pos, pos_tex, idx);
		pos_shared[0] = make_scalar3( tpos.x, tpos.y, tpos.z ); 
	}
	__syncthreads();
	
	// Box dimension
	Scalar3 L = box.getL();
	Scalar3 Ld2 = L / 2.0;
	
	// Retrieve position from shared memory
	Scalar3 pos = pos_shared[0];
	
	// Fractional position within box 
	Scalar3 pos_frac = box.makeFraction(pos);
	
	pos_frac.x *= (Scalar)Nx;
	pos_frac.y *= (Scalar)Ny;
	pos_frac.z *= (Scalar)Nz;
	
	int x = int( pos_frac.x );
	int y = int( pos_frac.y );
	int z = int( pos_frac.z );
	
	int3 n, t;
        n.x = ( P + blockDim.x - 1 ) / blockDim.x; // ceiling
        n.y = ( P + blockDim.y - 1 ) / blockDim.y;
        n.z = ( P + blockDim.z - 1 ) / blockDim.z;
 
	// Grid point associated with current thread
	int Pd2 = P / 2; // integer division does floor
	
	for ( int ii = 0; ii < n.x; ++ii ){

		t.x = threadIdx.x + ii*blockDim.x;

		for ( int jj = 0; jj < n.y; ++jj ){

			t.y = threadIdx.y + jj*blockDim.y;

			for ( int kk = 0; kk < n.z; ++kk ){

				t.z = threadIdx.z + kk*blockDim.z;

				if( ( t.x < P ) && ( t.y < P ) && ( t.z < P ) ){

					int x_inp = x + t.x - Pd2 + 1 - (P % 2) * ( pos_frac.x - Scalar( x ) < 0.5  );
					int y_inp = y + t.y - Pd2 + 1 - (P % 2) * ( pos_frac.y - Scalar( y ) < 0.5  );
					int z_inp = z + t.z - Pd2 + 1 - (P % 2) * ( pos_frac.z - Scalar( z ) < 0.5  );
					
					x_inp = (x_inp<0) ? x_inp+Nx : ( (x_inp>Nx-1) ? x_inp-Nx : x_inp );
					y_inp = (y_inp<0) ? y_inp+Ny : ( (y_inp>Ny-1) ? y_inp-Ny : y_inp );
					z_inp = (z_inp<0) ? z_inp+Nz : ( (z_inp>Nz-1) ? z_inp-Nz : z_inp );
					
					Scalar3 pos_grid;
					pos_grid.x = gridh.x*x_inp - Ld2.x;
					pos_grid.y = gridh.y*y_inp - Ld2.y;
					pos_grid.z = gridh.z*z_inp - Ld2.z;

					pos_grid.x = pos_grid.x + box.getTiltFactorXY() * pos_grid.y; // shear lattic position
					
					int grid_idx = x_inp * Ny * Nz + y_inp * Nz + z_inp;
					
					// Distance from particle to grid node
					Scalar3 r = pos_grid - pos;
					r = box.minImage(r);
					Scalar rsq = r.x*r.x + r.y*r.y + r.z*r.z;
					
					// Spreading Factor
					Scalar Cfac = prefac * expf( -expfac * rsq );
					
					// Get velocity from reduction (THIS IS THE SLOW STEP):
					velocity[thread_offset] += Cfac * make_scalar3( gridX[grid_idx].x, gridY[grid_idx].x, gridZ[grid_idx].x );
				}
			}//kk
		}//jj
	}//ii

	int offs = block_size;
	int offs_prev; 
	while (offs > 1)
	{
	      offs_prev = offs; 
	      offs = ( offs + 1 ) / 2;
		__syncthreads();
	    	if (thread_offset + offs < offs_prev)
	        {
	        	velocity[thread_offset] += velocity[thread_offset + offs];
	        }
	    	
	}
	
	// Combine components of velocity
	if (thread_offset == 0){
		d_vel[idx] = make_scalar4(velocity[0].x, velocity[0].y, velocity[0].z, d_vel[idx].w);
	}
	
}

/*!
	Compute wave space part of Mobility ( Same Size Particles )

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
void gpu_stokes_Mwave_wrap( Scalar4 *d_pos,
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
    
	// Spreading and contraction stuff
	dim3 Cgrid( group_size, 1, 1);
	int B = ( P < 10 ) ? P : 10;
	dim3 Cthreads(B, B, B);

	Scalar quadW = gridh.x * gridh.y * gridh.z;
	Scalar xisq = xi * xi;
	Scalar prefac = ( 2.0 * xisq / 3.1415926536 / eta ) * sqrtf( 2.0 * xisq / 3.1415926536 / eta );
	Scalar expfac = 2.0 * xisq / eta;
	
	// Reset the grid ( remove any previously distributed forces )
	gpu_stokes_ZeroGrid_kernel<<<gridNBlock,gridBlockSize>>>(d_gridX,NxNyNz);
	gpu_stokes_ZeroGrid_kernel<<<gridNBlock,gridBlockSize>>>(d_gridY,NxNyNz);
	gpu_stokes_ZeroGrid_kernel<<<gridNBlock,gridBlockSize>>>(d_gridZ,NxNyNz);
	
	// Spread forces onto grid
	gpu_stokes_Spread_kernel<<<Cgrid, Cthreads>>>( d_pos, d_diameter, d_net_force, d_gridX, d_gridY, d_gridZ, group_size, Nx, Ny, Nz, d_group_members, box, P, gridh, xi, eta, prefac, expfac );
	
	// Perform FFT on gridded forces
	cufftExecC2C(plan, d_gridX, d_gridX, CUFFT_FORWARD);
	cufftExecC2C(plan, d_gridY, d_gridY, CUFFT_FORWARD);
	cufftExecC2C(plan, d_gridZ, d_gridZ, CUFFT_FORWARD);
	
	// Apply wave space scaling to FFT'd forces
	gpu_stokes_Green_kernel<<<gridNBlock,gridBlockSize>>>( d_gridX, d_gridY, d_gridZ, d_gridk, NxNyNz);
	
	// Return rescaled forces to real space
	cufftExecC2C(plan, d_gridX, d_gridX, CUFFT_INVERSE);
	cufftExecC2C(plan, d_gridY, d_gridY, CUFFT_INVERSE);
	cufftExecC2C(plan, d_gridZ, d_gridZ, CUFFT_INVERSE);
	
	// Evaluate contribution of grid velocities at particle centers
	gpu_stokes_Contract_kernel<<<Cgrid, Cthreads, (B*B*B+1)*sizeof(float3)>>>( d_pos, d_vel, d_gridX, d_gridY, d_gridZ, group_size, Nx, Ny, Nz, xi, eta, d_group_members, box, P, gridh, d_diameter, quadW*prefac, expfac );
 
}

// Add real space Ewald summation to velocity of each particle
// NLIST Method
/*! \param d_pos            positions of the particles, actually they are fetched on texture memory
    \param d_vel            particle velocity
    \param d_net_force      net forces on the particles
    \param group_size       size of the group, i.e. number of particles
    \param xi               Ewald splitting parameter
    \param d_ewaldC         Pre-tabulated form of the real-space Ewald sum for the Velocity-Force coupling
    \param ewald_cut        Cut-off distance for real-space interaction
    \param ewald_n          Number of entries in the Ewald tables
    \param ewald_dr         Distance spacing using in computing the pre-tabulated tables
    \param d_group_members  index array to global HOOMD tag on each particle
    \param box              array containing box dimensions
    \param d_n_neigh        list containing number of neighbors for each particle
    \param d_nlist          list containing neighbors of each particle
    \param nli              index into nlist
    \param d_diameter       array of particle diameters
*/
__global__
void gpu_stokes_Mreal_kernel( 	Scalar4 *d_pos,
			      	Scalar4 *d_vel,
			      	Scalar4 *d_net_force,
			      	int group_size,
			      	Scalar xi,
			      	Scalar4 *d_ewaldC1, 
			      	Scalar self,
			      	Scalar ewald_cut,
			      	int ewald_n,
			      	Scalar ewald_dr,
			      	unsigned int *d_group_members,
			      	BoxDim box,
			      	const unsigned int *d_n_neigh,
                              	const unsigned int *d_nlist,
                              	const unsigned int *d_headlist,
			      	const Scalar *d_diameter) {
 
	// Index for current thread 
	int group_idx = blockDim.x * blockIdx.x + threadIdx.x;
	
	// Initialize contribution to velocity
	Scalar4 u = make_scalar4( 0.0, 0.0, 0.0, 0.0 );
	
	if (group_idx < group_size) {
	  
		// Particle for this thread
		unsigned int idx = d_group_members[group_idx];
		
		// Number of neighbors for current particle
		unsigned int n_neigh = d_n_neigh[idx]; 
		unsigned int head_idx = d_headlist[idx];
		
		// Particle position and table ID
		Scalar4 posi = texFetchScalar4(d_pos, pos_tex, idx);
		
		// Self contribution
		Scalar4 F = d_net_force[idx];
		u = make_scalar4( self * F.x, self * F.y, self * F.z, 0.0 );
		
		// Calculate contribution to this particle from all neighbors
		unsigned int cur_j = 0;
		unsigned int next_j = d_nlist[head_idx];
		
		// Minimum and maximum distance for pair calculation
		Scalar mindistSq = ewald_dr * ewald_dr;
		Scalar maxdistSq = ewald_cut * ewald_cut;
		
		      for (int neigh_idx = 0; neigh_idx < n_neigh; neigh_idx++) {
		
			// Statement might be necessary for bug on older architectures?
			cur_j = next_j;
			next_j = d_nlist[head_idx + neigh_idx + 1];
		
			// Position and size of neighbor particle
			Scalar4 posj = texFetchScalar4(d_pos, pos_tex, cur_j);
		
			// Distance vector between current particle and neighbor
			Scalar3 r = make_scalar3( posi.x - posj.x, posi.y - posj.y, posi.z - posj.z );
			r = box.minImage(r);
			Scalar distSqr = dot(r,r);
		
			// Add neighbor contribution if it is within the real space cutoff radius
			if ( ( distSqr < maxdistSq ) && ( distSqr >= mindistSq ) ) {
		
				// Need distance 
				Scalar dist = sqrtf( distSqr );
				
				// Force on neighbor particle
				Scalar4 Fj = d_net_force[cur_j];
			
				// Fetch relevant elements from textured table for real space interaction
				int r_ind = __scalar2int_rd( ewald_n * ( dist - ewald_dr ) / ( ewald_cut - ewald_dr ) );
				int offset = r_ind;
		
				Scalar4 tewaldC1 = texFetchScalar4(d_ewaldC1, tables1_tex, offset);
		
				// Linear interpolation of table
				Scalar fac = dist / ewald_dr - r_ind - Scalar(1.0);
		
				Scalar Imrr = tewaldC1.x + ( tewaldC1.z - tewaldC1.x ) * fac;
				Scalar rr = tewaldC1.y + ( tewaldC1.w - tewaldC1.y ) * fac;
		
				// Update velocity
				Scalar rdotf = ( r.x*Fj.x + r.y*Fj.y + r.z*Fj.z ) / distSqr;
		
				u.x += Imrr * Fj.x + ( rr - Imrr ) * rdotf * r.x;
				u.y += Imrr * Fj.y + ( rr - Imrr ) * rdotf * r.y;
				u.z += Imrr * Fj.z + ( rr - Imrr ) * rdotf * r.z;
		
			}
		
		}
		
		// Write to output
		d_vel[idx] = u;
	
	}    
}



/*!
	Wrap all the functions to compute U = M * F ( SAME SIZE PARTICLES )
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
void gpu_stokes_Mobility_wrap( Scalar4 *d_pos,
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
	gpu_stokes_Mwave_wrap( d_pos, d_vel1, d_net_force, d_group_members, group_size, box, xi, eta, d_gridk, d_gridX, d_gridY, d_gridZ, plan, Nx, Ny, Nz, NxNyNz, grid, threads, gridBlockSize, gridNBlock, P, gridh, d_diameter );
	
	// Add the real space contribution to the velocity
	//
	// Real space calculation takes care of self contributions
	gpu_stokes_Mreal_kernel<<<grid, threads>>>(d_pos, d_vel2, d_net_force, group_size, xi, d_ewaldC1, self, ewald_cut, ewald_n, ewald_dr, d_group_members, box, d_n_neigh, d_nlist, d_headlist, d_diameter );
	
	// Add real and wave space parts together
	gpu_stokes_LinearCombination_kernel<<<grid, threads>>>(d_vel1, d_vel2, d_vel, 1.0, 1.0, group_size, d_group_members);
	
	// Free memory
	cudaFree(d_vel1);
	cudaFree(d_vel2);
 
}


