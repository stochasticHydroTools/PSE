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


#include "Helper.cuh"
#include "saruprngCUDA.h"
#include <stdio.h>
#include "TextureTools.h"

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


/*! \file Helper.cu
    	\brief Helper functions to perform additions, dot products, etc., for Mobility and Brownian
*/

//! Shared memory array for partial sum of dot product kernel
extern __shared__ Scalar partial_sum[];

//! Zero out the force grid
/*! 
	\param grid the grid going to be zero out
   	\param NxNyNz dimension of the grid
*/
__global__
void gpu_stokes_ZeroGrid_kernel(CUFFTCOMPLEX *grid, unsigned int NxNyNz) {

  unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
 
  if ( tid < NxNyNz ) {
  
  	grid[tid] = make_scalar2( 0.0, 0.0 );  

  }
}

/*!
	Linear combination helper function
	C = a*A + b*B
	C can be A or B, so that A or B will be overwritten
	The fourth element of Scalar4 is not changed!

	\param d_a              input vector, A
	\param d_b              input vector, B
	\param d_c              output vector, C
	\param coeff_a          scaling factor for A, a
	\param coeff_b          scaling factor for B, b
	\param group_size       length of vectors
	\param d_group_members  index into vectors
*/
__global__
void gpu_stokes_LinearCombination_kernel(Scalar4 *d_a, Scalar4 *d_b, Scalar4 *d_c, Scalar coeff_a, Scalar coeff_b, unsigned int group_size, unsigned int *d_group_members){
	int group_idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (group_idx < group_size) {
		unsigned int idx = d_group_members[group_idx];
		Scalar4 A4 = d_a[idx];
		Scalar4 B4 = d_b[idx];
		Scalar3 A = make_scalar3(A4.x, A4.y, A4.z);
		Scalar3 B = make_scalar3(B4.x, B4.y, B4.z);
		A = coeff_a * A + coeff_b * B;
		d_c[idx] = make_scalar4(A.x, A.y, A.z, d_c[idx].w);
	}
}

/*!
	Dot product helper function: First step
	d_a .* d_b -> d_c -> Partial sum
	BlockDim of this kernel should be 2^n, which is 512. (Based on HOOMD ComputeThermoGPU class)
	
	\param d_a              first vector in dot product
	\param d_b              second vector in dot product
	\param dot_sum          partial dot product sum
	\param group_size       length of vectors a and b
        \param d_group_members  index into vectors
*/
__global__
void gpu_stokes_DotStepOne_kernel(Scalar4 *d_a, Scalar4 *d_b, Scalar *dot_sum, unsigned int group_size, unsigned int *d_group_members){
	int group_idx = blockDim.x * blockIdx.x + threadIdx.x;
	Scalar temp;

	if (group_idx < group_size) {

		unsigned int idx = d_group_members[group_idx];
		Scalar4 a4 = d_a[idx];
		Scalar4 b4 = d_b[idx];
		Scalar3 a = make_scalar3(a4.x, a4.y, a4.z);
		Scalar3 b = make_scalar3(b4.x, b4.y, b4.z);

		temp = dot(a,b); // Partial sum, each thread, shared memory

	}
	else {
		temp = 0;
	}

	partial_sum[threadIdx.x] = temp;

	__syncthreads();

	int offs = blockDim.x >> 1;

	while (offs > 0)
        {
        	if (threadIdx.x < offs)
            	{
            		partial_sum[threadIdx.x] += partial_sum[threadIdx.x + offs];
            	}
        	offs >>= 1;
        	__syncthreads();
        }

	if (threadIdx.x == 0){
		dot_sum[blockIdx.x] = partial_sum[0];
	}
}



/*!
	Dot product helper function: Second step
	Partial sum -> Final sum
	Only one block will be launched for this step

	\param dot_sum           partial sum from first dot product kernel
	\param num_partial_sums  length of dot_sum array

*/
__global__
void gpu_stokes_DotStepTwo_kernel(Scalar *dot_sum, unsigned int num_partial_sums){

	partial_sum[threadIdx.x] = 0.0;
	__syncthreads();
	for (unsigned int start = 0; start < num_partial_sums; start += blockDim.x)
       	{
        	if (start + threadIdx.x < num_partial_sums)
            	{
            		partial_sum[threadIdx.x] += dot_sum[start + threadIdx.x];
            	}
	}

	int offs = blockDim.x >> 1;
	while (offs > 0)
       	{
		__syncthreads();
            	if (threadIdx.x < offs)
                {
                	partial_sum[threadIdx.x] += partial_sum[threadIdx.x + offs];
                }
            	offs >>= 1;
            	
        }
	__syncthreads();
        if (threadIdx.x == 0)
	{
            	dot_sum[0] = partial_sum[0]; // Save the dot product to the first element of dot_sum array
	}

}

/*!
	Set vector to a constant value
	
	\param d_a   the vector
	\param a     value to set vector to
	\param group_size       length of vector
	\param d_group_members  index into vector

	d_a = a
*/
__global__
void gpu_stokes_SetValue_kernel(Scalar4 *d_a, Scalar3 a, unsigned int group_size, unsigned int *d_group_members){
	int group_idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (group_idx < group_size) {
		unsigned int idx = d_group_members[group_idx];

		d_a[idx] = make_scalar4(a.x, a.y, a.z, d_a[idx].w);
	}
}

/*!

	Perform matrix-vector multiply needed for the Lanczos contribution to the Brownian velocity

	\param d_A 		matrix, N x m
	\param d_x		multiplying vector, m x 1
	\param d_b		result vector, A*x, m x 1
	\param group_size	number of particles
	\param m		number of iterations ( number of columns of A, length of x )

*/

__global__
void gpu_stokes_MatVecMultiply_kernel(Scalar4 *d_A, Scalar *d_x, Scalar4 *d_b, unsigned int group_size, int m){
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < group_size) {

		Scalar3 tempprod = make_scalar3( 0.0, 0.0, 0.0 );

		for ( int ii = 0; ii < m; ++ii ){

		    Scalar4 matidx = d_A[ idx + ii*group_size ];

		    Scalar xcurr = d_x[ii];

		    tempprod.x = tempprod.x + matidx.x * xcurr;
		    tempprod.y = tempprod.y + matidx.y * xcurr;
		    tempprod.z = tempprod.z + matidx.z * xcurr;

		}

		d_b[idx] = make_scalar4( tempprod.x, tempprod.y, tempprod.z, d_A[idx].w );

	}
}


/*!
	Add two grid vectors
	C = A + B

	\param d_a              input vector, A
	\param d_b              input vector, B
	\param d_c              output vector, C
	\param N                length of vectors
*/
__global__
void gpu_stokes_AddGrids_kernel(CUFFTCOMPLEX *d_a, CUFFTCOMPLEX *d_b, CUFFTCOMPLEX *d_c, unsigned int NxNyNz){
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	if ( tidx < NxNyNz) {
		unsigned int idx = tidx;
		CUFFTCOMPLEX A = d_a[idx];
		CUFFTCOMPLEX B = d_b[idx];
		d_c[idx] = make_scalar2(A.x+B.x, A.y+B.y);
	}
}


/*!
	Add scale the grid
	A = s * A

	\param d_a   input vector, A
	\param s     scale factor
	\param N     length of vectors
*/
__global__
void gpu_stokes_ScaleGrid_kernel(CUFFTCOMPLEX *d_a, Scalar s, unsigned int NxNyNz){
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	if ( tidx < NxNyNz) {
		unsigned int idx = tidx;
		CUFFTCOMPLEX A = d_a[idx];
		d_a[idx] = make_scalar2(s*A.x, s*A.y);
	}
}

/*!
  Kernel function to calculate position of each grid in reciprocal space: gridk
  */
__global__
void gpu_stokes_SetGridk_kernel(Scalar4 *gridk,
                                int Nx,
                                int Ny,
                                int Nz,
                                unsigned int NxNyNz,
                                BoxDim box,
                                Scalar xi,
				Scalar eta)
{
        int tid = blockDim.x * blockIdx.x + threadIdx.x;

        if ( tid < NxNyNz ) {

                int i = tid / (Ny*Nz);
                int j = (tid - i * Ny * Nz) / Nz;
                int k = tid % Nz;

                Scalar3 L = box.getL();
                Scalar xy = box.getTiltFactorXY();
                Scalar4 gridk_value;

                gridk_value.x = (i < (Nx+1) / 2) ? i : i - Nx;
                gridk_value.y = ( ((j < (Ny+1) / 2) ? j : j - Ny) - xy * gridk_value.x * L.y / L.x ) / L.y; // Fixed by Zsigi 2015
                gridk_value.x = gridk_value.x / L.x;
                gridk_value.z = ((k < (Nz+1) / 2) ? k : k - Nz) / L.z;

		gridk_value.x *= 2.0*3.1416926536;
		gridk_value.y *= 2.0*3.1416926536;
		gridk_value.z *= 2.0*3.1416926536;

                Scalar k2 = gridk_value.x*gridk_value.x + gridk_value.y*gridk_value.y + gridk_value.z*gridk_value.z;
		Scalar xisq = xi * xi;

		// Scaling factor used in wave space sum
		if (i == 0 && j == 0 && k == 0){
			gridk_value.w = 0.0;
		}
		else{
			// Have to divide by Nx*Ny*Nz to normalize the FFTs
			gridk_value.w = 6.0*3.1415926536 * (1.0 + k2/4.0/xisq) * expf( -(1-eta) * k2/4.0/xisq ) / ( k2 ) / Scalar( Nx*Ny*Nz );
		}

                gridk[tid] = gridk_value;

        }
}


