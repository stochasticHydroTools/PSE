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

// HOOMD Maintainer: joaander
// Modified by Andrew Fiore

#include "Brownian.cuh"
#include "Mobility.cuh"
#include "Helper.cuh"

#include "hoomd/Saru.h"
#include "hoomd/TextureTools.h"
using namespace hoomd;

#include <stdio.h>
#include <math.h>

#include "lapacke.h"
#include "cblas.h"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif


/*! \file Brownian.cu
    \brief Defines functions for PSE calculation of the Brownian Displacements

    // Uses LAPACKE to perform the final square root of the tridiagonal matrix
  resulting from the Lanczos Method
*/

//! Shared memory array for partial sum of dot product kernel
extern __shared__ Scalar partial_sum[];
extern __shared__ Scalar4 shared_Fpos[];

/*!
    Generate random numbers on particles
  
  \param d_psi            random vector
        \param group_size       number of particles
  \param d_group_members  index to particle arrays
  \param timestep         current time step
  \param seed             seed for random number generation

  Thread-per-particle operations to generate random numbers
  for the real space part of the Brownian calculation. Grid
  and blocks are 1-D.

*/
__global__ void gpu_stokes_BrownianGenerate_kernel(
          Scalar4 *d_psi,
          unsigned int group_size,
          unsigned int *d_group_members,
          const unsigned int timestep, 
          const unsigned int seed
          ){

  // Thread ID
  int group_idx = blockDim.x * blockIdx.x + threadIdx.x;

  // Make sure that thread is in bounds
  if (group_idx < group_size) {

    // Global particle index
    unsigned int idx = d_group_members[group_idx];

    // Initialize random number generator
    detail::Saru s(idx, timestep + seed);

    // Draw numbers from a Uniform distribution (-sqrt(3),sqrt(3),
    // so that variance = 1/3
    Scalar sqrt3 = 1.73205080757;
    Scalar randomx = s.f( -sqrt3, sqrt3 );
    Scalar randomy = s.f( -sqrt3, sqrt3 );
    Scalar randomz = s.f( -sqrt3, sqrt3 );

    // Write to global memory, leaving the 4th element unchanged
    d_psi[idx] = make_scalar4(randomx, randomy, randomz, d_psi[idx].w);

  }
}

/*!
    Generate random numbers for wave space Brownian motion ( random numbers on grid )
          - scale forces as they're generated and add directly to the existing grid.  

  \param d_gridX    x-component of vectors on grid
  \param d_gridY    y-component of vectors on grid
  \param d_gridZ    z-component of vectors on grid
  \param d_gridk    reciprocal lattice vectors for each grid point
  \param NxNyNz    total number of grid points
  \param Nx    number of grid points in x-direction
  \param Ny    number of grid points in y-direction
  \param Nz    number of grid points in z-direction
  \param timestep         current simulation time step
  \param seed             seed for random number generation
  \param T    simulation temperature
  \param dt    simulation time step size
  \param quadW    quadrature weight for spectral Ewald integration

  Thread per grid node. 1-D grid of blocks, 1-D block of threads.

*/
__global__ void gpu_stokes_BrownianGridGenerate_kernel(  
            CUFFTCOMPLEX *gridX,
            CUFFTCOMPLEX *gridY,
            CUFFTCOMPLEX *gridZ,
            Scalar4 *gridk,
                  unsigned int NxNyNz,
            int Nx,
            int Ny,
            int Nz,
                  const unsigned int timestep, 
                  const unsigned int seed,
            Scalar T,
            Scalar dt,
            Scalar quadW
                       ){

  // Current thread index
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  // Check if threads are in bounds
  if ( idx < NxNyNz ) {
      
    // Random number generator
    detail::Saru s(idx, timestep + seed);
  
    // Square root of 3.0 / 2.0
    Scalar sqrt3d2 = 1.2247448713915889;
  
    // Get random numbers from uniform distribution
    // on (-sqrt(3/2),sqrt(3/2)) so that variance
    // of ( reX + reY ) = 1.0
    Scalar reX = s.f( -sqrt3d2, sqrt3d2 );
    Scalar reY = s.f( -sqrt3d2, sqrt3d2 );
    Scalar reZ = s.f( -sqrt3d2, sqrt3d2 );
    Scalar imX = s.f( -sqrt3d2, sqrt3d2 );
    Scalar imY = s.f( -sqrt3d2, sqrt3d2 );
    Scalar imZ = s.f( -sqrt3d2, sqrt3d2 );
    
    // Modulo arithmetic for indices for current grid point
    int kk = idx % Nz;
    int jj = ( ( idx - kk ) / Nz ) % Ny;
    int ii = ( ( idx - kk ) / Nz - jj ) / Ny;
      
    // Scaling factor for covariance
    Scalar fac = sqrtf(2.0*T/dt/quadW);
    
    // Variables required to place values on the grid
    Scalar2 fX, fY, fZ;         // forces for thread's point
    Scalar2 fX_conj, fY_conj, fZ_conj; // forces for thread's conjugate point
    Scalar2 kdF, kdF_conj;       // dot(k,F) for thread and conjugate point
    Scalar B12, B12_conj;       // Scaling factors for thread and conjugate point
    
    // Only do work on half the grid points because we are simultaneously assigning values
    // to each grid point and its conjugate. The following check makes sure we pick all of
    // the points without conjugates (zeros and nyquist points) as well as all the points
    // in the upper half of the grid. Also, ignore the origin in the wave space sum. (Sum
    // is over all k!= 0)
    if ( 
         !( 2 * kk >= Nz + 1 ) &&
         !( ( kk == 0 ) && ( 2 * jj >= Ny + 1 ) ) &&
         !( ( kk == 0 ) && ( jj == 0 ) && ( 2 * ii >= Nx + 1 ) ) &&
         !( ( kk == 0 ) && ( jj == 0 ) && ( ii == 0 ) ) 
    ){

      // Is current grid point a nyquist point
      bool ii_nyquist = ( ( ii == Nx/2 ) && ( Nx/2 == (Nx+1)/2 ) );
      bool jj_nyquist = ( ( jj == Ny/2 ) && ( Ny/2 == (Ny+1)/2 ) );
      bool kk_nyquist = ( ( kk == Nz/2 ) && ( Nz/2 == (Nz+1)/2 ) );
      
      // Index of conjugate point
      int ii_conj, jj_conj, kk_conj;
      if ( ii == 0 ){
        ii_conj = ii;
      }
      else {
        ii_conj = Nx - ii;
      }
      if ( jj == 0 ){
        jj_conj = jj;
      }
      else {
        jj_conj = Ny - jj;
      }
      if ( kk == 0 ){
        kk_conj = kk;
      }
      else {
        kk_conj = Nz - kk;
      }
    
      // Global index of conjugate grid point
      int conj_idx = ii_conj * Ny*Nz + jj_conj * Nz + kk_conj;
    
      // Current wave-space vector, conjugate wave space vector, and their
      // magnitudes
      Scalar4 tk = gridk[idx];
      Scalar4 tk_conj = gridk[conj_idx];
    
      Scalar ksq = tk.x*tk.x + tk.y*tk.y + tk.z*tk.z;
      Scalar ksq_conj = tk_conj.x*tk_conj.x + tk_conj.y*tk_conj.y + tk_conj.z*tk_conj.z;
    
      // Assign fluctuating values to the Nyquist points (no conjugate points)
      if ( ( ii == 0    && jj_nyquist && kk == 0 ) ||
           ( ii_nyquist && jj == 0    && kk == 0 ) ||
           ( ii_nyquist && jj_nyquist && kk == 0 ) ||
           ( ii == 0    && jj == 0    && kk_nyquist ) ||
           ( ii == 0    && jj_nyquist && kk_nyquist ) ||
           ( ii_nyquist && jj == 0    && kk_nyquist ) ||
           ( ii_nyquist && jj_nyquist && kk_nyquist ) ){
    
        // At the nyquist point, the random quantity only has a real component. Have to
        // multiply by sqrt(2.0) to make sure the variance is still 1
        Scalar sqrt2 = 1.4142135623730951;
        fX = make_scalar2( sqrt2*reX, 0.0 );
        fY = make_scalar2( sqrt2*reY, 0.0 );
        fZ = make_scalar2( sqrt2*reZ, 0.0 );
    
        // Dot product of wave-vector with stochastic quantity
        kdF = make_scalar2( ( tk.x*fX.x + tk.y*fY.x + tk.z*fZ.x ) / ksq,  ( tk.x*fX.y + tk.y*fY.y + tk.z*fZ.y ) / ksq );
        
        // Scaling factor
        B12 = sqrtf( tk.w );
        Scalar k = sqrtf( ksq );
        B12 *= sinf( k ) / k;
    
        // Add random quantity to the grid AND scale by B^(1/2) simultaneously to save effort
        gridX[idx].x = gridX[idx].x + fac * ( fX.x - tk.x * kdF.x ) * B12;
        gridX[idx].y = gridX[idx].y + fac * ( fX.y - tk.x * kdF.y ) * B12;
        
        gridY[idx].x = gridY[idx].x + fac * ( fY.x - tk.y * kdF.x ) * B12;
        gridY[idx].y = gridY[idx].y + fac * ( fY.y - tk.y * kdF.y ) * B12;
        
        gridZ[idx].x = gridZ[idx].x + fac * ( fZ.x - tk.z * kdF.x ) * B12;
        gridZ[idx].y = gridZ[idx].y + fac * ( fZ.y - tk.z * kdF.y ) * B12;
    
      }
      else {
    
        // Construct random force
        fX = make_scalar2( reX, imX );
        fY = make_scalar2( reY, imY );
        fZ = make_scalar2( reZ, imZ );
    
        // The random force at the conjugate point is the conjugate of the force at
        // the current point
        fX_conj = make_scalar2( reX, -imX );
        fY_conj = make_scalar2( reY, -imY );
        fZ_conj = make_scalar2( reZ, -imZ );

        // Dot prodcut of force with wave vector at current and conjugate point
        kdF = make_scalar2( ( tk.x*fX.x + tk.y*fY.x + tk.z*fZ.x ) / ksq,  ( tk.x*fX.y + tk.y*fY.y + tk.z*fZ.y ) / ksq );
        kdF_conj = make_scalar2( ( tk_conj.x*fX_conj.x + tk_conj.y*fY_conj.x + tk_conj.z*fZ_conj.x ) / ksq_conj,  ( tk_conj.x*fX_conj.y + tk_conj.y*fY_conj.y + tk_conj.z*fZ_conj.y ) / ksq_conj );
      
        // Scaling factors at current and conjugate point
        B12 = sqrtf( tk.w );
        B12_conj = sqrtf( tk_conj.w );
        
        Scalar k = sqrtf( ksq );
        Scalar kconj = sqrtf( ksq_conj );
        B12 *= sinf( k ) / k;
        B12_conj *= sinf( kconj ) / kconj;
    
        // Add random quantity to the grid AND scale by B^(1/2) simultaneously to save effort
        // Current grid point
        gridX[idx].x = gridX[idx].x + fac * ( fX.x - tk.x * kdF.x ) * B12;
        gridX[idx].y = gridX[idx].y + fac * ( fX.y - tk.x * kdF.y ) * B12;
        
        gridY[idx].x = gridY[idx].x + fac * ( fY.x - tk.y * kdF.x ) * B12;
        gridY[idx].y = gridY[idx].y + fac * ( fY.y - tk.y * kdF.y ) * B12;
        
        gridZ[idx].x = gridZ[idx].x + fac * ( fZ.x - tk.z * kdF.x ) * B12;
        gridZ[idx].y = gridZ[idx].y + fac * ( fZ.y - tk.z * kdF.y ) * B12;
      
        // Add random quantity to the grid AND scale by B^(1/2) simultaneously to save effort
        // Conjugate grid point
        gridX[conj_idx].x = gridX[conj_idx].x + fac * ( fX_conj.x - tk_conj.x * kdF_conj.x ) * B12_conj;
        gridX[conj_idx].y = gridX[conj_idx].y + fac * ( fX_conj.y - tk_conj.x * kdF_conj.y ) * B12_conj;
        
        gridY[conj_idx].x = gridY[conj_idx].x + fac * ( fY_conj.x - tk_conj.y * kdF_conj.x ) * B12_conj;
        gridY[conj_idx].y = gridY[conj_idx].y + fac * ( fY_conj.y - tk_conj.y * kdF_conj.y ) * B12_conj;
        
        gridZ[conj_idx].x = gridZ[conj_idx].x + fac * ( fZ_conj.x - tk_conj.z * kdF_conj.x ) * B12_conj;
        gridZ[conj_idx].y = gridZ[conj_idx].y + fac * ( fZ_conj.y - tk_conj.z * kdF_conj.y ) * B12_conj;
    
      }
    
    
    
    }

 
      }
}


/*!
  Use Lanczos method to compute Mreal^0.5 * psi

  This method is detailed in:
    "Preconditioned Krylov Subspace Methods for Sampling Multivariate Gaussian Distributions"
    Edmond Chow and Yousef Saad, IAM J. Sci. Comput., 36(2), A588–A608


*/
void gpu_stokes_BrealLanczos_wrap(   
          Scalar4 *d_psi,
             Scalar4 *d_pos,
                                     unsigned int *d_group_members,
                                     unsigned int group_size,
                                     const BoxDim& box,
                                     Scalar dt,
                   Scalar4 *d_vel,
                   const Scalar T,
                   const unsigned int timestep,
                   const unsigned int seed,
                   Scalar xi,
                   Scalar ewald_cut,
                   Scalar ewald_dr,
                   int ewald_n,
                   Scalar4 *d_ewaldC1, 
                   const unsigned int *d_n_neigh,
                                     const unsigned int *d_nlist,
                                     const unsigned int *d_headlist,
                   int& m,
             Scalar cheb_error,
                   dim3 grid,
                   dim3 threads,
                   int gridBlockSize,
                   int gridNBlock,
                   Scalar3 gridh,
                   Scalar self 
          ){

  // Dot product kernel specifications
  unsigned int thread_for_dot = 512; // Must be 2^n
  unsigned int grid_for_dot = (group_size/thread_for_dot) + 1;

  // Temp var for dot product.
  Scalar *dot_sum;
  cudaMalloc( (void**)&dot_sum, grid_for_dot*sizeof(Scalar) );

  // Allocate storage
  // 
  int m_in = m;
  int m_max = 100;

        // Storage vectors for tridiagonal factorization
  float *alpha, *beta, *alpha_save, *beta_save;
        alpha = (float *)malloc( (m_max)*sizeof(float) );
        alpha_save = (float *)malloc( (m_max)*sizeof(float) );
        beta = (float *)malloc( (m_max+1)*sizeof(float) );
        beta_save = (float *)malloc( (m_max+1)*sizeof(float) );

  // Vectors for Lapacke and square root
  float *W;
  W = (float *)malloc( (m_max*m_max)*sizeof(float) );
  float *W1; // W1 = Lambda^(1/2) * ( W^T * e1 )
  W1 = (float *)malloc( (m_max)*sizeof(float) );
  float *Tm;
  Tm = (float *)malloc( m_max*sizeof(float) );
  Scalar *d_Tm;
  cudaMalloc( (void**)&d_Tm, m_max * sizeof(Scalar) );

  // Vectors for Lanczos iterations
  Scalar4 *d_v, *d_vj, *d_vjm1;
  cudaMalloc( (void**)&d_v, group_size*sizeof(Scalar4) );
  cudaMalloc( (void**)&d_vj, group_size*sizeof(Scalar4) );
  cudaMalloc( (void**)&d_vjm1, group_size*sizeof(Scalar4) );

  // Storage vector for M*vj
  Scalar4 *d_Mvj;
  cudaMalloc( (void**)&d_Mvj, group_size*sizeof(Scalar4) );

  // Storage array for V
  Scalar4 *d_V;
  cudaMalloc( (void**)&d_V, m_max*group_size * sizeof(Scalar4) );

  // Step-norm things
  Scalar4 *d_vel_old, *d_Mpsi;
  cudaMalloc( (void**)&d_vel_old, group_size*sizeof(Scalar4) );
  cudaMalloc( (void**)&d_Mpsi, group_size*sizeof(Scalar4) );
  Scalar psiMpsi;

  // Temporary pointer
  Scalar4 *d_temp;

  // Copy random vector to v0
  cudaMemcpy( d_vj, d_psi, group_size*sizeof(Scalar4), cudaMemcpyDeviceToDevice );
  
  // Compute the norm of the d_psi (also the norm of basis vector v0)
        Scalar vnorm;
  gpu_stokes_DotStepOne_kernel<<< grid_for_dot, thread_for_dot, thread_for_dot*sizeof(Scalar) >>>(d_vj, d_vj, dot_sum, group_size, d_group_members);
  gpu_stokes_DotStepTwo_kernel<<< 1, thread_for_dot, thread_for_dot*sizeof(Scalar) >>>(dot_sum, grid_for_dot);
  cudaMemcpy(&vnorm, dot_sum, sizeof(Scalar), cudaMemcpyDeviceToHost);
  vnorm = sqrtf( vnorm );

  Scalar psinorm = vnorm;

      // Compute psi * M * psi ( for step norm )
      gpu_stokes_Mreal_kernel<<<grid, threads>>>(d_pos, d_Mpsi, d_psi, group_size, xi, d_ewaldC1, self, ewald_cut, ewald_n, ewald_dr, d_group_members, box, d_n_neigh, d_nlist, d_headlist );
      gpu_stokes_DotStepOne_kernel<<< grid_for_dot, thread_for_dot, thread_for_dot*sizeof(Scalar) >>>(d_psi, d_Mpsi, dot_sum, group_size, d_group_members);
      gpu_stokes_DotStepTwo_kernel<<< 1, thread_for_dot, thread_for_dot*sizeof(Scalar) >>>(dot_sum, grid_for_dot);
      cudaMemcpy(&psiMpsi, dot_sum, sizeof(Scalar), cudaMemcpyDeviceToHost);

  psiMpsi = psiMpsi / ( psinorm * psinorm );

        // First iteration, vjm1 = 0, vj = psi / norm( psi )
  gpu_stokes_LinearCombination_kernel<<<grid, threads>>>(d_vj, d_vj, d_vjm1, 0.0, 0.0, group_size, d_group_members);
  gpu_stokes_LinearCombination_kernel<<<grid, threads>>>(d_vj, d_vj, d_vj, 1.0/vnorm, 0.0, group_size, d_group_members);

  // Start by computing (m-1) iterations, so that the stepnorm for the given
  // number of iterations can be compute
  m = m_in - 1;
  m = m < 1 ? 1 : m;

  // Values for current alpha and beta in the iteration
  Scalar tempalpha;
  Scalar tempbeta = 0.0;

  // Apply the Lanczos method
  for ( int jj = 0; jj < m; ++jj ){

    // Store current basis vector
    cudaMemcpy( &d_V[jj*group_size], d_vj, group_size*sizeof(Scalar4), cudaMemcpyDeviceToDevice );

    // Store beta
    beta[jj] = tempbeta;

    // v = M*vj - betaj*vjm1
        gpu_stokes_Mreal_kernel<<<grid, threads>>>(d_pos, d_Mvj, d_vj, group_size, xi, d_ewaldC1, self, ewald_cut, ewald_n, ewald_dr, d_group_members, box, d_n_neigh, d_nlist, d_headlist );
    gpu_stokes_LinearCombination_kernel<<<grid, threads>>>(d_Mvj, d_vjm1, d_v, 1.0, -1.0*tempbeta, group_size, d_group_members);

    // vj dot v
          gpu_stokes_DotStepOne_kernel<<< grid_for_dot, thread_for_dot, thread_for_dot*sizeof(Scalar) >>>(d_vj, d_v, dot_sum, group_size, d_group_members);
          gpu_stokes_DotStepTwo_kernel<<< 1, thread_for_dot, thread_for_dot*sizeof(Scalar) >>>(dot_sum, grid_for_dot);
          cudaMemcpy(&tempalpha, dot_sum, sizeof(Scalar), cudaMemcpyDeviceToHost);

    // Store updated alpha
    alpha[jj] = tempalpha;
  
    // v = v - alphaj*vj
    gpu_stokes_LinearCombination_kernel<<<grid, threads>>>(d_v, d_vj, d_v, 1.0, -1.0*tempalpha, group_size, d_group_members);

    // v dot v 
          gpu_stokes_DotStepOne_kernel<<< grid_for_dot, thread_for_dot, thread_for_dot*sizeof(Scalar) >>>(d_v, d_v, dot_sum, group_size, d_group_members);
          gpu_stokes_DotStepTwo_kernel<<< 1, thread_for_dot, thread_for_dot*sizeof(Scalar) >>>(dot_sum, grid_for_dot);
          cudaMemcpy(&vnorm, dot_sum, sizeof(Scalar), cudaMemcpyDeviceToHost);
    vnorm = sqrtf( vnorm );

    // betajp1 = norm( v )
    tempbeta = vnorm;

    // Check that the basis vector is not too small. If so, end the iteration
    // (If norm is too small, will have numerical trouble)
    if ( vnorm < 1E-8 ){
        m = jj;
        break;
    }

    // vjp1 = v / betajp1
    gpu_stokes_LinearCombination_kernel<<<grid, threads>>>(d_v, d_v, d_v, 1.0/tempbeta, 0.0, group_size, d_group_members);

    // Swap pointers
    d_temp = d_vjm1;
    d_vjm1 = d_vj;
    d_vj = d_v;
    d_v = d_temp;
    
  }

  // Save alpha, beta vectors (will be overwritten by lapack)
  for ( int ii = 0; ii < m; ++ii ){
    alpha_save[ii] = alpha[ii];
    beta_save[ii] = beta[ii];
  }
  beta_save[m] = beta[m];

  // Now that we have alpha, beta, have to compute the square root of the tridiagonal
  // matrix Tm. Do this using eigen-decomposition.
  //
  // Compute eigen-decomposition of tridiagonal matrix
  //   alpha (input) - vector of entries on main diagonal
  //      alpha (output) - eigenvalues sorted in descending order
  //      beta (input) - vector of entries of sub-diagonal
  //      beta (output) - overwritten (zeros?)
  //      W - (output) - matrix of eigenvectors. ith column corresponds to ith eigenvalue
  //   INFO (output) = 0 if operation was succesful
  int INFO = LAPACKE_spteqr( LAPACK_ROW_MAJOR, 'I', m, alpha, &beta[1], W, m );

  // Check whether the eigen-decomposition failed, and throw error on failure
  if ( INFO != 0 ){
    printf("Eigenvalue decomposition #1 failed \n");
    printf("INFO = %i \n", INFO);

    printf("\n alpha: \n");
    for( int ii = 0; ii < m; ++ii ){
        printf("%f \n", alpha_save[ii]);
    } 
    printf("\n beta: \n");
    for( int ii = 0; ii < m; ++ii ){
        printf("%f \n", beta_save[ii]);
    }
    printf("%f \n", beta_save[m]); 
      
    printf("Note to User: restart simulation and proceed. \n");

    exit(EXIT_FAILURE);
  }

  // Now, we have to compute Tm^(1/2) * e1
  //   Tm^(1/2) = W * Lambda^(1/2) * W^T * e1
  //           = W * Lambda^(1/2) * ( W^T * e1 )
  // The quantity in parentheses is the first row of W 
  // Lambda^(1/2) only has diagonal entries, so it's product with the first row of W
  //     is easy to compute.
  for ( int ii = 0; ii < m; ++ii ){
      W1[ii] = sqrtf( alpha[ii] ) * W[ii];
  }

  // Tm = W * W1 = W * Lambda^(1/2) * W^T * e1
  float tempsum;
  for ( int ii = 0; ii < m; ++ii ){
      tempsum = 0.0;
      for ( int jj = 0; jj < m; ++jj ){
    int idx = m*ii + jj;

    tempsum += W[idx] * W1[jj];
      }
      Tm[ii] = tempsum;
  }

  // Copy matrix to GPU
  cudaMemcpy( d_Tm, Tm, m*sizeof(Scalar), cudaMemcpyHostToDevice );

  // Multiply basis vectors by Tm, [ V0, V1, ..., Vm-1 ] * Tm
  gpu_stokes_MatVecMultiply_kernel<<<grid,threads>>>(d_V, d_Tm, d_vel, group_size, m);

  // Copy velocity
  cudaMemcpy( d_vel_old, d_vel, group_size*sizeof(Scalar4), cudaMemcpyDeviceToDevice );

  // Restore alpha, beta
  for ( int ii = 0; ii < m; ++ii ){
    alpha[ii] = alpha_save[ii];
    beta[ii] = beta_save[ii];
  }
  beta[m] = beta_save[m];


  //
  // Keep adding to basis vectors until the step norm is small enough
  //
  Scalar stepnorm = 1.0;
  int jj;
  while( stepnorm > cheb_error && m < m_max ){
    m++;
    jj = m - 1;

    //
    // Do another Lanczos iteration
    //

    // Store the current basis vector
    cudaMemcpy( &d_V[jj*group_size], d_vj, group_size*sizeof(Scalar4), cudaMemcpyDeviceToDevice );

    // Store beta
    beta[jj] = tempbeta;

    // v = M*vj - betaj*vjm1
    gpu_stokes_Mreal_kernel<<<grid, threads>>>(d_pos, d_Mvj, d_vj, group_size, xi, d_ewaldC1, self, ewald_cut, ewald_n, ewald_dr, d_group_members, box, d_n_neigh, d_nlist, d_headlist );
    gpu_stokes_LinearCombination_kernel<<<grid, threads>>>(d_Mvj, d_vjm1, d_v, 1.0, -1.0*tempbeta, group_size, d_group_members);

    // vj dot v
          gpu_stokes_DotStepOne_kernel<<< grid_for_dot, thread_for_dot, thread_for_dot*sizeof(Scalar) >>>(d_vj, d_v, dot_sum, group_size, d_group_members);
          gpu_stokes_DotStepTwo_kernel<<< 1, thread_for_dot, thread_for_dot*sizeof(Scalar) >>>(dot_sum, grid_for_dot);
          cudaMemcpy(&tempalpha, dot_sum, sizeof(Scalar), cudaMemcpyDeviceToHost);

    // Store updated alpha
    alpha[jj] = tempalpha;
  
    // v = v - alphaj*vj
    gpu_stokes_LinearCombination_kernel<<<grid, threads>>>(d_v, d_vj, d_v, 1.0, -1.0*tempalpha, group_size, d_group_members);

    // v dot v 
          gpu_stokes_DotStepOne_kernel<<< grid_for_dot, thread_for_dot, thread_for_dot*sizeof(Scalar) >>>(d_v, d_v, dot_sum, group_size, d_group_members);
          gpu_stokes_DotStepTwo_kernel<<< 1, thread_for_dot, thread_for_dot*sizeof(Scalar) >>>(dot_sum, grid_for_dot);
          cudaMemcpy(&vnorm, dot_sum, sizeof(Scalar), cudaMemcpyDeviceToHost);
    vnorm = sqrtf( vnorm );

    // betajp1 = norm( v )
    tempbeta = vnorm; 

    // Check if the norm of the basis vector is too small. If
    // so, end the iteration. 
    if ( vnorm < 1E-8 ){
        m = jj;
        break;
    }

    // vjp1 = v / betajp1
    gpu_stokes_LinearCombination_kernel<<<grid, threads>>>(d_v, d_v, d_v, 1.0/tempbeta, 0.0, group_size, d_group_members);

    // Swap pointers
    d_temp = d_vjm1;
    d_vjm1 = d_vj;
    d_vj = d_v;
    d_v = d_temp;
      
    // Save alpha, beta vectors (will be overwritten by lapack)
    for ( int ii = 0; ii < m; ++ii ){
      alpha_save[ii] = alpha[ii];
      beta_save[ii] = beta[ii];
    }
    beta_save[m] = beta[m];
  
    //
    // Square root calculation with addition of latest Lanczos iteration
    // (see first implementation above more description)
    //
  
    // Compute eigen-decomposition of tridiagonal matrix
    int INFO = LAPACKE_spteqr( LAPACK_ROW_MAJOR, 'I', m, alpha, &beta[1], W, m );

    // Check whether the eigen-decomposition failed, and throw error on failure
    if ( INFO != 0 ){
          printf("Eigenvalue decomposition #2 failed \n");
          printf("INFO = %i \n", INFO); 
      
              printf("\n alpha: \n");
              for( int ii = 0; ii < m; ++ii ){
                  printf("%f \n", alpha_save[ii]);
              } 
              printf("\n beta: \n");
              for( int ii = 0; ii < m; ++ii ){
                  printf("%f \n", beta_save[ii]);
              }
          printf("%f \n", beta_save[m]); 
    
      printf("Note to User: restart simulation and proceed. \n");
      
      exit(EXIT_FAILURE);
    }

    // Now, we have to compute Tm^(1/2) * e1
    for ( int ii = 0; ii < m; ++ii ){
        W1[ii] = sqrtf( alpha[ii] ) * W[ii];
    }

    // Tm = W * W1 = W * Lambda^(1/2) * W^T * e1
    float tempsum;
    for ( int ii = 0; ii < m; ++ii ){
        tempsum = 0.0;
        for ( int jj = 0; jj < m; ++jj ){
      int idx = m*ii + jj;

      tempsum += W[idx] * W1[jj];
        }
        Tm[ii] = tempsum;
    }

    // Copy matrix to GPU
    cudaMemcpy( d_Tm, Tm, m*sizeof(Scalar), cudaMemcpyHostToDevice );

    // Multiply basis vectors by Tm -- velocity = Vm * Tm
    gpu_stokes_MatVecMultiply_kernel<<<grid,threads>>>(d_V, d_Tm, d_vel, group_size, m);

    // Compute step norm error
        gpu_stokes_LinearCombination_kernel<<<grid, threads>>>(d_vel, d_vel_old, d_vel_old, 1.0, -1.0, group_size, d_group_members);
          gpu_stokes_DotStepOne_kernel<<< grid_for_dot, thread_for_dot, thread_for_dot*sizeof(Scalar) >>>(d_vel_old, d_vel_old, dot_sum, group_size, d_group_members);
          gpu_stokes_DotStepTwo_kernel<<< 1, thread_for_dot, thread_for_dot*sizeof(Scalar) >>>(dot_sum, grid_for_dot);
          cudaMemcpy(&stepnorm, dot_sum, sizeof(Scalar), cudaMemcpyDeviceToHost);

    stepnorm = sqrtf( stepnorm / psiMpsi );

    // Copy velocity
    cudaMemcpy( d_vel_old, d_vel, group_size*sizeof(Scalar4), cudaMemcpyDeviceToDevice );

    // Restore alpha, beta
    for ( int ii = 0; ii < m; ++ii ){
      alpha[ii] = alpha_save[ii];
      beta[ii] = beta_save[ii];
    }
    beta[m] = beta_save[m];

  }

  // Rescale by original norm of Psi and include thermal variance
  gpu_stokes_LinearCombination_kernel<<<grid, threads>>>(d_vel, d_vel, d_vel, psinorm * sqrtf(2.0*T/dt), 0.0, group_size, d_group_members);
  
  //
  // Clean up
  //
  cudaFree(dot_sum);
  cudaFree(d_Mvj);
  cudaFree(d_v);
  cudaFree(d_vj);
  cudaFree(d_vjm1);
  cudaFree(d_V);
  cudaFree(d_Tm);
  cudaFree(d_vel_old);
  cudaFree(d_Mpsi);

  d_temp = NULL;

  free(alpha);
  free(beta);
  free(alpha_save);
  free(beta_save);

  free(W);
  free(W1);
  free(Tm);
  
}


// Wrap up everything to compute mobility AND brownian if necessary
//   - Combine Fourier components of Deterministic and Brownian calculation
//    in order to save extra FFTs and contraction operations
//      - Add deterministic and stochastic real space contributions
void gpu_stokes_CombinedMobilityBrownian_wrap(  
        Scalar4 *d_pos,
        Scalar4 *d_net_force,
                                unsigned int *d_group_members,
                                unsigned int group_size,
                                const BoxDim& box,
                                Scalar dt,
              Scalar4 *d_vel,
              const Scalar T,
              const unsigned int timestep,
              const unsigned int seed,
              Scalar xi,
        Scalar eta,
        Scalar P,
              Scalar ewald_cut,
              Scalar ewald_dr,
              int ewald_n,
              Scalar4 *d_ewaldC1, 
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
              unsigned int NxNyNz,
              dim3 grid,
              dim3 threads,
              int gridBlockSize,
              int gridNBlock,
        Scalar3 gridh,
              Scalar cheb_error,
        Scalar self ){

  // Real space velocity to add
  Scalar4 *d_vel2;
  cudaMalloc( (void**)&d_vel2, group_size*sizeof(Scalar4) );
  
  // Generate uniform distribution (-1,1) on d_psi
  Scalar4 *d_psi;
  cudaMalloc( (void**)&d_psi, group_size*sizeof(Scalar4) );
  gpu_stokes_BrownianGenerate_kernel<<<grid, threads>>>( d_psi, group_size, d_group_members, timestep, seed );
  
  // Spreading and contraction grid information and parameters
  dim3 Cgrid( group_size, 1, 1);
  int B = ( P < 10 ) ? P : 10;
  dim3 Cthreads(B, B, B);
  
  Scalar quadW = gridh.x * gridh.y * gridh.z;
  Scalar xisq = xi * xi;
  Scalar prefac = ( 2.0 * xisq / 3.1415926536 / eta ) * sqrtf( 2.0 * xisq / 3.1415926536 / eta );
  Scalar expfac = 2.0 * xisq / eta;
  
  // ********************************************
  // Wave Space Part of Deterministic Calculation
  // ********************************************
  
  // Reset the grid (remove any previously distributed forces)
  gpu_stokes_ZeroGrid_kernel<<<gridNBlock,gridBlockSize>>>(d_gridX,NxNyNz);
  gpu_stokes_ZeroGrid_kernel<<<gridNBlock,gridBlockSize>>>(d_gridY,NxNyNz);
  gpu_stokes_ZeroGrid_kernel<<<gridNBlock,gridBlockSize>>>(d_gridZ,NxNyNz);
  
  // Spread forces onto grid
  gpu_stokes_Spread_kernel<<<Cgrid, Cthreads>>>( d_pos, d_net_force, d_gridX, d_gridY, d_gridZ, group_size, Nx, Ny, Nz, d_group_members, box, P, gridh, xi, eta, prefac, expfac );
  
  // Perform FFT on gridded forces
  cufftExecC2C(plan, d_gridX, d_gridX, CUFFT_FORWARD);
  cufftExecC2C(plan, d_gridY, d_gridY, CUFFT_FORWARD);
  cufftExecC2C(plan, d_gridZ, d_gridZ, CUFFT_FORWARD);
  
  // Apply wave space scaling to FFT'd forces
  gpu_stokes_Green_kernel<<<gridNBlock,gridBlockSize>>>( d_gridX, d_gridY, d_gridZ, d_gridk, NxNyNz);
  
  
  // ***************************************
  // Wave Space Part of Brownian Calculation
  // ***************************************
  if ( T > 0.0 ){
  
    // Apply random fluctuations to wave space grid
    gpu_stokes_BrownianGridGenerate_kernel<<<gridNBlock,gridBlockSize>>>( d_gridX, d_gridY, d_gridZ, d_gridk, NxNyNz, Nx, Ny, Nz, timestep, seed, T, dt, quadW );
  
  }
  
  // ************************************
  // Finish the Wave Space Calculation
  // ************************************
  
  // Return rescaled forces to real space
  cufftExecC2C(plan, d_gridX, d_gridX, CUFFT_INVERSE);
  cufftExecC2C(plan, d_gridY, d_gridY, CUFFT_INVERSE);
  cufftExecC2C(plan, d_gridZ, d_gridZ, CUFFT_INVERSE);
  
  // Evaluate contribution of grid velocities at particle centers
  gpu_stokes_Contract_kernel<<<Cgrid, Cthreads, (B*B*B+1)*sizeof(float3)>>>( d_pos, d_vel, d_gridX, d_gridY, d_gridZ, group_size, Nx, Ny, Nz, xi, eta, d_group_members, box, P, gridh, quadW*prefac, expfac );
  
  // ***************************************
  // Real Space Part of Both Calculations
  // ***************************************
  
  // Deterministic part
  gpu_stokes_Mreal_kernel<<<grid, threads>>>(d_pos, d_vel2, d_net_force, group_size, xi, d_ewaldC1, self, ewald_cut, ewald_n, ewald_dr, d_group_members, box, d_n_neigh, d_nlist, d_headlist );
  
  // Add to velocity
  gpu_stokes_LinearCombination_kernel<<<grid, threads>>>(d_vel2, d_vel, d_vel, 1.0, 1.0, group_size, d_group_members);
  
  // Stochastic
  if ( T > 0.0 ){
  
    gpu_stokes_BrealLanczos_wrap(   d_psi,
            d_pos,
            d_group_members,
            group_size,
            box,
            dt,
            d_vel2,
            T,
            timestep,
            seed,
            xi,
            ewald_cut,
            ewald_dr,
            ewald_n,
            d_ewaldC1, 
            d_n_neigh,
            d_nlist,
            d_headlist,
            m_Lanczos,
                cheb_error,
            grid,
            threads,
            gridBlockSize,
            gridNBlock,
            gridh,
                self );
  
    // Add to velocity
    gpu_stokes_LinearCombination_kernel<<<grid, threads>>>(d_vel2, d_vel, d_vel, 1.0, 1.0, group_size, d_group_members);
  
  }
  
  // Free Memory
  cudaFree( d_vel2 );
  cudaFree( d_psi );

}

