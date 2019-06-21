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
// Modified by Andrew Fiore

/*! \file Brownian.cuh
    \brief Declares GPU kernel codes for Brownian Calculations.
*/
#include "hoomd/ParticleData.cuh"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"

#include <cufft.h>

//! Define the kernel
#ifndef __BROWNIAN_CUH__
#define __BROWNIAN_CUH__

//! Definition for complex variable storage
#ifdef SINGLE_PRECISION
#define CUFFTCOMPLEX cufftComplex
#else
#define CUFFTCOMPLEX cufftComplex
#endif

__global__ void gpu_stokes_BrownianGenerate_kernel(
        Scalar4 *d_psi,
        unsigned int group_size,
        unsigned int *d_group_members,
        const unsigned int timestep, 
        const unsigned int seed 
        );

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
          );

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
        Scalar self 
        );

#endif
