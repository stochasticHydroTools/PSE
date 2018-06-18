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

/*! \file Stokes.cuh
    \brief Declares GPU kernel code for integration considering hydrodynamic interactions on the GPU. Used by Stokes.
*/
#include "hoomd/hoomd_config.h"
#include "ParticleData.cuh"
#include "HOOMDMath.h"
#include <cufft.h>
#include "Index1D.h"

//! Define the step_one kernel
#ifndef __HELPER_CUH__
#define __HELPER_CUH__

//! Definition for comxplex variable storage
#ifdef SINGLE_PRECISION
#define CUFFTCOMPLEX cufftComplex
#else
#define CUFFTCOMPLEX cufftComplex
#endif

__global__ void gpu_stokes_ZeroGrid_kernel(CUFFTCOMPLEX *grid, unsigned int NxNyNz);

__global__ void gpu_stokes_LinearCombination_kernel(Scalar4 *d_a, Scalar4 *d_b, Scalar4 *d_c, Scalar coeff_a, Scalar coeff_b, unsigned int group_size, unsigned int *d_group_members);

__global__ void gpu_stokes_DotStepOne_kernel(Scalar4 *d_a, Scalar4 *d_b, Scalar *dot_sum, unsigned int group_size, unsigned int *d_group_members);

__global__ void gpu_stokes_DotStepTwo_kernel(Scalar *dot_sum, unsigned int num_partial_sums);

__global__ void gpu_stokes_SetValue_kernel(Scalar4 *d_a, Scalar3 a, unsigned int group_size, unsigned int *d_group_members);

__global__ void gpu_stokes_MatVecMultiply_kernel(Scalar4 *d_A, Scalar *d_x, Scalar4 *d_b, unsigned int group_size, int m);

__global__ void gpu_stokes_AddGrids_kernel(CUFFTCOMPLEX *d_a, CUFFTCOMPLEX *d_b, CUFFTCOMPLEX *d_c, unsigned int NxNyNz);

__global__ void gpu_stokes_ScaleGrid_kernel(CUFFTCOMPLEX *d_a, Scalar s, unsigned int NxNyNz);

__global__ void gpu_stokes_SetGridk_kernel(Scalar4 *gridk, int Nx, int Ny, int Nz, unsigned int NxNyNz, BoxDim box, Scalar xi, Scalar eta);


#endif
