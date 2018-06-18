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

/*! \file Stokes.cuh
    \brief Declares GPU kernel code for integration considering hydrodynamic interactions on the GPU. Used by Stokes.
*/
#include "hoomd/hoomd_config.h"
#include "ParticleData.cuh"
#include "HOOMDMath.h"
#include <cufft.h>
#include "Index1D.h"

//! Define the step_one kernel
#ifndef __STOKES_CUH__
#define __STOKES_CUH__

//! Definition for comxplex variable storage
#ifdef SINGLE_PRECISION
#define CUFFTCOMPLEX cufftComplex
#else
#define CUFFTCOMPLEX cufftComplex
#endif


//! Kernel driver for the first part (no second part) of the Stokes update called by Stokes.cc
cudaError_t gpu_stokes_step_one(Scalar4 *d_pos,
                             Scalar4 *d_vel,
                             Scalar3 *d_accel,
                             int3 *d_image,
                             unsigned int *d_group_members,
                             unsigned int group_size,
                             const BoxDim& box,
                             Scalar deltaT,
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
			     Scalar4 *d_ewald1,
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
			     const Scalar *d_diameter,
			     Scalar cheb_error,
			     Scalar current_shear_rate);


#endif
