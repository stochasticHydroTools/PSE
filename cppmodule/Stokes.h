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

#include "hoomd/hoomd_config.h"
#include "IntegrationMethodTwoStep.h"
#include "Variant.h"
#include <cufft.h>

#include "NeighborList.h"
#include "ShearFunction.h"

#ifndef SINGLE_PRECISION
#define CUFFTCOMPLEX cufftComplex
#else
#define CUFFTCOMPLEX cufftComplex
#endif

#ifndef __STOKES_H__
#define __STOKES_H__

/*! \file Stokes.h
    \brief Declares the Stokes class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

//! Integrates the system forward considering hydrodynamic interactions by GPU
/*! Implements overdamped integration (one step) through IntegrationMethodTwoStep interface, runs on the GPU
*/

class Stokes : public IntegrationMethodTwoStep
    {
    public:
        //! Constructs the integration method and associates it with the system
        Stokes(boost::shared_ptr<SystemDefinition> sysdef,
                   boost::shared_ptr<ParticleGroup> group,
				   boost::shared_ptr<Variant> T,
				   unsigned int seed,
				   boost::shared_ptr<NeighborList> nlist,
				   Scalar xi,
				   Scalar error);
        virtual ~Stokes();

        //! Set a new temperature
        /*! \param T new temperature to set */
        void setT(boost::shared_ptr<Variant> T)
        {
        	m_T = T;
        }

        //! Performs the first step of the integration
        virtual void integrateStepOne(unsigned int timestep);

        //! Performs the second step of the integration
        virtual void integrateStepTwo(unsigned int timestep);

        //! Set the parameters for Ewald summation
        void setParams();

	//! Set the shear rate and shear frequency
    	void setShear(boost::shared_ptr<ShearFunction> shear_func, Scalar max_strain) {
      		m_shear_func = shear_func;
      		m_max_strain = max_strain;
  	}

    protected:

	boost::shared_ptr<Variant> m_T;   //!< The Temperature of the Stochastic Bath
        unsigned int m_seed;              //!< The seed for the RNG of the Stochastic Bath

        cufftHandle plan;       //!< Used for the Fast Fourier Transformations performed on the GPU

        boost::shared_ptr<NeighborList> m_nlist;    //!< The neighborlist to use for the computation

	boost::shared_ptr<ShearFunction> m_shear_func; //!< mutable shared pointer towards a ShearFunction object
	Scalar m_max_strain; //!< Maximum total strain before box resizing

        Scalar m_xi;                   //!< ewald splitting parameter xi
        Scalar m_ewald_cut;            //!< Real space cutoff
        GPUArray<Scalar4> m_ewaldC1;   //!< Real space Ewald coefficients table
        int m_ewald_n;                 //!< Number of entries in table of Ewald coefficients
        Scalar m_ewald_dr;             //!< Real space Ewald table spacing

	Scalar m_self; //!< self piece

        int m_Nx;  //!< Number of grid points in x direction
        int m_Ny;  //!< Number of grid points in y direction
        int m_Nz;  //!< Number of grid points in z direction

        GPUArray<Scalar4> m_gridk;        //!< k-vectors for each grid point
        GPUArray<CUFFTCOMPLEX> m_gridX;   //!< x component of the grid based force
        GPUArray<CUFFTCOMPLEX> m_gridY;   //!< x component of the grid based force
        GPUArray<CUFFTCOMPLEX> m_gridZ;   //!< x component of the grid based force

        Scalar m_gaussm;  //!< Gaussian width in standard deviations for wave space spreading/contraction
        int m_gaussP;     //!< Number of points in each dimension for Gaussian support
        Scalar m_eta;     //!< Gaussian spreading parameter
        Scalar3 m_gridh;  //!< Size of the grid box in 3 direction

        int m_m_Lanczos;       //!< Number of Lanczos Iterations to use for calculation of Brownian displacement

        Scalar m_error;  //!< Error tolerance for all calculations

    };

//! Exports the Stokes class to python
void export_Stokes();

#endif
