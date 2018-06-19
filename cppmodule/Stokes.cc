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

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4244 )
#endif

using namespace std;

#include <boost/python.hpp>
using namespace boost::python;
#include <boost/bind.hpp>
using namespace boost;

#include <vector>
#include <algorithm>

#include "Stokes.h"
#include "Stokes.cuh"
#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

/*! \file Stokes.cc
    \brief Contains code for the Stokes class
*/

/*!
	\param sysdef             SystemDefinition this method will act on. Must not be NULL.
    	\param group              The group of particles this integration method is to work on
	\param T                  temperature
	\param seed               seed for random number generator
	\param nlist              neighbor list
	\param xi                 Ewald parameter
	\param m_error            Tolerance for all calculations

*/
Stokes::Stokes(boost::shared_ptr<SystemDefinition> sysdef,
                       boost::shared_ptr<ParticleGroup> group,
					   boost::shared_ptr<Variant> T,
					   unsigned int seed,
					   boost::shared_ptr<NeighborList> nlist,
					   Scalar xi,
					   Scalar error)
					   : IntegrationMethodTwoStep(sysdef, group),
					   m_T(T),
					   m_seed(seed),
					   m_nlist(nlist),
					   m_xi(xi),
					   m_error(error)
    {
    m_exec_conf->msg->notice(5) << "Constructing Stokes" << endl;

	// Hash the User's Seed to make it less likely to be a low positive integer
	m_seed = m_seed * 0x12345677 + 0x12345; m_seed ^= (m_seed >> 16); m_seed *= 0x45679;

	// only one GPU is supported
	if (!m_exec_conf->isCUDAEnabled())
	{
		m_exec_conf->msg->error() << "Creating a Stokes when CUDA is disabled" << endl;
		throw std::runtime_error("Error initializing Stokes");
	}

    }

//! Destructor for the Stokes class
Stokes::~Stokes()
    {
    m_exec_conf->msg->notice(5) << "Destroying Stokes" << endl;
	cufftDestroy(plan);
    }


/*!
	Set the parameters for Spectral Ewald Method
*/
void Stokes::setParams()
{
	// Try two Lanczos iterations to start (number of iterations will adapt as needed)
	m_m_Lanczos = 2;

	// Real space cutoff
	m_ewald_cut = sqrtf( - logf( m_error ) ) / m_xi;

	// Number of grid points
	int kmax = int( 2.0 * sqrtf( - logf( m_error ) ) * m_xi ) + 1;

	const BoxDim& box = m_pdata->getBox(); // Only for box not changing with time.
	Scalar3 L = box.getL();

	m_Nx = int( kmax * L.x / (2.0 * 3.1415926536 ) * 2.0 ) + 1;
	m_Ny = int( kmax * L.y / (2.0 * 3.1415926536 ) * 2.0 ) + 1;
	m_Nz = int( kmax * L.z / (2.0 * 3.1415926536 ) * 2.0 ) + 1;

	// Get list of int values between 8 and 512 that can be written as
	// 	(2^a)*(3^b)*(5^c)
	// Then sort list from low to high
	std::vector<int> Mlist;
	for ( int ii = 0; ii < 10; ++ii ){
		int pow2 = 1;
		for ( int i = 0; i < ii; ++i ){
			pow2 *= 2;
		}
		for ( int jj = 0; jj < 6; ++jj ){
			int pow3 = 1;
			for ( int j = 0; j < jj; ++j ){
				pow3 *= 3;
			}
			for ( int kk = 0; kk < 4; ++kk ){
				int pow5 = 1;
				for ( int k = 0; k < kk; ++k ){
					pow5 *= 5;
				}
				int Mcurr = pow2 * pow3 * pow5;
				if ( Mcurr >= 8 && Mcurr <= 512 ){
					Mlist.push_back(Mcurr);
				}
			}
		}
	}
	std::sort(Mlist.begin(), Mlist.end());
	const int nmult = Mlist.size(); // 62 such values should exist

	// Compute the number of grid points in each direction
	//
	// Number of grid points should be a power of 2,3,5 for most efficient FFTs
	for ( int ii = 0; ii < nmult; ++ii ){
		if (m_Nx <= Mlist[ii]){
			 m_Nx = Mlist[ii];
			break;
		}
	}
	for ( int ii = 0; ii < nmult; ++ii ){
		if (m_Ny <= Mlist[ii]){
			m_Ny = Mlist[ii];
			break;
		}
	}
	for ( int ii = 0; ii < nmult; ++ii ){
		if (m_Nz <= Mlist[ii]){
			m_Nz = Mlist[ii];
			break;
		}
	}

	if ( m_Nx * m_Ny * m_Nz > 512*512*512 ){

		printf("Requested Number of Fourier Nodes Exceeds Max Dimension of 512^3\n");
		printf("Mx = %i \n", m_Nx);
		printf("My = %i \n", m_Ny);
		printf("Mz = %i \n", m_Nz);
		printf("Mx*My*Mz = %i \n", m_Nx * m_Ny * m_Nz);

		exit(EXIT_FAILURE);
	}

	// Maximum eigenvalue of A'*A to scale P
	Scalar gamma = m_max_strain;
	Scalar gamma2 = gamma*gamma;
	Scalar lambda = 1.0 + gamma2/2.0 + gamma*sqrtf(1.0 + gamma2/4.0);

	// Grid spacing
	m_gridh = L / make_scalar3(m_Nx,m_Ny,m_Nz);

	// Parameters for the Spectral Ewald Method (Lindbo and Tornberg, J. Comp. Phys., 2011)
	m_gaussm = 1.0;
	while ( erfcf( m_gaussm / sqrtf(2.0*lambda) ) > m_error ){
	    m_gaussm = m_gaussm + 0.01;
	}
	m_gaussP = int( m_gaussm*m_gaussm / 3.1415926536 )  + 1;

	if (m_gaussP > m_Nx) m_gaussP = m_Nx; // Can't be supported beyond grid
	if (m_gaussP > m_Ny) m_gaussP = m_Ny;
	if (m_gaussP > m_Nz) m_gaussP = m_Nz;
	Scalar w = m_gaussP*m_gridh.x / 2.0;	               // Gaussian width in simulation units
	Scalar xisq  = m_xi * m_xi;
	m_eta = (2.0*w/m_gaussm)*(2.0*w/m_gaussm) * ( xisq );  // Gaussian splitting parameter

	// Print summary to command line output
	printf("\n");
	printf("\n");
	m_exec_conf->msg->notice(2) << "--- NUFFT Hydrodynamics Statistics ---" << endl;
	m_exec_conf->msg->notice(2) << "Mx: " << m_Nx << endl;
	m_exec_conf->msg->notice(2) << "My: " << m_Ny << endl;
	m_exec_conf->msg->notice(2) << "Mz: " << m_Nz << endl;
	m_exec_conf->msg->notice(2) << "rcut: " << m_ewald_cut << endl;
	m_exec_conf->msg->notice(2) << "Points per radius (x,y,z): " << m_Nx / L.x << ", " << m_Ny / L.y << ", " << m_Nz / L.z << endl;
	m_exec_conf->msg->notice(2) << "--- Gaussian Spreading Parameters ---"  << endl;
	m_exec_conf->msg->notice(2) << "gauss_m: " << m_gaussm << endl;
        m_exec_conf->msg->notice(2) << "gauss_P: " << m_gaussP << endl;
	m_exec_conf->msg->notice(2) << "gauss_eta: " << m_eta << endl;
	m_exec_conf->msg->notice(2) << "gauss_w: " << w << endl;
	m_exec_conf->msg->notice(2) << "gauss_gridh (x,y,z): " << L.x/m_Nx << ", " << L.y/m_Ny << ", " << L.z/m_Nz << endl;
	printf("\n");
	printf("\n");

	// Create plan for CUFFT on the GPU
	cufftPlan3d(&plan, m_Nx, m_Ny, m_Nz, CUFFT_C2C);

	// Prepare GPUArrays for grid vectors and gridded forces
	GPUArray<Scalar4> n_gridk(m_Nx*m_Ny*m_Nz, m_exec_conf);
	m_gridk.swap(n_gridk);
	GPUArray<CUFFTCOMPLEX> n_gridX(m_Nx*m_Ny*m_Nz, m_exec_conf);
	m_gridX.swap(n_gridX);
	GPUArray<CUFFTCOMPLEX> n_gridY(m_Nx*m_Ny*m_Nz, m_exec_conf);
	m_gridY.swap(n_gridY);
	GPUArray<CUFFTCOMPLEX> n_gridZ(m_Nx*m_Ny*m_Nz, m_exec_conf);
	m_gridZ.swap(n_gridZ);

	// Get list of reciprocal space vectors, and scaling factor for the wave space calculation at each grid point
	ArrayHandle<Scalar4> h_gridk(m_gridk, access_location::host, access_mode::readwrite);
	for (int i = 0; i < m_Nx; i++) {
		for (int j = 0; j < m_Ny; j++) {
			for (int k = 0; k < m_Nz; k++) {

				// Index into grid vector storage array
				int idx = i * m_Ny*m_Nz + j * m_Nz + k;

				// k goes from -N/2 to N/2
				h_gridk.data[idx].x = 2.0*3.1415926536 * ((i < ( m_Nx + 1 ) / 2) ? i : i - m_Nx) / L.x;
				h_gridk.data[idx].y = 2.0*3.1415926536 * ((j < ( m_Ny + 1 ) / 2) ? j : j - m_Ny) / L.y;
				h_gridk.data[idx].z = 2.0*3.1415926536 * ((k < ( m_Nz + 1 ) / 2) ? k : k - m_Nz) / L.z;

				// k dot k
				Scalar k2 = h_gridk.data[idx].x*h_gridk.data[idx].x + h_gridk.data[idx].y*h_gridk.data[idx].y + h_gridk.data[idx].z*h_gridk.data[idx].z;

				// Scaling factor used in wave space sum
				//
				// Can't include k=0 term in the Ewald sum
				if (i == 0 && j == 0 && k == 0){
					h_gridk.data[idx].w = 0;
				}
				else{
					// Have to divide by Nx*Ny*Nz to normalize the FFTs
					h_gridk.data[idx].w = 6.0*3.1415926536 * (1.0 + k2/4.0/xisq) * expf( -(1-m_eta) * k2/4.0/xisq ) / ( k2 ) / Scalar( m_Nx*m_Ny*m_Nz );
				}

			}
		}
	}

	// Store the coefficients for the real space part of Ewald summation
	//
	// Will precompute scaling factors for real space component of summation for a given
	//     discretization to speed up GPU calculations
	m_ewald_dr = 0.001; 		           // Distance resolution
	m_ewald_n = m_ewald_cut / m_ewald_dr - 1;  // Number of entries in tabulation

	double dr = 0.0010000000000000;

	// Get particle diameter and self bit
	ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::read);
	Scalar diameter = h_diameter.data[0];

        Scalar pi12 = 1.77245385091;
        Scalar aa = diameter / 2.0;
	Scalar axi = aa * m_xi;
	Scalar axi2 = axi * axi;
        m_self = (1. + 4.*pi12*axi*erfc(2.*axi) - exp(-4.*axi2))/(4.*pi12*axi*aa);

	// Allocate storage for real space Ewald table
	int nR = m_ewald_n + 1; // number of entries in ewald table
	GPUArray<Scalar4> n_ewaldC1( nR, m_exec_conf);
	m_ewaldC1.swap(n_ewaldC1);
	ArrayHandle<Scalar4> h_ewaldC1(m_ewaldC1, access_location::host, access_mode::readwrite);

	// Functions are complicated so calculation should be done in double precision, then truncated to single precision
	// in order to ensure accurate evaluation
	double xi  = m_xi;
	double Pi = 3.141592653589793;
	double a = aa;

	// Fill tables
	for ( int kk = 0; kk < nR; kk++ )
	{

		// Initialize entries
		h_ewaldC1.data[ kk ].x = 0.0; // UF1 at r
		h_ewaldC1.data[ kk ].y = 0.0; // UF2 at r
		h_ewaldC1.data[ kk ].z = 0.0; // UF1 at r + dr
		h_ewaldC1.data[ kk ].w = 0.0; // UF2 at r + dr

		// Distance for current entry
		double r = double( kk ) * dr + dr;
		double Imrr = 0, rr = 0;

		// Expression have been simplified assuming no overlap, touching, and overlap
		if ( r > 2.0*a ){

			Imrr = -pow(a,-1) + (pow(a,2)*pow(r,-3))/2. + (3*pow(r,-1))/4. + (3*erfc(r*xi)*pow(a,-2)*pow(r,-3)*(-12*pow(r,4) + pow(xi,-4)))/128. +
   pow(a,-2)*((9*r)/32. - (3*pow(r,-3)*pow(xi,-4))/128.) +
   (erfc((2*a + r)*xi)*(128*pow(a,-1) + 64*pow(a,2)*pow(r,-3) + 96*pow(r,-1) + pow(a,-2)*(36*r - 3*pow(r,-3)*pow(xi,-4))))/256. +
   (erfc(2*a*xi - r*xi)*(128*pow(a,-1) - 64*pow(a,2)*pow(r,-3) - 96*pow(r,-1) + pow(a,-2)*(-36*r + 3*pow(r,-3)*pow(xi,-4))))/
    256. + (3*exp(-(pow(r,2)*pow(xi,2)))*pow(a,-2)*pow(Pi,-0.5)*pow(r,-2)*pow(xi,-3)*(1 + 6*pow(r,2)*pow(xi,2)))/64. +
   (exp(-(pow(2*a + r,2)*pow(xi,2)))*pow(a,-2)*pow(Pi,-0.5)*pow(r,-3)*pow(xi,-3)*
      (8*r*pow(a,2)*pow(xi,2) - 16*pow(a,3)*pow(xi,2) + a*(2 - 28*pow(r,2)*pow(xi,2)) - 3*(r + 6*pow(r,3)*pow(xi,2))))/128. +
   (exp(-(pow(-2*a + r,2)*pow(xi,2)))*pow(a,-2)*pow(Pi,-0.5)*pow(r,-3)*pow(xi,-3)*
      (8*r*pow(a,2)*pow(xi,2) + 16*pow(a,3)*pow(xi,2) + a*(-2 + 28*pow(r,2)*pow(xi,2)) - 3*(r + 6*pow(r,3)*pow(xi,2))))/128.;

			rr = -pow(a,-1) - pow(a,2)*pow(r,-3) + (3*pow(r,-1))/2. + (3*pow(a,-2)*pow(r,-3)*(4*pow(r,4) + pow(xi,-4)))/64. +
   (erfc(2*a*xi - r*xi)*(64*pow(a,-1) + 64*pow(a,2)*pow(r,-3) - 96*pow(r,-1) + pow(a,-2)*(-12*r - 3*pow(r,-3)*pow(xi,-4))))/128. +
   (erfc((2*a + r)*xi)*(64*pow(a,-1) - 64*pow(a,2)*pow(r,-3) + 96*pow(r,-1) + pow(a,-2)*(12*r + 3*pow(r,-3)*pow(xi,-4))))/128. +
   (3*exp(-(pow(r,2)*pow(xi,2)))*pow(a,-2)*pow(Pi,-0.5)*pow(r,-2)*pow(xi,-3)*(-1 + 2*pow(r,2)*pow(xi,2)))/32. -
   ((2*a + 3*r)*exp(-(pow(-2*a + r,2)*pow(xi,2)))*pow(a,-2)*pow(Pi,-0.5)*pow(r,-3)*pow(xi,-3)*
      (-1 - 8*a*r*pow(xi,2) + 8*pow(a,2)*pow(xi,2) + 2*pow(r,2)*pow(xi,2)))/64. +
   ((2*a - 3*r)*exp(-(pow(2*a + r,2)*pow(xi,2)))*pow(a,-2)*pow(Pi,-0.5)*pow(r,-3)*pow(xi,-3)*
      (-1 + 8*a*r*pow(xi,2) + 8*pow(a,2)*pow(xi,2) + 2*pow(r,2)*pow(xi,2)))/64. -
   (3*erfc(r*xi)*pow(a,-2)*pow(r,-3)*pow(xi,-4)*(1 + 4*pow(r,4)*pow(xi,4)))/64.;

		}
		else if ( r == 2.0*a ){

			Imrr = -(pow(a,-5)*(3 + 16*a*xi*pow(Pi,-0.5))*pow(xi,-4))/2048. + (3*erfc(2*a*xi)*pow(a,-5)*(-192*pow(a,4) + pow(xi,-4)))/1024. +
   erfc(4*a*xi)*(pow(a,-1) - (3*pow(a,-5)*pow(xi,-4))/2048.) +
   (exp(-16*pow(a,2)*pow(xi,2))*pow(a,-4)*pow(Pi,-0.5)*pow(xi,-3)*(-1 - 64*pow(a,2)*pow(xi,2)))/256. +
   (3*exp(-4*pow(a,2)*pow(xi,2))*pow(a,-4)*pow(Pi,-0.5)*pow(xi,-3)*(1 + 24*pow(a,2)*pow(xi,2)))/256.;

			rr = (pow(a,-5)*(3 + 16*a*xi*pow(Pi,-0.5))*pow(xi,-4))/1024. + erfc(2*a*xi)*((-3*pow(a,-1))/8. - (3*pow(a,-5)*pow(xi,-4))/512.) +
   erfc(4*a*xi)*(pow(a,-1) + (3*pow(a,-5)*pow(xi,-4))/1024.) +
   (exp(-16*pow(a,2)*pow(xi,2))*pow(a,-4)*pow(Pi,-0.5)*pow(xi,-3)*(1 - 32*pow(a,2)*pow(xi,2)))/128. +
   (3*exp(-4*pow(a,2)*pow(xi,2))*pow(a,-4)*pow(Pi,-0.5)*pow(xi,-3)*(-1 + 8*pow(a,2)*pow(xi,2)))/128.;

		}
		else if ( r < 2*a){

			Imrr = (-9*r*pow(a,-2))/32. + pow(a,-1) - (pow(a,2)*pow(r,-3))/2. - (3*pow(r,-1))/4. +
   (3*erfc(r*xi)*pow(a,-2)*pow(r,-3)*(-12*pow(r,4) + pow(xi,-4)))/128. +
   (erfc((-2*a + r)*xi)*(-128*pow(a,-1) + 64*pow(a,2)*pow(r,-3) + 96*pow(r,-1) + pow(a,-2)*(36*r - 3*pow(r,-3)*pow(xi,-4))))/
    256. + (erfc((2*a + r)*xi)*(128*pow(a,-1) + 64*pow(a,2)*pow(r,-3) + 96*pow(r,-1) + pow(a,-2)*(36*r - 3*pow(r,-3)*pow(xi,-4))))/
    256. + (3*exp(-(pow(r,2)*pow(xi,2)))*pow(a,-2)*pow(Pi,-0.5)*pow(r,-2)*pow(xi,-3)*(1 + 6*pow(r,2)*pow(xi,2)))/64. +
   (exp(-(pow(2*a + r,2)*pow(xi,2)))*pow(a,-2)*pow(Pi,-0.5)*pow(r,-3)*pow(xi,-3)*
      (8*r*pow(a,2)*pow(xi,2) - 16*pow(a,3)*pow(xi,2) + a*(2 - 28*pow(r,2)*pow(xi,2)) - 3*(r + 6*pow(r,3)*pow(xi,2))))/128. +
   (exp(-(pow(-2*a + r,2)*pow(xi,2)))*pow(a,-2)*pow(Pi,-0.5)*pow(r,-3)*pow(xi,-3)*
      (8*r*pow(a,2)*pow(xi,2) + 16*pow(a,3)*pow(xi,2) + a*(-2 + 28*pow(r,2)*pow(xi,2)) - 3*(r + 6*pow(r,3)*pow(xi,2))))/128.;

			rr = ((2*a + 3*r)*pow(a,-2)*pow(2*a - r,3)*pow(r,-3))/16. +
   (erfc((-2*a + r)*xi)*(-64*pow(a,-1) - 64*pow(a,2)*pow(r,-3) + 96*pow(r,-1) + pow(a,-2)*(12*r + 3*pow(r,-3)*pow(xi,-4))))/128. +
   (erfc((2*a + r)*xi)*(64*pow(a,-1) - 64*pow(a,2)*pow(r,-3) + 96*pow(r,-1) + pow(a,-2)*(12*r + 3*pow(r,-3)*pow(xi,-4))))/128. +
   (3*exp(-(pow(r,2)*pow(xi,2)))*pow(a,-2)*pow(Pi,-0.5)*pow(r,-2)*pow(xi,-3)*(-1 + 2*pow(r,2)*pow(xi,2)))/32. -
   ((2*a + 3*r)*exp(-(pow(-2*a + r,2)*pow(xi,2)))*pow(a,-2)*pow(Pi,-0.5)*pow(r,-3)*pow(xi,-3)*
      (-1 - 8*a*r*pow(xi,2) + 8*pow(a,2)*pow(xi,2) + 2*pow(r,2)*pow(xi,2)))/64. +
   ((2*a - 3*r)*exp(-(pow(2*a + r,2)*pow(xi,2)))*pow(a,-2)*pow(Pi,-0.5)*pow(r,-3)*pow(xi,-3)*
      (-1 + 8*a*r*pow(xi,2) + 8*pow(a,2)*pow(xi,2) + 2*pow(r,2)*pow(xi,2)))/64. -
   (3*erfc(r*xi)*pow(a,-2)*pow(r,-3)*pow(xi,-4)*(1 + 4*pow(r,4)*pow(xi,4)))/64.;

		}

		// Save values to table
		h_ewaldC1.data[ kk ].x = Scalar( Imrr ); // UF1
		h_ewaldC1.data[ kk ].y = Scalar( rr );   // UF2

	} // kk loop over distances

	// Both pieces of UF data for faster interpolation (r and r+dr stored in same Scalar4)
	for ( int kk = 0; kk < (nR-1); kk++ ){

		int offset1 = kk;
		int offset2 = (kk+1);

		h_ewaldC1.data[ offset1 ].z = h_ewaldC1.data[ offset2 ].x;
		h_ewaldC1.data[ offset1 ].w = h_ewaldC1.data[ offset2 ].y;
	}

}

/*! \param timestep Current time step
\post Particle positions and velocities are moved forward to timestep+1
*/
void Stokes::integrateStepOne(unsigned int timestep)
{

	// Recompute neighborlist ( if needed )
	m_nlist->compute(timestep);

	// access the neighbor list
	ArrayHandle<unsigned int> d_n_neigh(m_nlist->getNNeighArray(), access_location::device, access_mode::read);
	ArrayHandle<unsigned int> d_nlist(m_nlist->getNListArray(), access_location::device, access_mode::read);
	ArrayHandle<unsigned int> d_headlist(m_nlist->getHeadList(), access_location::device, access_mode::read);

	// Consistency check
	unsigned int group_size = m_group->getNumMembers();
	assert(group_size <= m_pdata->getN());
	if (group_size == 0)
		return;

	// Get particle forces
	const GPUArray< Scalar4 >& net_force = m_pdata->getNetForce();

	// profile this step
	if (m_prof)
		m_prof->push(m_exec_conf, "Stokes step 1 (no step 2)");

	// access all the needed data
	ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::readwrite);
	ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::readwrite);
	ArrayHandle<Scalar3> d_accel(m_pdata->getAccelerations(), access_location::device, access_mode::readwrite);
	ArrayHandle<Scalar4> d_net_force(net_force, access_location::device, access_mode::read);
	ArrayHandle<int3> d_image(m_pdata->getImages(), access_location::device, access_mode::readwrite);

	// Read diameters for calculation
	ArrayHandle<Scalar> d_diameter(m_pdata->getDiameters(), access_location::device, access_mode::read);

	BoxDim box = m_pdata->getBox();
	ArrayHandle< unsigned int > d_index_array(m_group->getIndexArray(), access_location::device, access_mode::read);

	// Grid vectors
	ArrayHandle<Scalar4> d_gridk(m_gridk, access_location::device, access_mode::readwrite);
	ArrayHandle<CUFFTCOMPLEX> d_gridX(m_gridX, access_location::device, access_mode::readwrite);
	ArrayHandle<CUFFTCOMPLEX> d_gridY(m_gridY, access_location::device, access_mode::readwrite);
	ArrayHandle<CUFFTCOMPLEX> d_gridZ(m_gridZ, access_location::device, access_mode::readwrite);

	// Real space interaction tabulation
	ArrayHandle<Scalar4> d_ewaldC1(m_ewaldC1, access_location::device, access_mode::read);

        // Calculate the shear rate of the current timestep
        Scalar current_shear_rate = m_shear_func -> getShearRate(timestep);

	// perform the update on the GPU
	gpu_stokes_step_one(d_pos.data,
		d_vel.data,
		d_accel.data,
		d_image.data,
		d_index_array.data,
		group_size,
		box,
		m_deltaT,
		256,
		d_net_force.data,
		m_T->getValue(timestep),
		timestep,
		m_seed,
		m_xi,
		m_eta,
		m_ewald_cut,
		m_ewald_dr,
		m_ewald_n,
		d_ewaldC1.data,
		m_self,
		d_gridk.data,
		d_gridX.data,
		d_gridY.data,
		d_gridZ.data,
		plan,
		m_Nx,
		m_Ny,
		m_Nz,
		d_n_neigh.data,
		d_nlist.data,
		d_headlist.data,
		m_m_Lanczos,
		m_pdata->getN(),
		m_gaussP,
		m_gridh,
		d_diameter.data,
		m_error,
		current_shear_rate);

	if (m_exec_conf->isCUDAErrorCheckingEnabled())
		CHECK_CUDA_ERROR();

	// done profiling
	if (m_prof)
		m_prof->pop(m_exec_conf);

}

/*! \param timestep Current time step
\post Nothing is done.
*/
void Stokes::integrateStepTwo(unsigned int timestep)
{
}

void export_Stokes()
    {
    class_<Stokes, boost::shared_ptr<Stokes>, bases<IntegrationMethodTwoStep>, boost::noncopyable>
		("Stokes", init< boost::shared_ptr<SystemDefinition>, boost::shared_ptr<ParticleGroup>, boost::shared_ptr<Variant>, unsigned int, boost::shared_ptr<NeighborList>, Scalar, Scalar >() )
		.def("setT", &Stokes::setT)
		.def("setParams", &Stokes::setParams)
                .def("setShear", &Stokes::setShear)

        ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif
