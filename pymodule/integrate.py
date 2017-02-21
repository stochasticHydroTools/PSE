# -- start license --
# Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
# (HOOMD-blue) Open Source Software License Copyright 2009-2014 The Regents of
# the University of Michigan All rights reserved.

# HOOMD-blue may contain modifications ("Contributions") provided, and to which
# copyright is held, by various Contributors who have granted The Regents of the
# University of Michigan the right to modify and/or distribute such Contributions.

# You may redistribute, use, and create derivate works of HOOMD-blue, in source
# and binary forms, provided you abide by the following conditions:

# * Redistributions of source code must retain the above copyright notice, this
# list of conditions, and the following disclaimer both in the code and
# prominently in any materials provided with the distribution.

# * Redistributions in binary form must reproduce the above copyright notice, this
# list of conditions, and the following disclaimer in the documentation and/or
# other materials provided with the distribution.

# * All publications and presentations based on HOOMD-blue, including any reports
# or published results obtained, in whole or in part, with HOOMD-blue, will
# acknowledge its use according to the terms posted at the time of submission on:
# http://codeblue.umich.edu/hoomd-blue/citations.html

# * Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
# http://codeblue.umich.edu/hoomd-blue/

# * Apart from the above required attributions, neither the name of the copyright
# holder nor the names of HOOMD-blue's contributors may be used to endorse or
# promote products derived from this software without specific prior written
# permission.

# Disclaimer

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
# WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# -- end license --

# this simple python interface just actiavates the c++ ExampleUpdater from cppmodule
# Check out any of the python code in lib/hoomd-python-module/hoomd_script for moreexamples

# First, we need to import the C++ module. It has the same name as this module (plugin_template) but with an underscore
# in front
import _PSEv1

# Next, since we are extending an updater, we need to bring in the base class updater and some other parts from 
# hoomd_script
from hoomd_script.integrate import _integration_method

from hoomd_script import util
from hoomd_script import globals
import hoomd
from hoomd_script import variant
from hoomd_script import compute
import math

## One step overdamped integration with hydrodynamic interactions

class PSEv1(_integration_method):
    ## Specifies the Stokes integrator
    #
    # \param group              Group of particles on which to apply this method.
    # \param T                  Temperature of the simulation (in energy units)
    # \param seed               Random seed to use for the run. Simulations that are identical, except for the seed, will follow
    #                             different trajectories.
    # \param limit              (optional) Enforce that no particle moves more than a distance of \a limit in a single time step
    # \param n_cheb             Order of Chebyshev approximation
    # \param cheb_reset_period  how often to recompute Chebyshev approximation
    # \param xi                 Ewald splitting parameter
    # \param Ewald_cut          Cutoff for real space interactions
    # \param Ewald_Nx           Number of grid nodes in x-direction
    # \param Ewald_Ny           Number of grid nodes in y-direction
    # \param Ewald_Nz           Number of grid nodes in z-direction
    # \param gaussm             Number of standard deviations contained in Gaussians
    # \param gaussP             Number of grid nodes in the support of each Gaussian
    #
    # \a T can be a variant type, allowing for temperature ramps in simulation runs.
    #
    # Internally, a compute.thermo is automatically specified and associated with \a group.
    #
    #
    # \warning If starting from a restart binary file, the energy of the reservoir will be reset to zero.
    # \b Examples:
    # \code
    # all = group.all();
    # integrate.bdnvt(group=all, T=1.0, seed=5)
    # integrator = integrate.bdnvt(group=all, T=1.0, seed=100)
    # integrate.bdnvt(group=all, T=1.0, limit=0.01, gamma_diam=1, tally=True)
    # typeA = group.type('A');
    # integrate.bdnvt(group=typeA, T=variant.linear_interp([(0, 4.0), (1e6, 1.0)]))
    # \endcode
    def __init__(self, group, T, limit=None, seed=0, xi = 0.5, error = 0.001):
        util.print_status_line();

        # initialize base class
        _integration_method.__init__(self);

	# setup the variant inputs
        T = variant._setup_variant_input(T);

        # create the compute thermo
        compute._get_unique_thermo(group=group);

	#self.rcut = Ewald_cut;
	self.rcut = math.sqrt( - math.log( error ) ) / xi;
	#self.rcut = 5.0;
	# If this line is changed, remember to change in C++ code as well!!

	# update the neighbor list
        # neighbor_list = pair._update_global_nlist(self.rcut)
        # neighbor_list.subscribe(lambda: self.rcut)

        # initialize the reflected c++ class
        if not globals.exec_conf.isCUDAEnabled():
            globals.msg.error("Sorry, we have not written CPU code for Stokesian Dynamics simulation. \n");
            raise RuntimeError('Error creating Stokes');
        else:
	    # Create a new neighbor list
	    cl_stokes = hoomd.CellListGPU(globals.system_definition);
	    globals.system.addCompute(cl_stokes, "stokes_cl")
	    self.neighbor_list = hoomd.NeighborListGPUBinned(globals.system_definition, self.rcut, 0.4, cl_stokes);
	    self.neighbor_list.setEvery(1, True);
	    globals.system.addCompute(self.neighbor_list, "stokes_nlist")
	    #self.neighbor_list.clearExclusions();
            #self.neighbor_list.setFilterBody(False);
            #self.neighbor_list.setFilterDiameter(False);
	    self.neighbor_list.countExclusions();

            self.cpp_method = _PSEv1.Stokes(globals.system_definition, group.cpp_group, T.cpp_variant, seed, self.neighbor_list, xi, error); 

        # set the limit
        if limit is not None:
            self.cpp_method.setLimit(limit);

        self.cpp_method.validateGroup()

	self.cpp_method.setParams()

    ## Changes parameters of an existing integrator
    # \param self self
    # \param limit (if set) New limit value to set. Removes the limit if limit is False
    # \param T Temperature
    #
    # To change the parameters of an existing integrator, you must save it in a variable when it is
    # specified, like so:
    # \code
    # integrator = integrate.nve(group=all)
    # \endcode
    #
    # \b Examples:
    # \code
    # integrator.set_params(limit=0.01)
    # integrator.set_params(limit=False)
    # \endcode
    def set_params(self, limit=None, T=None):
        util.print_status_line();
        self.check_initialization();

        # change the parameters
        if limit is not None:
            if limit == False:
                self.cpp_method.removeLimit();
            else:
                self.cpp_method.setLimit(limit);

	if T is not None:
            # setup the variant inputs
            T = variant._setup_variant_input(T);
            self.cpp_method.setT(T.cpp_variant);
