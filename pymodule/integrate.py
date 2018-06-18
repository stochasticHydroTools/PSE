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
import shear_function

## One step overdamped integration with hydrodynamic interactions

class PSEv1(_integration_method):
    ## Specifies the Stokes integrator
    #
    # \param group              Group of particles on which to apply this method.
    # \param T                  Temperature of the simulation (in energy units)
    # \param seed               Random seed to use for the run. Simulations that are identical, except for the seed, will follow
    #                             different trajectories.
    # \param xi                 Ewald splitting parameter
    # \param error		Relative error for all calculations
    # \param function_form	Functional form for shear
    # \param max_strain		Maximum box deformation for shear
    #
    #
    # T can be a variant type, allowing for temperature ramps in simulation runs.
    #
    # Internally, a compute.thermo is automatically specified and associated with \a group.
    
    def __init__(self, group, T, seed=0, xi = 0.5, error = 0.001, function_form = None, max_strain = 0.5, nlist_type = "cell" ):
        
	util.print_status_line();

        # initialize base class
        _integration_method.__init__(self);

	# setup the variant inputs
        T = variant._setup_variant_input(T);

        # create the compute thermo
        compute._get_unique_thermo(group=group);

	# Real space neighborlist cutoff based on error estimate for spectral sums
	self.rcut = math.sqrt( - math.log( error ) ) / xi;
	# If this line is changed, remember to change in C++ code as well!!

        # initialize the reflected c++ class
        if not globals.exec_conf.isCUDAEnabled():
            globals.msg.error("Sorry, we have not written CPU code for PSE RPY simulation. \n");
            raise RuntimeError('Error creating Stokes');
        else:
	    
	    # Create a neighborlist exclusively for real space interactions. Use cell lists by 
	    # default, but also allow the user to specify
            if ( nlist_type.upper() == "CELL" ):

	    	cl_stokes = hoomd.CellListGPU(globals.system_definition);
	    	globals.system.addCompute(cl_stokes, "stokes_cl")
	    	self.neighbor_list = hoomd.NeighborListGPUBinned(globals.system_definition, self.rcut, 0.4, cl_stokes);

	    elif ( nlist_type.upper() == "TREE" ):

		self.neighbor_list = hoomd.NeighborListGPUTree(globals.system_definition, self.rcut, 0.4)

	    elif ( nlist_type.upper() == "STENCIL" ):

            	cl_stokes  = hoomd.CellListGPU(globals.system_definition)
            	globals.system.addCompute(cl_stokes, "stokes_cl")
            	cls_stokes = hoomd.CellListStencil( globals.system_definition, cl_stokes )
            	globals.system.addCompute( cls_stokes, "stokes_cls")
            	self.neighbor_list = hoomd.NeighborListGPUStencil(globals.system_definition, self.rcut, 0.4, cl_stokes, cls_stokes)

	    else:
            	globals.msg.error("Invalid neighborlist method specified. Valid options are: cell, tree, stencil. \n");
            	raise RuntimeError('Error constructing neighborlist');

	    # Set neighborlist properties
	    self.neighbor_list.setEvery(1, True);
	    globals.system.addCompute(self.neighbor_list, "stokes_nlist")
	    self.neighbor_list.countExclusions();

	    # Call the stokes integrator
            self.cpp_method = _PSEv1.Stokes(globals.system_definition, group.cpp_group, T.cpp_variant, seed, self.neighbor_list, xi, error);

        self.cpp_method.validateGroup()

	if function_form is not None:
            self.cpp_method.setShear(function_form.cpp_function, max_strain)
        else:
            no_shear_function = shear_function.steady(dt = 0)
            self.cpp_method.setShear(no_shear_function.cpp_function, max_strain)

	self.cpp_method.setParams()

    ## Changes parameters of an existing integrator
    # \param self self
    # \param T Temperature
    #
    # To change the parameters of an existing integrator, you must save it in a variable when it is
    # specified, like so:
    # \code
    # integrator = integrate.nve(group=all)
    # \endcode
    
    def set_params(self, T=None, function_form = None, max_strain=0.5):
        util.print_status_line();
        self.check_initialization();

	if T is not None:
            # setup the variant inputs
            T = variant._setup_variant_input(T);
            self.cpp_method.setT(T.cpp_variant);

	if function_form is not None:
            self.cpp_method.setShear(function_form.cpp_function, max_strain)

    ## Stop any shear
    def stop_shear(self, max_strain = 0.5):
        no_shear_function = shear_function.steady(dt = 0)
        self.cpp_method.setShear(no_shear_function.cpp_function, max_strain)
