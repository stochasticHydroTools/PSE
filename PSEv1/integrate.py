# First, we need to import the C++ module. It has the same name as this module (plugin_template) but with an underscore
# in front
from hoomd.PSEv1 import _PSEv1
from hoomd.PSEv1 import shear_function

# Next, since we are extending an integrator, we need to bring in the base class integrator and some other parts from
# hoomd_script
import hoomd
from hoomd import _hoomd
from hoomd import compute
from hoomd.md import _md
import math

## One step overdamped integration with hydrodynamic interactions
class PSEv1(hoomd.md.integrate._integration_method):
    ## Specifies the Stokes integrator
    #
    # \param group              Group of particles on which to apply this method.
    # \param T                  Temperature of the simulation (in energy units)
    # \param seed               Random seed to use for the run. Simulations that are identical, except for the seed, will follow
    #                             different trajectories.
    # \param xi                 Ewald splitting parameter
    # \param error    Relative error for all calculations
    # \param function_form  Functional form for shear
    # \param max_strain    Maximum box deformation for shear
    #
    #
    # T can be a variant type, allowing for temperature ramps in simulation runs.
    #
    # Internally, a compute.thermo is automatically specified and associated with \a group.
    
    def __init__(self, group, T, seed=0, xi = 0.5, error = 0.001, function_form = None, max_strain = 0.5, nlist_type = "cell" ):

  # Print the status of the initialization        
        hoomd.util.print_status_line();
        
        # initialize base class
        hoomd.md.integrate._integration_method.__init__(self);
        
        # setup the variant inputs
        T = hoomd.variant._setup_variant_input(T);
        
        # create the compute thermo
        compute._get_unique_thermo(group=group);
        
        # Real space neighborlist cutoff based on error estimate for spectral sums
        self.rcut = math.sqrt( - math.log( error ) ) / xi;
        # If this line is changed, remember to change in C++ code as well!!
        
        # initialize the reflected c++ class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            hoomd.context.msg.error("Sorry, we have not written CPU code for PSE RPY simulation. \n");
            raise RuntimeError('Error creating Stokes');
        else:
      
      # Create a neighborlist exclusively for real space interactions. Use cell lists by 
      # default, but also allow the user to specify
            if ( nlist_type.upper() == "CELL" ):

                cl_stokes = _hoomd.CellListGPU(hoomd.context.current.system_definition);
                hoomd.context.current.system.addCompute(cl_stokes, "stokes_cl")
                self.neighbor_list = _md.NeighborListGPUBinned(hoomd.context.current.system_definition, self.rcut, 0.4, cl_stokes);

            elif ( nlist_type.upper() == "TREE" ):

                self.neighbor_list = _md.NeighborListGPUTree(hoomd.context.current.system_definition, self.rcut, 0.4)

            elif ( nlist_type.upper() == "STENCIL" ):

                cl_stokes  = _hoomd.CellListGPU(hoomd.context.current.system_definition)
                hoomd.context.current.system.addCompute(cl_stokes, "stokes_cl")
                cls_stokes = _hoomd.CellListStencil( hoomd.context.current.system_definition, cl_stokes )
                hoomd.context.current.system.addCompute( cls_stokes, "stokes_cls")
                self.neighbor_list = _md.NeighborListGPUStencil(hoomd.context.current.system_definition, self.rcut, 0.4, cl_stokes, cls_stokes)

            else:
                hoomd.context.msg.error("Invalid neighborlist method specified. Valid options are: cell, tree, stencil. \n");
                raise RuntimeError('Error constructing neighborlist');

            # Set neighborlist properties
            self.neighbor_list.setEvery(1, True);
            hoomd.context.current.system.addCompute(self.neighbor_list, "stokes_nlist")
            self.neighbor_list.countExclusions();
        
            # Call the stokes integrator
            self.cpp_method = _PSEv1.Stokes(hoomd.context.current.system_definition, group.cpp_group, T.cpp_variant, seed, self.neighbor_list, xi, error);

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
            T = hoomd.variant._setup_variant_input(T);
            self.cpp_method.setT(T.cpp_variant);
        
        if function_form is not None:
            self.cpp_method.setShear(function_form.cpp_function, max_strain)

    ## Stop any shear
    def stop_shear(self, max_strain = 0.5):
        no_shear_function = shear_function.steady(dt = 0)
        self.cpp_method.setShear(no_shear_function.cpp_function, max_strain)


