import hoomd;
from hoomd import _hoomd
from hoomd.md import _md
import hoomd.PSEv1
import os;
import math
hoomd.context.initialize('');

# Time stepping information
dt = 1e-3      # time step
tf = 10e0       # the final time of the simulation (in units of bare particle diffusion time)
nrun = tf / dt # number of steps

# Particle size
#
# Changing this won't change the PSE hydrodynamics, which assumes that all particles
# have radius = 1.0, and ignores HOOMD's size data. However, might be necessary if 
# hydrodynamic radius is different from other radii needed.
radius = 1.0
diameter = 2.0 * radius

# File output location
loc = 'Data/'
if not os.path.isdir( loc ):
        os.mkdir( loc )

# Simple cubic crystal of 1000 particles
N = 6400
L = 40 
n = math.ceil(N ** (1.0/3.0)) # number of particles along 1D
a = L / n # spacing between particles

# Create the box and particles
hoomd.init.create_lattice(unitcell=hoomd.lattice.sc(a=a),n=n)

# Shear function form, using sinusoidal oscillatory shear as example
#
# Options are: none (no shear. default if left unspecified in integrator call)
#              steady (steady shear)
#              sine (sinusoidal oscillatory shear)
#              chirp (chirp frequency sweep)
function_form = hoomd.PSEv1.shear_function.sine( dt = dt, shear_rate = 1.0, shear_freq = 1.0 )

# Set up PSE integrator
#
# Arguments to PSE integrator (default values given in parentheses):
# 	group -- group of particle to act on (should be all)
#	seed (1) -- Seed for the random number generator used in Brownian calculations
#       T (1.0) -- Temperature
#       xi (0.5) -- Ewald splitting parameter. Changing value will not affect results, only speed.
#       error (1E-3) -- Calculation error tolerance
#       function_form (none) -- Functional form for shearing. See above (or source code) for valid options. 
hoomd.md.integrate.mode_standard(dt=dt)
pse = hoomd.PSEv1.integrate.PSEv1( group = hoomd.group.all(), seed = 1, T = 1.0, xi = 0.5, error = 1E-3, function_form = function_form )

# Run the simulation
hoomd.run( nrun )


