from hoomd_script import *
from hoomd_plugins import PSEv1
import math
import random
import os

#Simulation parameters
dt = 1e-4    # time step size
nsteps = 1E4 # number of time steps
T=1          # temperature
radius = 1.0 # particle radius

# Read in system
system = init.create_random(N=32000, phi_p=0.10, min_dist=0.90)

# Set radius to 1
for p in system.particles:
    p.diameter = 2.0 * radius

# Output directory
if ( not os.path.isdir( 'data' ) ):
    os.mkdir('data')

# Simulation parameters
xi = 0.5
tol = 1E-3

# Define the hard sphere potential for stokes simulation
def hs_potential_stokes(r, rmin, rmax, coeff1, coeff2, coeff3):
    V = 8.0 * coeff2 / 3.0 * coeff1 * ( 2.0 * coeff2 * math.log( 2.0 * coeff2 / r ) + ( r - 2.0 * coeff2 ) )
    F = 8.0 * coeff2 / 3.0 * coeff1 / r * ( 2.0 * coeff2 - r )
    return (V, F)

table_hs_stokes = pair.table(width=1000)
table_hs_stokes.pair_coeff.set( 'A', 'A', func=hs_potential_stokes, rmin=0.001, rmax=radius+radius, coeff=dict(coeff1=1/dt, coeff2=radius, coeff3=radius ) )

# Set up integrator
all =group.all()
integrate.mode_standard(dt = dt)

PSEv1.integrate.PSEv1(group=all, seed = 0, T = 1, xi = 0.5, error = 10**(-3.0))

# Trajectory output
dcd = dump.dcd(filename='data/motion.dcd',period=1000,overwrite=True)
xml = dump.xml(filename='data/particles',period = 1000)

# Stress output
compute.thermo(group=all)
stress = analyze.log(filename = 'data/stress.log', quantities = ['pressure_xx','pressure_yy','pressure_zz','pressure_xy','pressure_yz','pressure_xz'], period=100, header_prefix='#',overwrite=True)

# Run
run(nsteps)

