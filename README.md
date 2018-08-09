# Positively Split Ewald (PSE)
PSE is a HOOMD plugin by Andrew M. Fiore containing a GPU implemention of the Positively Split Ewald
(PSE) algorithm for calculation of the Rotne-Prager-Yamakawa (RPY)
hydrodynamic mobility and stochastic thermal displacements. 
 
The theory behind the PSE method is described in the reference:

1. **Rapid Sampling of Stochastic Displacements in Brownian Dynamics
Simulations**, Andrew M. Fiore, Florencio Balboa Usabiaga, Aleksandar
Donev, and James W. Swan, The Journal of Chemical Physics, **146**,
124116 (2017).[DOI](http://doi.org/10.1063/1.4978242) [arXiv](https://arxiv.org/abs/1611.09322)


## Files that come in this template
 - doc/TUTORIAL.pdf : a tutorial to use PSE.
 - CMakeLists.txt   : main CMake configuration file for the plugin
 - FindHOOMD.cmake  : script to find a HOOMD-Blue installation to link against
 - README           : This file
 - PSEv1            : Directory containing C++ and CUDA source code that interacts with HOOMD. Also contains python UI level source code that drives the C++ module
 - cppmodule        : Directory containing C++ and CUDA source code that interacts with HOOMD
 - examples/run.py  : python example to use PSE.

## Software requirements

The PSE plugin requires the following additional software:
 - HOOMD, compiled with CUDA (tested with version 2.3.3). 
 - CUDA (tested with version 9.2).
 - LAPACKE (tested with version 3.6.1).
 - CBLAS (tested with version 3.6.1).

## Software Installation

HOOMD can be installed following the instructions given in the [documentation](http://hoomd-blue.readthedocs.io/en/stable/compiling.html). HOOMD must be compiled with CUDA enabled. It is recommended to use the following cmake command
```
cmake ../ -DCMAKE_INSTALL_PREFIX=${SOFTWARE_ROOT}/lib/python -DCMAKE_CXX_FLAGS=-march=native -DCMAKE_C_FLAGS=-march=native -DENABLE_CUDA=ON -DENABLE_MPI=ON
```
where `${SOFTWARE_ROOT}` is the path variable specifying the installation location for HOOMD.

LAPACKE and CBLAS can be install manually after downloading the source code from [netlib](http://www.netlib.org/lapacke) and [openblas](https://www.openblas.net) or from repositorities. In Ubuntu, the simplest method is via repository:
```
sudo apt-get install liblapack3 liblapack-dev liblapacke liblapacke-dev
sudo apt-get install libblas3 libblas-dev libopenblas-dev libatlas-base-dev
```

## Plugin Compilation
To compile this example plugin, follow steps similar to those in compiling HOOMD-Blue. The process of finding a HOOMD 
installation to link to will be fully automatic IF you have hoomd_install_dir/bin in your PATH when running cmake.

Note that plugins can only be built against a HOOMD build that has been installed via a package or compiled and then
installed via 'make install'. HOOMD must be built with CUDA enabled -DENABLE_CUDA=ON in order for the package to work.
Plugins can only be built against hoomd when it is built as a shared library.

From the root PSE folder do: 

```
$ mkdir plugin_build
$ cd plugin_build
$ cmake ../
$ make -j6
$ make install
```

If hoomd is not in your PATH, you can specify the root using

`$ cmake -DHOOMD_ROOT=/path/to/hoomd ../`

You can also provide to `cmake`  the location of `LAPACKE`, `LAPACK`, `CBLAS`,
`BLAS` and the `python` version with the options

```
$ cmake -DHOOMD_ROOT=/path/to/hoomd  \
-DCBLAS_LIBRARIES=/path/to/cblas     \
-DBLAS_LIBRARIES=/path/to/blas       \
-DLAPACKE_LIBRARIES=/path/to/lapacke \
-DLAPACK_LIBRARIES=/path/to/lapack   \
-DPYTHON_EXECUTABLE=`which python`   \
../
```
however, these options are unecessary if these libraries have been installed into the standard directories. 

By default, make install will install the plugin into

`${HOOMD_ROOT}/lib/python/hoomd/PSEv1`

This works if you have `make install`ed hoomd into your home directory. 

### Using the Plugin
A sample script demonstrating how the plugin is used can be found in examples/run.py. You can
call this script with the command
```
python3 run.py
```
