# Positively Split Ewald (PSE)
PSE is a HOOMD plugin by Andrew M. Fiore containing a GPU implemention of the Positively Split Ewald
(PSE) algorithm for calculation of the Rotne-Prager-Yamakawa (RPY)
hydrodynamic mobility and stochastic thermal displacements. 
 
The theory behind the PSE method is described in the reference:

1. **Rapid Sampling of Stochastic Displacements in Brownian Dynamics
Simulations**, Andrew M. Fiore, Florencio Balboa Usabiaga, Aleksandar
Donev, and James W. Swan, 2016. [arXiv](https://arxiv.org/abs/1611.09322)


## Files that come in this template
 - doc/TUTORIAL.pdf : a tutorial to use PSE.
 - CMakeLists.txt   : main CMake configuration file for the plugin
 - FindHOOMD.cmake  : script to find a HOOMD-Blue installation to link against
 - README           : This file
 - cppmodule        : Directory containing C++ and CUDA source code that interacts with HOOMD
 - pymodule         : Directory containing python UI level source code that drives the C++ module
 

## Compilation
To compile this example plugin, follow steps similar to those in compiling HOOMD-Blue. The process of finding a HOOMD 
installation to link to will be fully automatic IF you have hoomd_install_dir/bin in your PATH when running ccmake.

Note that plugins can only be built against a hoomd build that has been installed via a package or compiled and then
installed via 'make install'. Plugins can only be built against hoomd when it is built as a shared library.

```
$ mkdir plugin_build
$ cd plugin_build
$ ccmake /path/to/plugin_template_cpp 
(follow normal cmake steps)
$ make -j6
$ make install
```

If hoomd is not in your PATH, you can specify the root using

`$ ccmake /path/to/plugin_template -DHOOMD_ROOT=/path/to/hoomd`

where `${HOOMD_ROOT}/bin/hoomd` is where the hoomd executable is installed

By default, make install will install the plugin into

`${HOOMD_ROOT}/lib/hoomd/python_module/hoomd_plugins/plugin_template`

This works if you have `make install`ed hoomd into your home directory. 

If hoomd is installed in a system directory (such as via an rpm or deb package), then you can still use plugins.
Delete the plugin_build directory and start over. Set the environment
variable `HOOMD_PLUGINS_DIR` in your `.bash_profile`, as an example

`export HOOMD_PLUGINS_DIR=${HOME}/hoomd_plugins`  

When running ccmake, add `-DHOOMD_PLUGINS_DIR=${HOOMD_PLUGINS_DIR}`
to the options, that is it

 `ccmake /path/to/plugin_template_cpp
 -DHOOMD_PLUGINS_DIR=${HOOMD_PLUGINS_DIR}`

Now, `make install` will install the plugins into `${HOOMD_PLUGINS_DIR}` and hoomd, when launched, will look there
for the plugins.

The plugin can now be used in any hoomd script.
Example of how to use an installed plugin:

```
from hoomd_script import *
from hoomd_plugins import plugin_template
init.create_random(N=1000, phi_p=0.20)
plugin_template.update.example(period=10)
```

To create a plugin that actually does something useful:

 * copy plugin_template_cpp to a new location
 * change the PROJECT() line in CMakeLists.txt to the name of your new plugin. This is the name that it will install to
 * Modify the source in cppmodule and pymodule. The existing files in those directories serve as examples and include
   many of the details in comments.
