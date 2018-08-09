#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#ifndef __SHEAR_FUNCTION_H__
#define __SHEAR_FUNCTION_H__

#include <cmath>

//! Abstract class representing the function of shear rate and shear strain
/*! ShearFunction class, having three public pure virtual functions:
        1) getShearRate; 2) getStrain; and 3) getOffset
    This interface can make it easier to add new shear functionality to HOOMD.
    Compared with previous approach, we can simply subclass this interface without
    changing any existing code or creating a new plugin.
*/
class ShearFunction
{
public:

    //! Get shear rate at certain timestep
    /*! \param timestep the timestep
     */
    virtual double getShearRate(unsigned int timestep){ return double(0.0); }

    //! Get strain at certain timestep (unwrapped)
    /*! \param timestep the timestep
     */
    virtual double getStrain(unsigned int timestep){ return double(0.0); }

    //! Get the offset of timestep (typically offset is the timestep when the shear starts)
    virtual unsigned int getOffset(){ return int(0); }

};

//! Export the ShearFunction class to python
void export_ShearFunction(pybind11::module& m);

#endif
