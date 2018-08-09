// Maintainer: Gang Wang
// Updated to HOOMD2.x compatibility by Andrew M. Fiore

/*! \file ShearFunction.cc
    \brief Defines ShearFunction class and relevant functions
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include "ShearFunction.h"

using namespace std;

void export_ShearFunction(pybind11::module& m)
{
    
    pybind11::class_<ShearFunction, std::shared_ptr<ShearFunction> >( m, "ShearFunction" )
    .def(pybind11::init< >())
    .def("getShearRate", &ShearFunction::getShearRate)
    .def("getStrain", &ShearFunction::getStrain)
    .def("getOffset", &ShearFunction::getOffset);

}

#ifdef WIN32
#pragma warning( pop )
#endif
