// Maintainer: Gang Wang
// Updated to HOOMD2.x compatibility by Andrew M. Fiore

/*! \file ShearFunction.cc
    \brief Defines ShearFunction class and relevant functions
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include "SpecificShearFunction.h"

using namespace std;

void export_SpecificShearFunction(pybind11::module& m)
{
    
    pybind11::class_<SinShearFunction, std::shared_ptr<SinShearFunction> >( m, "SinShearFunction", pybind11::base<ShearFunction>())
    .def(pybind11::init< double, double, unsigned int, double >());

    pybind11::class_<SteadyShearFunction, std::shared_ptr<SteadyShearFunction> > (m, "SteadyShearFunction", pybind11::base<ShearFunction>())
    .def(pybind11::init< double, unsigned int, double >());

    pybind11::class_<ChirpShearFunction, std::shared_ptr<ChirpShearFunction> >(m, "ChirpShearFunction", pybind11::base<ShearFunction>()) 
    .def(pybind11::init< double, double, double, double, unsigned int, double >());

    pybind11::class_<TukeyWindowFunction, std::shared_ptr<TukeyWindowFunction> >( m, "TukeyWindowFunction", pybind11::base<ShearFunction>()) 
    .def(pybind11::init< double, double, unsigned int, double >());

    pybind11::class_<WindowedFunction, std::shared_ptr<WindowedFunction> >(m, "WindowedFunction", pybind11::base<ShearFunction>()) 
    .def(pybind11::init< std::shared_ptr<ShearFunction>, std::shared_ptr<ShearFunction> >());
}

#ifdef WIN32
#pragma warning( pop )
#endif
