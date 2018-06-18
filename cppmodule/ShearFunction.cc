// Maintainer: Gang Wang

/*! \file ShearFunction.cc
    \brief Defines ShearFunction class and relevant functions
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include "ShearFunction.h"

#include <boost/python.hpp>
using namespace boost::python;

//! Wrapper class to expose pure virtual method to python
class ShearFunctionWrap: public ShearFunction, public wrapper<ShearFunction>
{
public:
    double getShearRate(unsigned int timestep) {
        return this->get_override("getShearRate")(timestep);
    }
    double getStrain(unsigned int timestep) {
        return this->get_override("getStrain")(timestep);
    }
    unsigned int getOffset() {
        return this->get_override("getOffset")();
    }
};


void export_ShearFunction()
{
    class_<ShearFunctionWrap, boost::shared_ptr<ShearFunctionWrap>, boost::noncopyable >("ShearFunction", init< >())
    .def("getShearRate", pure_virtual(&ShearFunction::getShearRate))
    .def("getStrain", pure_virtual(&ShearFunction::getStrain))
    .def("getOffset", pure_virtual(&ShearFunction::getOffset));

    class_<SinShearFunction, boost::shared_ptr<SinShearFunction>, bases<ShearFunction>, boost::noncopyable >("SinShearFunction", init< double, double, unsigned int, double >());
    class_<SteadyShearFunction, boost::shared_ptr<SteadyShearFunction>, bases<ShearFunction>, boost::noncopyable >("SteadyShearFunction", init< double, unsigned int, double >());
    class_<ChirpShearFunction, boost::shared_ptr<ChirpShearFunction>, bases<ShearFunction>, boost::noncopyable >("ChirpShearFunction", init< double, double, double, double, unsigned int, double >());
    class_<TukeyWindowFunction, boost::shared_ptr<TukeyWindowFunction>, bases<ShearFunction>, boost::noncopyable >("TukeyWindowFunction", init< double, double, unsigned int, double >());
    class_<WindowedFunction, boost::shared_ptr<WindowedFunction>, bases<ShearFunction>, boost::noncopyable >("WindowedFunction", init< boost::shared_ptr<ShearFunction>, boost::shared_ptr<ShearFunction> >());
}

#ifdef WIN32
#pragma warning( pop )
#endif
