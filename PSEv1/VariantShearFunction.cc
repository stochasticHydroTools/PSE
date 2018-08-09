// Maintainer: Gang Wang

/*! \file VariantShearFunction.cc
    \brief Defines VariantShearFunction class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include "VariantShearFunction.h"

using namespace std;


VariantShearFunction::VariantShearFunction(std::shared_ptr<ShearFunction> shear_func,
    unsigned int total_timestep,
    double min_value,
    double max_value) :
    m_shear_func(shear_func),
    m_total_timestep(total_timestep),
    m_min_value(min_value),
    m_max_value(max_value)
    {
        setOffset( m_shear_func -> getOffset() ); // This line ensures the offsets of ShearFunction and Variant class are equal
        m_value_range = m_max_value - m_min_value;
        m_end_value = wrapValue( m_shear_func -> getStrain( m_offset + m_total_timestep ) );
    }

/*! \param timestep Timestep to get the value at
    \return value by the user-specified function
*/
double VariantShearFunction::getValue(unsigned int timestep)
{
    if (timestep < m_offset) {
        return 0;
    }
    else if (timestep >= m_offset + m_total_timestep) {
        return m_end_value;
    }
    return wrapValue( m_shear_func -> getStrain(timestep) );
}

void export_VariantShearFunction(pybind11::module& m)
{
    pybind11::class_<VariantShearFunction, std::shared_ptr<VariantShearFunction> >(m, "VariantShearFunction", pybind11::base<Variant>())
    .def(pybind11::init< std::shared_ptr<ShearFunction>, unsigned int, double, double >());
}

#ifdef WIN32
#pragma warning( pop )
#endif
