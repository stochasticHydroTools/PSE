#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#include "ShearFunction.h"

#ifndef __SPECIFIC_SHEAR_FUNCTION_H__
#define __SPECIFIC_SHEAR_FUNCTION_H__

#include <cmath>


//! Simple sinusoidal shear implementing the abstract class ShearFunction
class SinShearFunction : public ShearFunction
{
public:
    //! Constructor of SinShearFunction class
    /*! \param max_shear_rate maximum shear rate
        \param frequency real (NOT angular) frequency of oscillatory shear
        \param offset the offset of oscillatory shear
        \param dt the time interval
     */
    SinShearFunction(double max_shear_rate, double frequency, unsigned int offset, double dt) :
        ShearFunction(),
        m_max_shear_rate(max_shear_rate),
        m_frequency(frequency),
        m_offset(offset),
        m_dt(dt) { }
    double getShearRate(unsigned int timestep) {
        return m_max_shear_rate * cos( m_frequency * 2 * m_pi * ( (timestep - m_offset) * m_dt ) );
    }
    double getStrain(unsigned int timestep) {
        return m_max_shear_rate * sin( m_frequency * 2 * m_pi * ( (timestep - m_offset) * m_dt ) ) / m_frequency / 2 / m_pi;
    }
    unsigned int getOffset() {
        return m_offset;
    }
private:
    const double m_max_shear_rate; //!< maximum shear rate
    const double m_frequency; //!< Real frequency, not angular frequency
    const unsigned int m_offset; //!< offset of the sinusoidal oscillatory shear
    const double m_dt; //!< time step
    static constexpr double m_pi = 3.1415926536;
};

//! Simple steady shear implementing the abstract class ShearFunction
class SteadyShearFunction : public ShearFunction
{
public:
    //! Constructor of SteadyShearFunction
    /*! \param shear_rate the shear rate
        \param offset the offset of the steady shear
        \param the time interval between each timestep
     */
    SteadyShearFunction(double shear_rate, unsigned int offset, double dt) :
        ShearFunction(),
        m_shear_rate(shear_rate),
        m_offset(offset),
        m_dt(dt) { }
    double getShearRate(unsigned int timestep) {
        return m_shear_rate;
    }
    double getStrain(unsigned int timestep) {
        return m_shear_rate * (timestep - m_offset) * m_dt;
    }
    unsigned int getOffset() {
        return m_offset;
    }
private:
  const double m_shear_rate; //!< constant shear rate
  const unsigned int m_offset; //!< offset of the steady shear
  const double m_dt; //!< time step
};

//! Chirp oscillatory shear implementing abstract class ShearFunction
/*! Adjusted from code of Zsigmond Varga, plugin PSEv1a_chirpv2
 */
class ChirpShearFunction : public ShearFunction
{
public:
    //! Constructor of ChirpShearFunction class
    /*! \param amp the strain amplitude of the chirp shear
        \param omega_0 the starting ANGULAR frequency of the shear
        \param omega_f the ending ANGULAR frequency of the shear
        \param periodT the total time of the chirp run
        \param offset the offset of the chirp return
        \param dt the time interval between each timestep
     */
    ChirpShearFunction(double amp, double omega_0, double omega_f, double periodT, unsigned int offset, double dt) :
        ShearFunction(),
        m_amp(amp),
        m_omega_0(omega_0),
        m_omega_f(omega_f),
        m_periodT(periodT),
        m_offset(offset),
        m_dt(dt) { }
    double getShearRate(unsigned int timestep) {
        double current_omega = getCurrentOmega(timestep);
        double current_phase = getCurrentPhase(timestep);
        return m_amp * current_omega * cos(current_phase);
    }
    double getStrain(unsigned int timestep) {
        double current_phase = getCurrentPhase(timestep);
        return m_amp * sin( current_phase );
    }
    unsigned int getOffset() {
        return m_offset;
    }
private:
    double getCurrentOmega(unsigned int timestep) {
        return m_omega_0 * exp( m_dt * (timestep - m_offset) * logf(m_omega_f / m_omega_0) / m_periodT );
    }
    double getCurrentPhase(unsigned int timestep) {
        return m_periodT * m_omega_0 / logf( m_omega_f / m_omega_0 ) * ( exp( m_dt * (timestep - m_offset) * logf(m_omega_f / m_omega_0) / m_periodT ) - 1 );
    }
    const double m_amp; //!< Amplitude
    const double m_omega_0; //!< Minimum angular frequency
    const double m_omega_f; //!< Maximum angular frequency
    const double m_periodT; //!< Final time of Chirp
    const unsigned int m_offset; //!< offset of the shear
    const double m_dt; //!< time step
};


//! Tukey Window Function implementing abstract class ShearFunction
/*! Strictly speaking, this function is not a ShearFunction since it will only be
    used as a window function and applied to other ShearFunctions. This class should
    never be used by itself. However, since ShearFunction provides all the abstract
    functions it needs. We will call this a ShearFunction to avoid duplicate base classes
    TODO: Change the names of ShearFunction/getShearRate/getStrain to more general descriptions.
 */
class TukeyWindowFunction : public ShearFunction
{
public:
    //! Constructor of TukeyWindowFunction class
    /*! \param periodT the total time of the window
        \param tukey_param the parameter of Tukey window function, must be within (0, 1]
        \param offset the offset of the window
        \param dt the time interval between each timestep
     */
    TukeyWindowFunction(double periodT, double tukey_param, unsigned int offset, double dt) :
        ShearFunction(),
        m_periodT(periodT),
        m_tukey_param(tukey_param),
        m_offset(offset),
        m_dt(dt) {
            m_omega_value = 2 * m_pi / tukey_param;
        }
    double getShearRate(unsigned int timestep) {
        double rel_time = (timestep - m_offset) * m_dt / m_periodT; // supposed to be within [0,1]
        if (rel_time <= 0 || rel_time >= 1) {
            return 0;
        }
        else if (rel_time >= m_tukey_param / 2 && rel_time <= 1 - m_tukey_param / 2) {
            return 0;
        }
        else if (rel_time < 0.5) {
            return -( sin( m_omega_value * (rel_time - m_tukey_param / 2) ) ) / 2 * m_omega_value / m_periodT;
        }
        else {
            return -( sin( m_omega_value * (rel_time - 1 + m_tukey_param / 2) ) ) / 2 * m_omega_value / m_periodT;
        }
    }
    double getStrain(unsigned int timestep) {
        double rel_time = (timestep - m_offset) * m_dt / m_periodT; // supposed to be within [0,1]
        if (rel_time <= 0 || rel_time >= 1) {
            return 0;
        }
        else if (rel_time >= m_tukey_param / 2 && rel_time <= 1 - m_tukey_param / 2) {
            return 1;
        }
        else if (rel_time < 0.5) {
            return ( 1 + cos( m_omega_value * (rel_time - m_tukey_param / 2) ) ) / 2;
        }
        else {
            return ( 1 + cos( m_omega_value * (rel_time - 1 + m_tukey_param / 2) ) ) / 2;
        }
    }
    unsigned int getOffset() {
        return m_offset;
    }
private:
    const double m_periodT; //!< The time period of the window
    const double m_tukey_param; //!< The parameter of Tukey window function (scales the cosine lobe)
    const unsigned int m_offset; //!< offset of the window function
    const double m_dt; //!< time step
    static constexpr double m_pi = 3.1415926536;
    double m_omega_value; //!< omega value of the cosine function
};


//! Windowed ShearFunction: A ShearFunction windowed by a window function (which is also a ShearFunction subclass)
/*! WindowedFunction represents a strain field whose strain is the product of original ShearFunction and the window
    function. Therefore, the shear rate satisfies the product rule of derivative.
 */
class WindowedFunction : public ShearFunction
{
public:
    //! Constructor of WindowedFunction class
    /*! It is recommended to use the same offset for base shear function and window function
        \param base_shear_func the base shear function
        \param window_func the window function
     */
    WindowedFunction(std::shared_ptr<ShearFunction> base_shear_func, std::shared_ptr<ShearFunction> window_func) :
        ShearFunction(),
        m_base_shear_func(base_shear_func),
        m_window_func(window_func) { }
    double getShearRate(unsigned int timestep) {
        return ( m_base_shear_func -> getShearRate(timestep) ) * ( m_window_func -> getStrain(timestep) ) +
            ( m_base_shear_func -> getStrain(timestep) ) * ( m_window_func -> getShearRate(timestep) );
    }
    double getStrain(unsigned int timestep) {
        return ( m_base_shear_func -> getStrain(timestep) ) * ( m_window_func -> getStrain(timestep) );
    }
    unsigned int getOffset() {
        return m_base_shear_func -> getOffset();
    }
private:
    const std::shared_ptr<ShearFunction> m_base_shear_func; //!< Base shear function
    const std::shared_ptr<ShearFunction> m_window_func; //!< Window function
};


void export_SpecificShearFunction(pybind11::module& m);

#endif
