## \package PSEv1.shear_function
# classes representing shear functions, which can be input of an integrator and variant
# to shear the box of a simulation

import _PSEv1
from hoomd_script import globals

## shear function interface representing shear flow field described by a function
class _shear_function:
    ## Constructor and check the validity of zero param
    # \param zero Specify absolute time step number location for 0 in \a points. Use 'now' to indicate the current step.
    def __init__(self, zero = 'now'):
        self.cpp_function = None

        if zero == 'now':
            self._offset = globals.system.getCurrentTimeStep()
        else:
            # validate zero
            if zero < 0:
                globals.msg.error("Cannot create a shear_function variant with a negative zero\n")
                raise RuntimeError('Error creating shear function')
            if zero > globals.system.getCurrentTimeStep():
                globals.msg.error("Cannot create a shear_function variant with a zero in the future\n")
                raise RuntimeError('Error creating shear function')
            self._offset = zero

    ## Get shear rate at a certain time step, might be useful when switching strain field
    # \param timestep the timestep
    def get_shear_rate(self, timestep):
        return self.cpp_function.getShearRate(timestep)

    ## Get the strain at a certain time step. The strain is not wrapped
    # \param timestep the timestep
    def get_strain(self, timestep):
        return self.cpp_function.getStrain(timestep)

    ## Get the offset of this shear function
    def get_offset(self):
        return self.cpp_function.getOffset()


## concrete class representing steady shear, no shear by default if shear_rate is not provided
class steady(_shear_function):
    ## Constructor of steady shear function
    # \param dt the time interval between each timestep, must be the same with the global timestep
    # \param shear_rate the shear rate of the shear, default is zero, should be zero or positive
    # \param zero the time offset
    def __init__(self, dt, shear_rate = 0, zero = 'now'):
        _shear_function.__init__(self, zero)
        self.cpp_function = _PSEv1.SteadyShearFunction(shear_rate, self._offset, dt)


## concrete class representing simple sinusoidal oscillatory shear
class sine(_shear_function):
    ## Constructor of simple sinusoidal oscillatory shear
    # \param dt the time interval between each timestep, must be the same with the global timestep
    # \param shear_rate the maximum shear rate of the ocsillatory shear, must be positive
    # \param shear_freq the frequency (real frequency, not angular frequency) of the ocsillatory shear, must be positive
    # \param zero the time offset
    def __init__(self, dt, shear_rate, shear_freq, zero = 'now'):

        if shear_rate <= 0:
            globals.msg.error("Shear rate must be positive (use steady class instead for zero shear)\n")
            raise RuntimeError("Error creating shear function")
        if shear_freq <= 0:
            globals.msg.error("Shear frequency must be positive (use steady class instead for steady shear)\n")
            raise RuntimeError("Error creating shear function")

        _shear_function.__init__(self, zero)
        self.cpp_function = _PSEv1.SinShearFunction(shear_rate, shear_freq, self._offset, dt)


## concrete class representing chirp oscillatory shear
class chirp(_shear_function):
    ## Constructor of chirp oscillatory shear
    # \param dt the time interval between each timestep, must be the same with the global timestep
    # \param amplitude the strain amplitude of Chirp oscillatory shear, must be positive
    # \param omega_0 minimum angular frequency, must be positive
    # \param omega_f maximum angular frequency, must be positive and larger than omega_0
    # \param periodT final time of chirp
    # \param zero the time offset
    def __init__(self, dt, amplitude, omega_0, omega_f, periodT, zero = 'now'):
        _shear_function.__init__(self, zero)
        self.cpp_function = _PSEv1.ChirpShearFunction(amplitude, omega_0, omega_f, periodT, self._offset, dt)


## concrete class representing Tukey window function
class tukey_window(_shear_function):
    ## Constructor of Tukey window function
    # \param dt the time interval between each timestep, must be the same with the global timestep
    # \param periodT time length of the Tukey window function
    # \param tukey_param Tukey window function parameter, must be within (0, 1]
    # \param zero the time offset
    def __init__(self, dt, periodT, tukey_param, zero = 'now'):

        if tukey_param <= 0 or tukey_param > 1:
            globals.msg.error("Tukey parameter must be within (0, 1]")
            raise RuntimeError("Error creating Tukey window function")

        _shear_function.__init__(self, zero)
        self.cpp_function = _PSEv1.TukeyWindowFunction(periodT, tukey_param, self._offset, dt)


## concrete class represeting a windowed shear function
class windowed(_shear_function):
    ## Constructor of a windowed shear function
    # The strain of the resulting windowed shear function will be the product of the original shear function and
    # the provided window function
    # \param function_form the original shear function
    # \param window the window function. It is recommended to make sure the offset (zero) of the window function is the same with shear function
    def __init__(self, function_form, window):
        _shear_function.__init__(self, 'now') # zero parameter is not used in windowed class anyways
        self.cpp_function = _PSEv1.WindowedFunction(function_form.cpp_function, window.cpp_function)
