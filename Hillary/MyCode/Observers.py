import numpy as np
import diagnostics


class ErrorObserver:
    """"Class for calculating the l2 and linf error norm at every time step"""
    def __init__(self, number_timesteps):
        self.number_timesteps = number_timesteps

        self.l2_array = np.zeros(number_timesteps)
        self.linf_array = np.zeros(number_timesteps)

    def calculate_errors(self, analytical_solution, numerical_solution, timestep):
        """Calculates the L_2 and L_Inf norm between the numerical and analytical solution. """
        self.l2_array[timestep] = diagnostics.l2_error_norm(analytical_solution, numerical_solution)
        self.linf_array[timestep] = diagnostics.l_inf_error_norm(analytical_solution, numerical_solution)


class MomentObserver:
    """"Class for calculating the mass and variance at every time step"""
    def __init__(self, number_timesteps):
        self.number_timesteps = number_timesteps

        self.mass_array = np.zeros(number_timesteps)
        self.variance_array = np.zeros(number_timesteps)

    def calculate_moments(self, quantity, timestep):
        """Calculates the moments, assumes the spacing between points Dx = 1"""
        self.mass_array[timestep] = np.sum(quantity)
        self.variance_array[timestep] = np.sum(quantity**2) - self.mass_array[timestep]