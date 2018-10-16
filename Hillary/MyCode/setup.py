import numpy as np


# This file contains functions for setting initial conditions. They take a linspaced python array


def initial_condition(x, initial_name, *args):
    """High level function that takes the grid and a name of the initial condition. It then calls the appropriate
    function. For convenience only."""

    assert type(initial_name) == str

    u_in = np.zeros(len(x))

    if initial_name == "gauss":
        u_in = gauss_curve(x, *args)
    else:
        raise ValueError("Non existing Label for initial condition.")

    return u_in


def gauss_curve(x, x_0=None):
    """Gives back a simple Gauss-Curve"""
    if x_0 is None:
        x_0 = np.mean(x)

    return np.exp(-(x - x_0) ** 2)
