import numpy as np

def ftcs(u_grid, c):
    pass

def time_evolution(u_grid, time_steps, c, advection_scheme_key):

    #Dictionary of possible advection schemes
    funcdict = {'FTCS': ftcs}

    #choose your desired advection scheme
    advection_scheme = func_dict[advection_scheme_key]

    for t in range(time_steps):
        u_grid = advection_scheme(u_grid,c)

    return (u_grid)
