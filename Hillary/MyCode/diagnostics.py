# Various function for plotting results and for calculating error measures

import numpy as np


def l2_error_norm(phi, phi_exact):
    "Calculates the l2 error norm (RMS error) of phi in comparison to"
    "phiExact"

    # calculate the error and the RMS error norm
    phi_error = phi - phi_exact
    l2 = np.sqrt(sum(phi_error ** 2) / sum(phi_exact ** 2))

    return l2


def l_inf_error_norm(phi, phi_exact):
    "Calculates the linf error norm (maximum normalised error) in comparison"
    "to phiExact"
    phi_error = phi - phi_exact
    linf = np.max(np.abs(phi_error)) / np.max(np.abs(phi_exact))
    return linf
