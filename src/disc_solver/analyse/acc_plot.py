# -*- coding: utf-8 -*-
"""
Acc-plot command for DiscSolver
"""
from math import pi

from numpy import degrees
import matplotlib.pyplot as plt

from ..constants import KM, AU, M_SUN, YEAR
from ..utils import get_normalisation
from .utils import single_solution_plotter, common_plotting_options

# AU^2 * Gauss^2 / 30 km/s in Msun/year
ACC_CONSTANT = ((AU ** 2) / (30 * KM)) / (M_SUN / YEAR)
# π * AU^2 * cm/s * g/cm^3 in Msun/year
WIND_CONSTANT = (pi * AU ** 2) / (M_SUN / YEAR)


def acc_plot_parser(parser):
    """
    Add arguments for acc-plot command to parser
    """
    common_plotting_options(parser)
    return parser


@single_solution_plotter
def plot_acc(
    soln, *, figargs=None
):
    """
    Friendlier plot for talks
    """
    if figargs is None:
        figargs = {}

    solution = soln.solution
    angles = soln.angles
    inp = soln.solution_input

    norms = get_normalisation(inp)  # need to allow config here
    B_norm, v_norm, ρ_norm = norms["B_norm"], norms["v_norm"], norms["ρ_norm"]

    B_θ = solution[:, 2] * B_norm
    B_φ = solution[:, 1] * B_norm
    v_θ = solution[:, 5] * v_norm
    ρ = solution[:, 6] * ρ_norm
    xpos = degrees(angles) < 1

    fig, (ax_in, ax_wind) = plt.subplots(ncols=2, **figargs)
    acc_in = B_θ * B_φ * ACC_CONSTANT
    acc_wind = v_θ * ρ * WIND_CONSTANT
    ax_in.plot(degrees(angles[xpos]), acc_in[xpos])
    ax_wind.plot(degrees(angles[xpos]), acc_wind[xpos])
    return fig
