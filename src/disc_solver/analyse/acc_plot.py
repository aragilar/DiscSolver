# -*- coding: utf-8 -*-
"""
Acc-plot command for DiscSolver
"""
from math import pi

from numpy import degrees

from ..constants import KM, AU, M_SUN, YEAR
from ..utils import get_normalisation, ODEIndex
from .utils import (
    single_solution_plotter, common_plotting_options, analyse_main_wrapper,
    get_common_plot_args, analysis_func_wrapper, plot_output_wrapper,
    DEFAULT_MPL_STYLE,
)

# AU^2 * Gauss^2 / 30 km/s in Msun/year
ACC_CONSTANT = ((AU ** 2) / (30 * KM)) / (M_SUN / YEAR)
# π * AU^2 * cm/s * g/cm^3 in Msun/year
WIND_CONSTANT = (pi * AU ** 2) / (M_SUN / YEAR)


def plot_parser(parser):
    """
    Add arguments for acc-plot command to parser
    """
    common_plotting_options(parser)
    return parser


@analyse_main_wrapper(
    "Plot derivs for DiscSolver",
    plot_parser,
    cmd_parser_splitters={
        "common_plot_args": get_common_plot_args,
    }
)
def acc_main(soln, *, soln_range, common_plot_args):
    """
    Entry point for ds-derivs-plot
    """
    return acc_plot(soln, soln_range=soln_range, **common_plot_args)


@analysis_func_wrapper
def acc_plot(
    soln, *, soln_range=None, plot_filename=None, show=False, stop=90,
    figargs=None, linestyle='.', title=None, close=True, filename,
    mpl_style=DEFAULT_MPL_STYLE
):
    """
    Show derivatives
    """
    # pylint: disable=too-many-function-args,unexpected-keyword-arg
    fig = generate_acc_plot(
        soln, soln_range, linestyle=linestyle, stop=stop, figargs=figargs,
        title=title, filename=filename, mpl_style=mpl_style,
    )

    return plot_output_wrapper(
        fig, file=plot_filename, show=show, close=close
    )


@single_solution_plotter
def generate_acc_plot(
    fig, soln, *, linestyle='.', stop=90, use_E_r=False
):
    """
    Friendlier plot for talks
    """
    # pylint: disable=unused-variable,unused-argument

    solution = soln.solution
    angles = soln.angles
    inp = soln.solution_input

    norms = get_normalisation(inp)  # need to allow config here
    B_norm, v_norm, ρ_norm = norms["B_norm"], norms["v_norm"], norms["ρ_norm"]

    B_θ = solution[:, ODEIndex.B_θ] * B_norm
    B_φ = solution[:, ODEIndex.B_φ] * B_norm
    v_θ = solution[:, ODEIndex.v_θ] * v_norm
    ρ = solution[:, ODEIndex.ρ] * ρ_norm
    xpos = degrees(angles) < 1

    ax_in, ax_wind = fig.subplots(ncols=2)
    acc_in = B_θ * B_φ * ACC_CONSTANT
    acc_wind = v_θ * ρ * WIND_CONSTANT
    ax_in.plot(degrees(angles[xpos]), acc_in[xpos])
    ax_wind.plot(degrees(angles[xpos]), acc_wind[xpos])
    return fig
