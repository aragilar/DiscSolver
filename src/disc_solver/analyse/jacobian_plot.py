# -*- coding: utf-8 -*-
"""
Jacobian-plot command for DiscSolver
"""
from math import ceil
import numpy as np
from numpy import degrees
import matplotlib.pyplot as plt
from scipy.linalg import eig

from ..utils import ODEIndex
from .utils import (
    single_solution_plotter, common_plotting_options, analyse_main_wrapper,
    get_common_plot_args, analysis_func_wrapper, savefig, get_jacobian,
)
from ..solve.solution import ode_system


def plot_parser(parser):
    """
    Add arguments for plot command to parser
    """
    common_plotting_options(parser)
    return parser


@analyse_main_wrapper(
    "Plot jacobian for DiscSolver",
    plot_parser,
    cmd_parser_splitters={
        "common_plot_args": get_common_plot_args,
    }
)
def jacobian_main(soln, *, soln_range, common_plot_args):
    """
    Entry point for ds-jacobian-plot
    """
    return jacobian_plot(soln, soln_range=soln_range, **common_plot_args)


@analysis_func_wrapper
def jacobian_plot(
    soln, *, soln_range=None, plot_filename=None, show=False, stop=90,
    figargs=None, linestyle='.', title=None
):
    """
    Show jacobian
    """
    # pylint: disable=too-many-function-args,unexpected-keyword-arg
    fig = generate_jacobian_plot(
        soln, soln_range, linestyle=linestyle, stop=stop, figargs=figargs,
        title=title,
    )
    if plot_filename is not None:
        savefig(fig, plot_filename)
    if show:
        plt.show()
    plt.close(fig)


@single_solution_plotter
def generate_jacobian_plot(
    soln, *, linestyle='.', figargs=None, stop=90
):
    """
    Generate plot of jacobians
    """
    if figargs is None:
        figargs = {}

    jacobian_data = soln.internal_data.jacobian_data
    angles = jacobian_data.angles
    jacobians = jacobian_data.jacobians
    params = jacobian_data.params
    indexes = degrees(angles) <= stop
    #print(jacobians[indexes].max())
    #tmp_jacs = jacobians[indexes]
    #for jac in tmp_jacs[1:] - tmp_jacs[:-1]:
    #    print(jac, end="\n\n\n")

    deriv_func, _ = ode_system(
        γ=soln.initial_conditions.γ, a_0=soln.initial_conditions.a_0,
        norm_kepler_sq=soln.initial_conditions.norm_kepler_sq,
        init_con=soln.initial_conditions.init_con,
        η_derivs=soln.solution_input.η_derivs, store_internal=False,
        with_taylor=False,
    )

    jacobians = []
    new_angles = []
    for angle, params_val in zip(angles, params):
        try:
            jacobians.append(get_jacobian(deriv_func, angle, params_val))
            new_angles.append(angle)
        except RuntimeError:
            print("Skipping angle {}".format(degrees(angle)))

    evals, evecs = zip(*[eig(j) for j in jacobians])
    evals = np.array(evals)
    evecs = np.array(evecs)
    log_mod = np.log10(np.absolute(evals))

    fig, ax = plt.subplots(tight_layout=True, **figargs)
    ax.plot(degrees(new_angles), log_mod, marker='.', linestyle='-')
    #ax.plot(degrees(new_angles), np.angle(evals), marker='.', linestyle='--')
    ax.legend([str(i) for i in range(11)])
    ax.set_xlabel("angle from plane (°)")
    for angle, evec in zip(new_angles, evecs[:, 2]):
        print(degrees(angle))
        print(evec, end='\n\n')
    return fig
