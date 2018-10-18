# -*- coding: utf-8 -*-
"""
hydro-check-plot command for DiscSolver
"""
from numpy import tan, degrees, absolute
import matplotlib.pyplot as plt

from .utils import (
    single_solution_plotter, common_plotting_options, analyse_main_wrapper,
    get_common_plot_args, analysis_func_wrapper, plot_output_wrapper,
    AnalysisError
)
from .validate_plot import get_values


def plot_parser(parser):
    """
    Add arguments for plot command to parser
    """
    common_plotting_options(parser)
    return parser


@analyse_main_wrapper(
    "Plot hydro-check for DiscSolver",
    plot_parser,
    cmd_parser_splitters={
        "common_plot_args": get_common_plot_args,
    }
)
def hydro_check_plot_main(soln, *, soln_range, common_plot_args):
    """
    Entry point for ds-hydro-check-plot
    """
    return hydro_check_plot(soln, soln_range=soln_range, **common_plot_args)


@analysis_func_wrapper
def hydro_check_plot(
    soln, *, soln_range=None, plot_filename=None, show=False, stop=90,
    figargs=None, linestyle='.', title=None, close=True, filename
):
    """
    Show difference between original equations and ode solution
    """
    # pylint: disable=too-many-function-args,unexpected-keyword-arg
    fig = generate_hydro_check_plot(
        soln, soln_range, linestyle=linestyle, stop=stop, figargs=figargs,
        title=title, filename=filename,
    )

    return plot_output_wrapper(
        fig, file=plot_filename, show=show, close=close
    )


@single_solution_plotter
def generate_hydro_check_plot(
    soln, *, linestyle='.', figargs=None, stop=90
):
    """
    Generate plot of difference between original equations and ode solution
    """
    if figargs is None:
        figargs = {}

    param_names = [
        {
            "name": "azimuthal momentum",
            "func": validate_hydro_azimuthal_mometum
        },
        {
            "name": "approximate momenta",
            "func": validate_hydro_approximate_momenta
        },
    ]

    if soln.internal_data is None:
        raise AnalysisError("Internal data required to generate plot")
    values = get_values(soln)
    indexes = degrees(values.angles) <= stop

    fig, axes = plt.subplots(**figargs)

    ax_val = axes
    ax_val.set_xlabel("angle from plane (°)")
    for settings in param_names:
        difference = settings["func"](
            values.initial_conditions, values
        )[indexes]
        print("{}: {}".format(settings["name"], max(absolute(difference))))

        ax_val.plot(
            degrees(values.angles[indexes]), difference, linestyle,
            label=settings["name"],
        )
    ax_val.legend()

    return fig


def validate_hydro_azimuthal_mometum(initial_conditions, values):
    """
    Validate azimuthal momentum equation
    """
    return (
        1/2 * values.v_r * values.v_φ - initial_conditions.a_0 / values.ρ * (
            values.B_θ * values.deriv.B_φ +
            (1 - initial_conditions.β) * values.B_r * values.B_φ -
            tan(values.angles) * values.B_θ * values.B_φ
        )
    )


def validate_hydro_approximate_momenta(initial_conditions, values):
    """
    Validate azimuthal momentum equation
    """
    return 1 - 2 * initial_conditions.a_0 * values.B_θ * values.deriv.B_φ / (
        values.ρ * values.v_r * values.v_φ
    )
