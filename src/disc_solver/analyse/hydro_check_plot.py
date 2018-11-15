# -*- coding: utf-8 -*-
"""
hydro-check-plot command for DiscSolver
"""
from numpy import tan, degrees, absolute, sqrt

from ..solve.hydrostatic import X_func

from .utils import (
    single_solution_plotter, common_plotting_options, analyse_main_wrapper,
    get_common_plot_args, analysis_func_wrapper, plot_output_wrapper,
    AnalysisError, DEFAULT_MPL_STYLE,
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
    figargs=None, linestyle='.', title=None, close=True, filename,
    mpl_style=DEFAULT_MPL_STYLE
):
    """
    Show difference between original equations and ode solution
    """
    # pylint: disable=too-many-function-args,unexpected-keyword-arg
    fig = generate_hydro_check_plot(
        soln, soln_range, linestyle=linestyle, stop=stop, figargs=figargs,
        title=title, filename=filename, mpl_style=mpl_style
    )

    return plot_output_wrapper(
        fig, file=plot_filename, show=show, close=close
    )


@single_solution_plotter
def generate_hydro_check_plot(
    fig, soln, *, linestyle='.', stop=90, use_E_r=False
):
    """
    Generate plot of difference between original equations and ode solution
    """
    if use_E_r:
        raise AnalysisError("Function needs modification to work with use_E_r")
    param_names = [
        {
            "name": "azimuthal momentum",
            "func": validate_hydro_azimuthal_mometum
        },
        {
            "name": "approximate momenta",
            "func": validate_hydro_approximate_momenta
        },
        {
            "name": "radial momentum (approx method)",
            "func": validate_radial_momentum_mod_φ
        },
    ]

    if soln.internal_data is None:
        raise AnalysisError("Internal data required to generate plot")
    values = get_values(soln)
    indexes = degrees(values.angles) <= stop

    axes = fig.subplots()

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


def validate_radial_momentum_hydro(initial_conditions, values):
    """
    Validate radial momentum equation
    """
    b_r = values.norm_B_r
    b_φ = values.norm_B_φ
    b_θ = values.norm_B_θ

    X = X_func(
        η_O=values.η_O, η_A=values.η_A, η_H=values.η_H, b_θ=b_θ, b_r=b_r,
        b_φ=b_φ
    )

    return (
        1 / 2 * values.v_r**2 - values.v_φ**2 +
        initial_conditions.norm_kepler_sq + 5 / 2 -
        X * values.v_r * values.v_φ / 2 -
        initial_conditions.a_0 / values.ρ * (
            values.B_φ ** 2 / 4 - X * values.B_φ * (
                values.B_r / 4 + values.B_θ * tan(values.angles)
            ) - values.B_θ / (
                values.η_O + values.η_A * (1 - b_φ ** 2)
            ) * (
                values.v_r * values.B_θ - values.B_φ * (
                    values.η_A * b_φ * (
                        b_r * tan(values.angles) - b_θ / 4
                    ) + values.η_H * (
                        b_r / 4 + b_θ * tan(values.angles)
                    )
                )
            )
        )
    )


def validate_radial_momentum_mod_φ(initial_conditions, values):
    """
    Validate radial momentum equation
    """
    b_r = values.norm_B_r
    b_φ = values.norm_B_φ
    b_θ = values.norm_B_θ

    X = X_func(
        η_O=values.η_O, η_A=values.η_A, η_H=values.η_H, b_θ=b_θ, b_r=b_r,
        b_φ=b_φ
    )

    norm_kepler = sqrt(initial_conditions.norm_kepler_sq)

    return norm_kepler - values.v_φ - (
        1 / 2 * values.v_r**2 + 5 / 2 + initial_conditions.a_0 / values.ρ * (
            values.B_φ ** 2 / 4 - X * values.deriv.B_φ * values.B_θ -
            values.B_θ / (
                values.η_O + values.η_A * (1 - b_φ ** 2)
            ) * (
                values.v_r * values.B_θ - values.B_φ * (
                    values.η_A * b_φ * (
                        b_r * tan(values.angles) - b_θ / 4
                    ) + values.η_H * (
                        b_r / 4 + b_θ * tan(values.angles)
                    )
                )
            )
        )
    ) / (2 * norm_kepler)
