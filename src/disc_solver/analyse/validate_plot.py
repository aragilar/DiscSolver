# -*- coding: utf-8 -*-
"""
Validate-plot command for DiscSolver
"""
from types import SimpleNamespace

from numpy import sqrt, tan, degrees
import matplotlib.pyplot as plt

from ..float_handling import float_type
from ..utils import ODEIndex

from .utils import (
    single_solution_plotter, common_plotting_options, analyse_main_wrapper,
    get_common_plot_args, analysis_func_wrapper, savefig, AnalysisError
)


def plot_parser(parser):
    """
    Add arguments for plot command to parser
    """
    common_plotting_options(parser)
    return parser


@analyse_main_wrapper(
    "Plot validate for DiscSolver",
    plot_parser,
    cmd_parser_splitters={
        "common_plot_args": get_common_plot_args,
    }
)
def validate_plot_main(soln, *, soln_range, common_plot_args):
    """
    Entry point for ds-validate-plot
    """
    return validate_plot(soln, soln_range=soln_range, **common_plot_args)


@analysis_func_wrapper
def validate_plot(
    soln, *, soln_range=None, plot_filename=None, show=False, stop=90,
    figargs=None, linestyle='.', title=None
):
    """
    Show difference between original equations and ode solution
    """
    # pylint: disable=too-many-function-args,unexpected-keyword-arg
    fig = generate_validate_plot(
        soln, soln_range, linestyle=linestyle, stop=stop, figargs=figargs,
        title=title,
    )
    if plot_filename is not None:
        savefig(fig, plot_filename)
    if show:
        plt.show()
    plt.close(fig)


@single_solution_plotter
def generate_validate_plot(
    soln, *, linestyle='.', figargs=None, stop=90
):
    """
    Generate plot of difference between original equations and ode solution
    """
    if figargs is None:
        figargs = {}

    param_names = [
        {
            "name": "continuity",
            "func": validate_continuity
        },
        {
            "name": "solenoid",
            "func": validate_solenoid
        },
        {
            "name": "radial momentum",
            "func": validate_radial_momentum
        },
        {
            "name": "azimuthal momentum",
            "func": validate_azimuthal_mometum
        },
        {
            "name": "polar momentum",
            "func": validate_polar_momentum
        },
        {
            "name": "polar induction",
            "func": validate_polar_induction
        },
        {
            "name": "E_φ",
            "func": validate_E_φ
        }

    ]

    if soln.internal_data is None:
        raise AnalysisError("Internal data required to generate plot")
    values = get_values(soln)
    indexes = degrees(values.angles) <= stop

    fig, axes = plt.subplots(
        nrows=1, ncols=2, tight_layout=True, **figargs
    )

    ax = axes[0]
    for settings in param_names:
        difference = settings["func"](
            values.initial_conditions, values
        )[indexes]
        print("{}: {}".format(settings["name"], max(difference)))

        ax.plot(
            degrees(values.angles[indexes]), difference, linestyle,
            label=settings["name"],
        )
    ax.set_xlabel("angle from plane (°)")
    ax.legend()

    ax2 = axes[1]
    ax2.plot(
        degrees(values.angles[indexes]), values.E_r[indexes], linestyle,
        label="$E_r$",
    )
    ax2.plot(
        degrees(values.angles[indexes]), values.E_θ[indexes], linestyle,
        label="$E_θ$",
    )
    ax2.set_xlabel("angle from plane (°)")
    ax2.legend()

    return fig


def get_values(solution):
    """
    Get clean namespace for comparison
    """
    internal_data = solution.internal_data
    init_con = solution.initial_conditions

    params = internal_data.params
    derivs = internal_data.derivs

    values = SimpleNamespace(
        deriv=SimpleNamespace(),
        initial_conditions=SimpleNamespace(),
    )

    values.initial_conditions.a_0 = init_con.a_0
    values.initial_conditions.β = float_type(5)/float_type(4) - init_con.γ
    values.initial_conditions.norm_kepler_sq = init_con.norm_kepler_sq

    values.B_r = params[:, ODEIndex.B_r]
    values.B_φ = params[:, ODEIndex.B_φ]
    values.B_θ = params[:, ODEIndex.B_θ]
    values.v_r = params[:, ODEIndex.v_r]
    values.v_φ = params[:, ODEIndex.v_φ]
    values.v_θ = params[:, ODEIndex.v_θ]
    values.ρ = params[:, ODEIndex.ρ]
    values.η_O = params[:, ODEIndex.η_O]
    values.η_A = params[:, ODEIndex.η_A]
    values.η_H = params[:, ODEIndex.η_H]
    values.deriv.B_r = derivs[:, ODEIndex.B_r]
    values.deriv.B_φ = derivs[:, ODEIndex.B_φ]
    values.deriv.B_θ = derivs[:, ODEIndex.B_θ]
    values.deriv.v_r = derivs[:, ODEIndex.v_r]
    values.deriv.v_φ = derivs[:, ODEIndex.v_φ]
    values.deriv.v_θ = derivs[:, ODEIndex.v_θ]
    values.deriv.ρ = derivs[:, ODEIndex.ρ]
    values.deriv.B_φ_prime = derivs[:, ODEIndex.B_φ_prime]
    values.deriv.η_O = derivs[:, ODEIndex.η_O]
    values.deriv.η_A = derivs[:, ODEIndex.η_A]
    values.deriv.η_H = derivs[:, ODEIndex.η_H]

    values.angles = internal_data.angles

    B_mag = sqrt(values.B_r ** 2 + values.B_φ ** 2 + values.B_θ ** 2)
    values.norm_B_r, values.norm_B_φ, values.norm_B_θ = (
        values.B_r/B_mag, values.B_φ/B_mag, values.B_θ/B_mag)

    values.E_r, values.E_θ, values.E_φ = get_E_values(
        values.initial_conditions, values
    )

    return values


def get_E_values(initial_conditions, values):
    """
    Compute values for E
    """
    A_r = values.deriv.B_φ - values.B_φ * tan(values.angles)
    A_θ = (1 - initial_conditions.β) * values.B_φ
    A_φ = values.B_θ * (1 - initial_conditions.β) - values.deriv.B_r

    E_r_dash = values.η_O * A_r + values.η_H * (
        A_θ * values.norm_B_φ - A_φ * values.norm_B_θ
    ) - values.η_A * (
        A_φ * values.norm_B_r * values.norm_B_φ +
        A_θ * values.norm_B_r * values.norm_B_θ -
        A_r * (1 - values.norm_B_r ** 2)
    )
    E_θ_dash = values.η_O * A_θ + values.η_H * (
        A_φ * values.norm_B_r - A_r * values.norm_B_φ
    ) - values.η_A * (
        A_r * values.norm_B_r * values.norm_B_θ +
        A_φ * values.norm_B_θ * values.norm_B_φ -
        A_θ * (1 - values.norm_B_θ ** 2)
    )
    E_φ_dash = values.η_O * A_φ + values.η_H * (
        A_r * values.norm_B_θ - A_θ * values.norm_B_r
    ) - values.η_A * (
        A_θ * values.norm_B_θ * values.norm_B_φ +
        A_r * values.norm_B_r * values.norm_B_φ -
        A_φ * (1 - values.norm_B_φ ** 2)
    )

    E_r = values.v_θ * values.B_φ - values.v_φ * values.B_θ - E_r_dash
    E_θ = values.v_φ * values.B_r - values.v_r * values.B_φ - E_θ_dash
    E_φ = values.v_r * values.B_θ - values.v_θ * values.B_r - E_φ_dash

    return E_r, E_θ, E_φ


def validate_continuity(initial_conditions, values):
    """
    Validate continuity equation
    """
    return ((
        float_type(5) / float_type(2) - 2 * initial_conditions.β
    ) * values.v_r + values.deriv.v_θ + values.v_θ / values.ρ * (
        values.deriv.ρ - values.ρ * tan(values.angles)
    ))


def validate_solenoid(initial_conditions, values):
    """
    Validate solenoid equation
    """
    return values.deriv.B_θ - (
        (initial_conditions.β - 2) * values.B_r +
        values.B_θ * tan(values.angles)
    )


def validate_radial_momentum(initial_conditions, values):
    """
    Validate radial momentum equation
    """
    return (
        values.v_θ * values.deriv.v_r -
        float_type(1) / float_type(2) * values.v_r**2 -
        values.v_θ**2 - values.v_φ**2 + initial_conditions.norm_kepler_sq -
        2 * initial_conditions.β - initial_conditions.a_0 / values.ρ * (
            values.B_θ * values.deriv.B_r + (initial_conditions.β - 1) * (
                values.B_θ**2 + values.B_φ**2
            )
        )
    )


def validate_azimuthal_mometum(initial_conditions, values):
    """
    Validate azimuthal momentum equation
    """
    return (
        values.v_θ * values.deriv.v_φ + 1/2 * values.v_r * values.v_φ -
        tan(values.angles) * values.v_θ * values.v_φ -
        initial_conditions.a_0 / values.ρ * (
            values.B_θ * values.deriv.B_φ +
            (1 - initial_conditions.β) * values.B_r * values.B_φ -
            tan(values.angles) * values.B_θ * values.B_φ
        )
    )


def validate_polar_momentum(initial_conditions, values):
    """
    Validate polar momentum equation
    """
    return (
        values.v_r * values.v_θ / 2 + values.v_θ * values.deriv.v_θ +
        tan(values.angles) * values.v_φ ** 2 +
        values.deriv.ρ / values.ρ + initial_conditions.a_0 / values.ρ * (
            (initial_conditions.β - 1) * values.B_θ * values.B_r +
            values.B_r * values.deriv.B_r +
            values.B_φ * values.deriv.B_φ -
            values.B_φ ** 2 * tan(values.angles)
        )
    )


def validate_polar_induction(initial_conditions, values):
    """
    Validate polar induction equation
    """
    return (
        values.v_θ * values.B_r - values.v_r * values.B_θ + (
            values.B_θ * (1 - initial_conditions.β) - values.deriv.B_r
        ) * (
            values.η_O + values.η_A * (1 - values.norm_B_φ**2)
        ) + values.deriv.B_φ * (
            values.η_H * values.norm_B_θ -
            values.η_A * values.norm_B_r * values.norm_B_φ
        ) + values.B_φ * (
            values.η_H * (
                values.norm_B_r * (1 - initial_conditions.β) -
                values.norm_B_θ * tan(values.angles)
            ) - values.η_A * values.norm_B_φ * (
                values.norm_B_θ * (1 - initial_conditions.β) -
                values.norm_B_r * tan(values.angles)
            )
        )
    )


def validate_E_φ(initial_conditions, values):
    """
    Validate E_φ conservation
    """
    # pylint: disable=unused-argument
    return values.E_φ
