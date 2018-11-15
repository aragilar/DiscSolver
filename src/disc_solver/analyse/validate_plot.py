# -*- coding: utf-8 -*-
"""
Validate-plot command for DiscSolver
"""
from types import SimpleNamespace

from numpy import sqrt, tan, degrees, absolute

from ..float_handling import float_type
from ..utils import ODEIndex

from .j_e_plot import J_func, E_func
from .utils import (
    single_solution_plotter, common_plotting_options, analyse_main_wrapper,
    get_common_plot_args, analysis_func_wrapper, plot_output_wrapper,
    AnalysisError, DEFAULT_MPL_STYLE,
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
    figargs=None, linestyle='.', title=None, close=True, filename,
    mpl_style=DEFAULT_MPL_STYLE
):
    """
    Show difference between original equations and ode solution
    """
    # pylint: disable=too-many-function-args,unexpected-keyword-arg
    fig = generate_validate_plot(
        soln, soln_range, linestyle=linestyle, stop=stop, figargs=figargs,
        title=title, filename=filename, mpl_style=mpl_style,
    )

    return plot_output_wrapper(
        fig, file=plot_filename, show=show, close=close
    )


@single_solution_plotter
def generate_validate_plot(
    fig, soln, *, linestyle='.', stop=90, use_E_r=False
):
    """
    Generate plot of difference between original equations and ode solution
    """
    if use_E_r:
        raise AnalysisError("Function needs modification to work with use_E_r")
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

    axes = fig.subplots(
        nrows=2, ncols=2, sharex=True, gridspec_kw=dict(hspace=0),
    )
    for ax in axes[1]:
        ax.set_xlabel("angle from plane (°)")
    axes = axes.flatten()

    ax_val = axes[0]
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

    ax_E = axes[1]
    ax_E.plot(
        degrees(values.angles[indexes]), values.E_r[indexes], linestyle,
        label="$E_r$",
    )
    ax_E.plot(
        degrees(values.angles[indexes]), values.E_θ[indexes], linestyle,
        label="$E_θ$",
    )
    ax_E.legend()

    ax_E_dash = axes[2]
    ax_E_dash.plot(
        degrees(values.angles[indexes]), values.E_r_dash[indexes], linestyle,
        label="$E_r'$",
    )
    ax_E_dash.plot(
        degrees(values.angles[indexes]), values.E_θ_dash[indexes], linestyle,
        label="$E_θ'$",
    )
    ax_E_dash.plot(
        degrees(values.angles[indexes]), values.E_φ_dash[indexes], linestyle,
        label="$E_φ'$",
    )
    ax_E_dash.legend()

    ax_J = axes[3]
    ax_J.plot(
        degrees(values.angles[indexes]), values.J_r[indexes], linestyle,
        label="$J_r$",
    )
    ax_J.plot(
        degrees(values.angles[indexes]), values.J_θ[indexes], linestyle,
        label="$J_θ$",
    )
    ax_J.plot(
        degrees(values.angles[indexes]), values.J_φ[indexes], linestyle,
        label="$J_φ$",
    )
    ax_J.legend()

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
    values.initial_conditions.γ = init_con.γ
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

    J, E_dash, E = get_E_values(
        values.initial_conditions, values
    )
    values.J_r, values.J_θ, values.J_φ = J
    values.E_r, values.E_θ, values.E_φ = E
    values.E_r_dash, values.E_θ_dash, values.E_φ_dash = E_dash

    return values


def get_E_values(initial_conditions, values):
    """
    Compute values for E
    """
    J_r, J_θ, J_φ = J_func(
        γ=initial_conditions.γ, θ=values.angles, B_θ=values.B_θ,
        B_φ=values.B_φ, deriv_B_φ=values.deriv.B_φ,
        deriv_B_r=values.deriv.B_r,
    )

    E_r, E_θ, E_φ = E_func(
        v_r=values.v_r, v_θ=values.v_θ, v_φ=values.v_φ, B_r=values.B_r,
        B_θ=values.B_θ, B_φ=values.B_φ, J_r=J_r, J_θ=J_θ, J_φ=J_φ,
        η_O=values.η_O, η_A=values.η_A, η_H=values.η_H,
    )

    E_r_dash = E_r + values.v_φ * values.B_θ - values.v_θ * values.B_φ
    E_θ_dash = E_θ + values.v_r * values.B_φ - values.v_φ * values.B_r
    E_φ_dash = E_φ + values.v_θ * values.B_r - values.v_r * values.B_θ

    return (
        (J_r, J_θ, J_φ),
        (E_r_dash, E_θ_dash, E_φ_dash),
        (E_r, E_θ, E_φ)
    )


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
