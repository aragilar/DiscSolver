# -*- coding: utf-8 -*-
"""
Plot command for DiscSolver
"""
from numpy import degrees, sqrt, tan

from ..solve.deriv_funcs import deriv_B_r_func
from ..utils import ODEIndex

from .utils import (
    single_solution_plotter, analyse_main_wrapper, analysis_func_wrapper,
    common_plotting_options, get_common_plot_args, plot_output_wrapper,
    AnalysisError, DEFAULT_MPL_STYLE,
)


def plot_parser(parser):
    """
    Add arguments for plot command to parser
    """
    common_plotting_options(parser)
    parser.add_argument("--only", default=None)
    return parser


def get_plot_args(args):
    """
    Parse plot args
    """
    return {
        "only": args.get("only", None)
    }


@analyse_main_wrapper(
    "Main plotter for DiscSolver",
    plot_parser,
    cmd_parser_splitters={
        "common_plot_args": get_common_plot_args,
        "plot_args": get_plot_args,
    }
)
def plot_main(soln, *, soln_range, common_plot_args, plot_args):
    """
    Entry point for ds-plot
    """
    return plot(soln, soln_range=soln_range, **common_plot_args, **plot_args)


@analysis_func_wrapper
def plot(
    soln, *, soln_range=None, plot_filename=None, show=False, linestyle='-',
    stop=90, figargs=None, title=None, close=True, only=None, filename,
    mpl_style=DEFAULT_MPL_STYLE
):
    """
    Plot solution to file
    """
    # pylint: disable=too-many-function-args,unexpected-keyword-arg
    fig = generate_plot(
        soln, soln_range, linestyle=linestyle, figargs=figargs, title=title,
        stop=stop, only=only, filename=filename, mpl_style=mpl_style,
    )

    return plot_output_wrapper(
        fig, file=plot_filename, show=show, close=close
    )


@single_solution_plotter
def generate_plot(
    fig, soln, *, linestyle='-', stop=90, only=None, use_E_r=False
):
    """
    Generate plot, with enough freedom to be able to format fig
    """
    if use_E_r:
        raise AnalysisError("Function needs modification to work with use_E_r")
    angles = soln.angles
    indexes = degrees(angles) <= stop
    plot_angles = degrees(angles[indexes])

    components = compute_components(soln, indexes, angles[indexes])

    if only:
        ax = fig.subplots()
        comps = components.get(only)
        if comps is None:
            raise AnalysisError("{} not found".format(only))

        for label, values in comps.items():
            ax.plot(plot_angles, values, linestyle, label=label)

        ax.set_xlabel("angle from plane (°)")
        ax.set_ylabel(only)
        ax.legend(loc=0)

    else:
        axes = fig.subplots(
            nrows=2, ncols=4, sharex=True, gridspec_kw=dict(hspace=0),
        )

        # only add label to bottom plots
        for ax in axes[1]:
            ax.set_xlabel("angle from plane (°)")

        for ax, comp_pair in zip(axes.flat, components.items()):
            eq_name, comps = comp_pair
            ax.set_ylabel(eq_name)
            for label, values in comps.items():
                ax.plot(plot_angles, values, linestyle, label=label)

            ax.legend(loc=0)

    return fig


def compute_components(soln, indexes, angles):
    """
    Return mapping containing all the components
    """
    solution = soln.solution
    cons = soln.initial_conditions

    B_r = solution[indexes, ODEIndex.B_r]
    B_φ = solution[indexes, ODEIndex.B_φ]
    B_θ = solution[indexes, ODEIndex.B_θ]
    v_r = solution[indexes, ODEIndex.v_r]
    v_φ = solution[indexes, ODEIndex.v_φ]
    v_θ = solution[indexes, ODEIndex.v_θ]
    ρ = solution[indexes, ODEIndex.ρ]
    deriv_B_φ = solution[indexes, ODEIndex.B_φ_prime]
    η_O = solution[indexes, ODEIndex.η_O]
    η_A = solution[indexes, ODEIndex.η_A]
    η_H = solution[indexes, ODEIndex.η_H]

    γ = cons.γ
    a_0 = cons.a_0
    norm_kepler_sq = cons.norm_kepler_sq

    norm_kepler = sqrt(norm_kepler_sq)

    B_mag = sqrt(B_r**2 + B_φ**2 + B_θ**2)
    norm_B_r, norm_B_φ, norm_B_θ = B_r/B_mag, B_φ/B_mag, B_θ/B_mag

    deriv_B_r = deriv_B_r_func(
        B_r=B_r, B_φ=B_φ, B_θ=B_θ, η_O=η_O, η_H=η_H, η_A=η_A, θ=angles,
        v_r=v_r, v_θ=v_θ, deriv_B_φ=deriv_B_φ, γ=γ,
    )

    deriv_v_θ = (
        v_r / 2 * (v_θ ** 2 - 4 * γ) + v_θ * (
            tan(angles) * (v_φ ** 2 + 1) + a_0 / ρ * (
                (1/4 - γ) * B_θ * B_r + B_r * deriv_B_r +
                B_φ * deriv_B_φ - B_φ ** 2 * tan(angles)
            )
        )
    ) / ((1 - v_θ) * (1 + v_θ))

    components = {
        "radial momentum": {
            "radial terms": v_r ** 2 + v_θ ** 2 + 5 / 2 - 2 * γ,
            "keplerian terms": (v_φ - norm_kepler) * (v_φ + norm_kepler),
            "magnetic terms": (
                a_0 / ρ * (B_θ * deriv_B_r + (1/4 - γ) * (B_θ ** 2 + B_φ ** 2))
            ),
            "all terms": (
                v_r ** 2 / 2 + v_θ ** 2 + 5/2 - 2 * γ +
                (v_φ - norm_kepler) * (v_φ + norm_kepler) + a_0 / ρ * (
                    B_θ * deriv_B_r + (1/4 - γ) * (B_θ ** 2 + B_φ ** 2)
                )
            ),
        },
        "azimuthal momentum": {
            "non-magnetic terms": v_φ * v_θ * tan(angles) - v_φ * v_r / 2,
            "magnetic terms": a_0 / ρ * (
                B_θ * deriv_B_φ - (1/4 - γ) * B_r * B_φ -
                B_θ * B_φ * tan(angles)
            ),
            "all terms": (
                v_φ * v_θ * tan(angles) - v_φ * v_r / 2 + a_0 / ρ * (
                    B_θ * deriv_B_φ - (1/4 - γ) * B_r * B_φ -
                    B_θ * B_φ * tan(angles)
                )
            ),
        },
        "polar momentum": {
            "no-B no-v_φ terms": v_r / 2 * (v_θ ** 2 - 4 * γ),
            "no-B v_φ terms": v_θ * tan(angles) * (v_φ ** 2 + 1),
            "magnetic terms": (
                v_θ * a_0 / ρ * (
                    (1/4 - γ) * B_θ * B_r + B_r * deriv_B_r +
                    B_φ * deriv_B_φ - B_φ ** 2 * tan(angles)
                )
            ),
            "all terms": (
                v_r / 2 * (v_θ ** 2 - 4 * γ) + v_θ * (
                    tan(angles) * (v_φ ** 2 + 1) + a_0 / ρ * (
                        (1/4 - γ) * B_θ * B_r + B_r * deriv_B_r +
                        B_φ * deriv_B_φ - B_φ ** 2 * tan(angles)
                    )
                )
            ),
        },
        "continuity": {
            "tan term": v_θ * tan(angles),
            "non-tan terms": - (2 * γ * v_r + deriv_v_θ),
            "all terms": - (2 * γ * v_r + deriv_v_θ) + v_θ * tan(angles),
        },
        "solenoid condition": {
            "B_θ terms": B_θ * tan(angles),
            "B_r terms": - (γ + 3/4) * B_r,
            "all terms": B_θ * tan(angles) - (γ + 3/4) * B_r,
        },
        "polar induction": {
            "v terms": (v_θ * B_r - v_r * B_θ) / (
                η_O + η_A * (1 - norm_B_φ) * (1 + norm_B_φ)
            ),
            "no-η terms": - B_θ * (1/4 - γ),
            "$B_φ'$ terms": - deriv_B_φ * (
                η_H * norm_B_θ + η_A * norm_B_r * norm_B_φ
            ) / (
                η_O + η_A * (1 - norm_B_φ) * (1 + norm_B_φ)
            ),
            "$B_φ$ terms": B_φ * (
                η_A * norm_B_φ * (
                    norm_B_r * tan(angles) - norm_B_θ * (1/4 - γ)
                ) + η_H * (
                    norm_B_r * (1/4 - γ) + norm_B_θ * tan(angles)
                )
            ) / (
                η_O + η_A * (1 - norm_B_φ) * (1 + norm_B_φ)
            ),
            "all terms": (
                (
                    v_θ * B_r - v_r * B_θ - deriv_B_φ * (
                        η_H * norm_B_θ +
                        η_A * norm_B_r * norm_B_φ
                    ) + B_φ * (
                        η_A * norm_B_φ * (
                            norm_B_r * tan(angles) -
                            norm_B_θ * (1/4 - γ)
                        ) + η_H * (
                            norm_B_r * (1/4 - γ) +
                            norm_B_θ * tan(angles)
                        )
                    )
                ) / (
                    η_O + η_A * (1 - norm_B_φ) * (1 + norm_B_φ)
                ) - B_θ * (1/4 - γ)
            ),
        },
    }

    return components
