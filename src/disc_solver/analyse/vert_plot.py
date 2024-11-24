# -*- coding: utf-8 -*-
"""
Plot command for DiscSolver
"""
from numpy import sqrt, ones as np_ones

from ..utils import (
    CylindricalODEIndex, convert_spherical_to_cylindrical,
    get_vertical_scaling,
)

from .utils import (
    single_solution_plotter, analyse_main_wrapper, analysis_func_wrapper,
    common_plotting_options, get_common_plot_args, plot_output_wrapper,
    DEFAULT_MPL_STYLE, get_common_arguments, PlotOrdering,
)


def plot_parser(parser):
    """
    Add arguments for plot command to parser
    """
    common_plotting_options(parser)
    parser.add_argument("--v_θ", choices=("log", "linear"), default="linear")
    parser.add_argument(
        "--with-sonic", action='store_true', default=False)
    return parser


def get_plot_args(args):
    """
    Parse plot args
    """
    return {
        "with_sonic": args.get("with_sonic", False),
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
    with_sonic=False, start=0, stop=90, figargs=None, title=None, close=True,
    filename, mpl_style=DEFAULT_MPL_STYLE, with_version=True
):
    """
    Plot solution to file
    """
    # pylint: disable=too-many-function-args,unexpected-keyword-arg
    fig = generate_plot(
        soln, soln_range, linestyle=linestyle, with_sonic=with_sonic,
        start=start, stop=stop, figargs=figargs, title=title,
        filename=filename, mpl_style=mpl_style, with_version=with_version,
    )

    return plot_output_wrapper(
        fig, file=plot_filename, show=show, close=close
    )


@single_solution_plotter
def generate_plot(
    fig, soln, *, linestyle='-', with_sonic=False, start=0, stop=90,
    use_E_r=False
):
    """
    Generate plot, with enough freedom to be able to format fig
    """
    solution = soln.solution
    angles = soln.angles
    inp = soln.solution_input
    cons = soln.initial_conditions

    heights, vert_soln = convert_spherical_to_cylindrical(
        angles, solution, γ=inp.γ, c_s_on_v_k=inp.c_s_on_v_k, use_E_r=use_E_r,
    )

    indexes = (start <= heights) & (heights <= stop)

    ordering = PlotOrdering.E_r_vert if use_E_r else (
        PlotOrdering.B_φ_prime_vert
    )

    param_names = get_common_arguments(
        ordering, initial_conditions=cons, v_φ_offset=sqrt(
            cons.norm_kepler_sq / get_vertical_scaling(
                angles, c_s_on_v_k=inp.c_s_on_v_k
            )
        ),
    )

    if with_sonic:
        param_names[CylindricalODEIndex.v_z]["extras"].append({
            "label": "sound",
            "data": np_ones(len(vert_soln)),
        })

    axes = fig.subplots(
        nrows=2, ncols=4, sharex=True, gridspec_kw=dict(hspace=0),
    )

    # only add label to bottom plots
    for ax in axes[1]:
        ax.set_xlabel("height from plane ($c_s/v_k$ scale heights)")

    axes.shape = len(param_names)
    for i, settings in enumerate(param_names):
        ax = axes[i]
        ax.plot(
            heights[indexes],
            (
                vert_soln[:, i] - settings.get("offset", 0)
            )[indexes], linestyle, label=settings["name"]
        )
        for extra in settings.get("extras", []):
            ax.plot(
                heights[indexes],
                extra["data"][indexes],
                label=extra.get("label")
            )
        ax.set_ylabel(settings["name"])
        ax.set_yscale(settings.get("scale", "linear"))
        if settings.get("legend"):
            ax.legend(loc=0)
    return fig
