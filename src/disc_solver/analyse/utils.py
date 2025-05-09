# -*- coding: utf-8 -*-
"""
Utils for analysis code
"""
import argparse
from enum import Enum
from functools import wraps
from os import fspath

import logbook
from logbook.compat import redirected_warnings, redirected_logging

from numpy import (
    sqrt, linspace, logical_not as npnot,
)

import matplotlib.pyplot as plt
from matplotlib.style import context as use_style
import mplcursors
import palettable

from h5preserve import open as h5open

from .. import __version__ as ds_version
from ..file_format import registries
from ..logging import log_handler
from ..utils import (
    ODEIndex, str_to_float, get_solutions, DiscSolverError,
    CylindricalODEIndex, main_entry_point_wrapper,
)

logger = logbook.Logger(__name__)

DEFAULT_MPL_STYLE = "bmh"

GREYS = plt.get_cmap("Greys")

COMMON_ARGUMENTS = {
    "B_r": {
        "name": "$B_r/B_0$",
    },
    "B_φ": {
        "name": "$B_φ/B_0$",
    },
    "B_θ": {
        "name": "$B_θ/B_0$",
    },
    "B_z": {
        "name": "$B_z/B_0$",
    },
    "v_r": {
        "name": "$v_r/c_s$",
    },
    "v_φ": {
        "name": "$(v_φ - v_k)/c_s$",
    },
    "v_θ": {
        "name": "$v_θ/c_s$",
        "legend": True,
    },
    "v_z": {
        "name": "$v_z/c_s$",
        "legend": True,
    },
    "ρ": {
        "name": "$ρ/ρ_0$",
        "scale": "log",
    },
    "B_φ_prime": {
        "name": "$B_φ'/B_0$",
    },
    "E_r": {
        "name": "$E_r/E_0$",
    },
}


class PlotOrdering(Enum):
    """
    Enum for the different orderings that plots can use
    """
    E_r = "E_r"
    B_φ_prime = "B_φ_prime"
    E_r_vert = "E_r_vert"
    B_φ_prime_vert = "B_φ_prime_vert"


PLOT_ORDERINGS = {
    PlotOrdering.B_φ_prime: [
        "B_r", "B_φ", "B_θ", "v_r", "v_φ", "v_θ", "ρ", "B_φ_prime"
    ],
    PlotOrdering.E_r: ["B_r", "B_φ", "B_θ", "v_r", "v_φ", "v_θ", "ρ", "E_r"],
    PlotOrdering.B_φ_prime_vert: [
        "B_r", "B_φ", "B_z", "v_r", "v_φ", "v_z", "ρ", "B_φ_prime"
    ],
    PlotOrdering.E_r_vert: [
        "B_r", "B_φ", "B_z", "v_r", "v_φ", "v_z", "ρ", "E_r"
    ],
}


def add_version_to_plot(fig):
    """
    Add disc solver version to plot
    """
    # We can use supxlabel here as this is still auto if x is set (setting y
    # would remove _autopos=True)
    fig.supxlabel(
        ds_version,
        # A small amount of space from the border
        x=0.01,
        # start laying this out from the left, so we don't overflow the left
        # border
        horizontalalignment="left",
    )


def get_common_arguments(
    ordering, *, v_θ_scale="linear", initial_conditions, no_v_φ_offset=False,
    v_φ_offset=None
):
    """
    Return a list containing what to plot based on a particular set of
    parameters
    """
    params = PLOT_ORDERINGS[ordering]
    args = [COMMON_ARGUMENTS[param] for param in params]

    if "v_θ" in params:
        args[ODEIndex.v_θ]["scale"] = v_θ_scale
        args[ODEIndex.v_θ]["extras"] = []
    if "v_z" in params:
        args[CylindricalODEIndex.v_z]["extras"] = []

    if v_φ_offset is not None and "v_φ" in params:
        args[ODEIndex.v_φ]["offset"] = v_φ_offset
    elif not no_v_φ_offset and "v_φ" in params:
        args[ODEIndex.v_φ]["offset"] = sqrt(initial_conditions.norm_kepler_sq)

    return args


def single_solution_plotter(func):
    """
    Pulls out common elements of plots which take a single solution
    """
    @wraps(func)
    def plot_wrapper(
        h5file, solution, *args, title=None, filename=None, figargs=None,
        mpl_style=DEFAULT_MPL_STYLE, with_version=True, **kwargs
    ):
        """
        Wraps plot functions
        """
        if solution is None:
            solution = "0"
        if filename is None:
            filename = "{}:{}".format(
                h5file.config_filename, h5file.config_input.label
            )
        if figargs is None:
            figargs = {}
        with use_style(mpl_style):
            fig = plt.figure(constrained_layout=True, **figargs)
            func(
                fig, get_solutions(h5file, solution), *args,
                use_E_r=h5file.use_E_r, **kwargs
            )
            if title is None:
                fig.suptitle("{}:{}".format(filename, solution))
            else:
                fig.suptitle(title)

            if with_version:
                add_version_to_plot(fig)

            return fig
    return plot_wrapper


def multiple_solution_plotter(func):
    """
    Pulls out common elements of plots which take multiple solutions
    """
    @wraps(func)
    def plot_wrapper(
        solution_pairs, *args, title=None, figargs=None,
        mpl_style=DEFAULT_MPL_STYLE, with_version=True, num_solutions=None,
        **kwargs
    ):
        """
        Wraps plot functions
        """
        if figargs is None:
            figargs = {}

        if num_solutions is None:
            num_solutions = len(solution_pairs)

        def solution_loader(solns):
            for run, name, filename in solns:
                if filename is None:
                    filename = "{}:{}".format(
                        run.config_filename, run.config_input.label
                    )
                soln = get_solutions(run, name)
                if soln is None:
                    logger.warning("{} not in {}".format(name, filename))
                    continue
                soln_name = "{}:{}".format(filename, name)
                yield soln_name, soln

        with use_style(mpl_style):
            fig = plt.figure(constrained_layout=True, **figargs)
            func(
                fig, solution_loader(solution_pairs), *args,
                num_solutions=num_solutions, **kwargs
            )
            if title is not None:
                fig.suptitle(title)

            if with_version:
                add_version_to_plot(fig)

            return fig
    return plot_wrapper


def analysis_func_wrapper_multisolution(func):
    """
    Wrapper for main analysis functions which take multiple solutions
    """
    @wraps(func)
    def wrapper(solutions_pairs, *args, num_solutions=None, **kwargs):
        """
        wrapper for analysis_func_wrapper_multisolution
        """
        if num_solutions is None:
            num_solutions = len(solutions_pairs)

        def file_loader(pairs):
            for filename, index_str in pairs:
                with h5open(filename, registries, mode='r') as soln_file:
                    yield soln_file["run"], index_str, filename
        return func(
            file_loader(solutions_pairs), *args, num_solutions=num_solutions,
            **kwargs
        )

    return wrapper


def common_output_plot_options(parser):
    """
    Control how plot is outputted, including filename and figure size.
    """
    parser.add_argument("--show", action="store_true", default=False)
    parser.add_argument("--filename", default=None)
    parser.add_argument(
        "--figsize", nargs=2, default=None, type=str_to_float
    )
    return parser


def common_plot_appearence_options(parser):
    """
    Common options for how the overall plot appears.
    """
    parser.add_argument("--title", default=None)
    parser.add_argument("--style", default=DEFAULT_MPL_STYLE)
    parser.add_argument(
        "--no-show-version", action="store_false", dest="with_version",
        default=True,
    )
    return parser


def common_plotting_options(parser):
    """
    Common plotting options to control appearance and output
    """
    common_output_plot_options(parser)
    common_plot_appearence_options(parser)
    parser.add_argument("--start", default='0', type=str_to_float)
    parser.add_argument("--stop", default='90', type=str_to_float)
    parser.add_argument("--linestyle", default="-")
    return parser


def get_common_plot_appearence_args(args):
    """
    Get arguments for how the overall plot appears.
    """
    return {
        "title": args["title"],
        "mpl_style": args["style"],
        "with_version": args["with_version"],
    }


def get_common_plot_output_args(args):
    """
    Control how plot is outputted, including filename and figure size.
    """
    figargs = {}
    if args.get("figsize") is not None:
        figargs["figsize"] = args["figsize"]

    return {
        "show": args["show"],
        "plot_filename": args["filename"],
        "figargs": figargs,
    }


def get_common_plot_args(args):
    """
    Extract the common plot options into the correct variables
    """
    appearence = get_common_plot_appearence_args(args)
    output = get_common_plot_output_args(args)
    other = {
        "start": args["start"],
        "stop": args["stop"],
        "linestyle": args["linestyle"],
    }
    return dict(**appearence, **output, **other)


def analyse_main_wrapper(
    description, cmd_parser_func, cmd_parser_splitters=None, **kwargs
):
    """
    Wrapper for main cmd for analysis cmds
    """
    if cmd_parser_splitters is None:
        cmd_parser_splitters = {}

    def decorator(cmd):
        """
        decorator for analyse_main_wrapper
        """
        @main_entry_point_wrapper(
            argument_default=argparse.SUPPRESS, description=description,
            **kwargs
        )
        @wraps(cmd)
        def wrap_analysis_main(argv, parser):
            """
            Actual main function for analysis code, deals with parsers and
            logging
            """
            cmd_args = {}
            parser.add_argument("soln_file")
            parser.add_argument("soln_range")
            cmd_parser = cmd_parser_func(parser)
            args = vars(cmd_parser.parse_args(argv))
            for name, func in cmd_parser_splitters.items():
                cmd_args[name] = func(args)
            with log_handler(args):
                with redirected_warnings(), redirected_logging():
                    return cmd(
                        args["soln_file"],
                        soln_range=args["soln_range"],
                        **cmd_args
                    )

        return wrap_analysis_main

    return decorator


def analyse_main_wrapper_multisolution(
    description, cmd_parser_func, cmd_parser_splitters=None, **kwargs
):
    """
    Wrapper for main cmd for analysis cmds
    """
    if cmd_parser_splitters is None:
        cmd_parser_splitters = {}

    def decorator(cmd):
        """
        decorator for analyse_main_wrapper_multisolution
        """
        @main_entry_point_wrapper(
            argument_default=argparse.SUPPRESS, description=description,
            **kwargs
        )
        @wraps(cmd)
        def wrap_analysis_main(argv, parser):
            """
            Actual main function for analysis code, deals with parsers and
            logging
            """
            cmd_args = {}
            parser.add_argument(
                "-r", "--runs", action="append", nargs=2,
                metavar=("soln_file", "soln_range"), dest="soln_pairs",
            )
            cmd_parser = cmd_parser_func(parser)
            args = vars(cmd_parser.parse_args(argv))
            for name, func in cmd_parser_splitters.items():
                cmd_args[name] = func(args)
            with log_handler(args):
                with redirected_warnings(), redirected_logging():
                    return cmd(args["soln_pairs"], **cmd_args)

        return wrap_analysis_main

    return decorator


def analyse_multisolution_wrapper(
    description, cmd_parser_func, cmd_parser_splitters=None, **kwargs
):
    """
    Wrapper for main entry point for analysis functions which use multiple
    solutions
    """
    def decorator(cmd):
        """
        decorator for analyse_main_wrapper
        """
        @main_entry_point_wrapper(
            argument_default=argparse.SUPPRESS, description=description,
            **kwargs
        )
        @wraps(cmd)
        def wrap_analysis_main(argv, parser):
            """
            Actual main function for analysis code, deals with parsers and
            logging
            """
            cmd_args = {}
            parser.add_argument("soln_file")
            cmd_parser = cmd_parser_func(parser)
            args = vars(cmd_parser.parse_args(argv))
            for name, func in cmd_parser_splitters.items():
                cmd_args[name] = func(args)
            with log_handler(args):
                with redirected_warnings(), redirected_logging():
                    soln_filename = args["soln_file"]
                    with h5open(
                        soln_filename, registries, mode='r'
                    ) as soln_file:
                        solutions = sorted(
                            soln_file["run"].solutions.items(),
                            key=lambda x: int(x[0])
                        )
                        return cmd(
                            solutions=solutions, filename=soln_filename,
                            **cmd_args
                        )

        return wrap_analysis_main

    return decorator


def analysis_func_wrapper(func):
    """
    Wrapper for main analysis functions which have common elements
    """
    @wraps(func)
    def wrapper(filename, *args, **kwargs):
        """
        wrapper for analysis_func_wrapper
        """
        with h5open(filename, registries, mode='r') as soln_file:
            return func(soln_file["run"], *args, filename=filename, **kwargs)
    return wrapper


def get_scale_height(solution):
    """
    Get the scale height of the solution
    """
    return solution.solution_input.c_s_on_v_k


class AnalysisError(DiscSolverError):
    """
    Error class for problems with analysis routines
    """
    pass


def distinct_color_map(size):
    """
    Generate a list of unique colours for matplotlib line/scatter plots
    """
    tab10 = palettable.tableau.Tableau_10.mpl_colors
    tab20 = palettable.tableau.Tableau_20.mpl_colors
    tab30 = (
        palettable.tableau.Tableau_10.mpl_colors +
        palettable.tableau.TableauLight_10.mpl_colors +
        palettable.tableau.TableauMedium_10.mpl_colors
    )
    if size <= len(tab10):
        return tab10
    elif size <= len(tab20):
        return tab20
    elif size <= len(tab30):
        return tab30
    grey_section = list(
        GREYS(linspace(0.1, 0.9, num=size - len(tab30)))
    )
    return grey_section + tab30


def plot_output_wrapper(
    fig, *, file=None, show=False, close=True, facecolor="none", **kwargs
):
    """
    Wrapper for handling whether a figure is shown, saved to file, and/or
    closed.
    """
    if file is not None:
        fig.savefig(fspath(file), facecolor=facecolor, **kwargs)
    if show:
        plt.show()
    if close:
        plt.close(fig)
        return None
    return fig


def add_label_display_on_select(axis):
    """
    When clicking on lines, show the associated label of the line
    """
    label_cursor = mplcursors.cursor(axis)
    label_cursor.connect(
        "add", lambda sel: sel.annotation.set_text(sel.artist.get_label())
    )


def plot_log_lines(ax, angles, values, *args, **kwargs):
    """
    Plot positive and negative values separately
    """
    pos_slice = values >= 0
    neg_slice = npnot(pos_slice)
    ax.plot(angles[pos_slice], values[pos_slice], *args, **kwargs)
    ax.plot(angles[neg_slice], - values[neg_slice], *args, **kwargs)
    ax.set_yscale("log")


def single_fig_legend_setup(fig, *, nrows, ncols, **kwargs):
    """
    Add a legend for the whole figure to the right of existing plots
    """
    axes = fig.subplots(nrows=nrows, ncols=ncols + 1, **kwargs)

    gs = axes[0, 0].get_gridspec()
    for ax in axes[:, -1]:
        ax.remove()

    def figlegend(**kwargs):
        ax0 = fig.axes[0]
        handles, labels = ax0.get_legend_handles_labels()
        legend_ax = fig.add_subplot(gs[:, -1])
        legend = legend_ax.legend(
            handles=handles, labels=labels, borderaxespad=0, **kwargs
        )
        legend_ax.axis("off")
        return legend

    return axes[:, :-1], figlegend
