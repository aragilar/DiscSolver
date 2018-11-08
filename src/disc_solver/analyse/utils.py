# -*- coding: utf-8 -*-
"""
Utils for analysis code
"""
import argparse
from functools import wraps
from os import fspath
import sys

from logbook.compat import redirected_warnings, redirected_logging
from matplotlib.colors import TABLEAU_COLORS, XKCD_COLORS
import matplotlib.pyplot as plt
from matplotlib.style import context as use_style
import matplotlib._layoutbox as layoutbox
from numpy import sqrt
from scipy.interpolate import interp1d

from h5preserve import open as h5open

from .. import __version__ as ds_version
from ..file_format import registries
from ..logging import log_handler, logging_options
from ..utils import ODEIndex, str_to_float, get_solutions, DiscSolverError

DEFAULT_MPL_STYLE = "bmh"

B_φ_PRIME_ORDERING = [
    "B_r", "B_φ", "B_θ", "v_r", "v_φ", "v_θ", "ρ", "B_φ_prime"
]
E_r_ORDERING = ["B_r", "B_φ", "B_θ", "v_r", "v_φ", "v_θ", "ρ", "E_r"]

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


def constrain_text(fig, text):
    """
    Hack to constrain text on the bottom of the figure

    from matplotlib.figure.Figure.suptitle
    """
    # pylint: disable=unused-variable,protected-access
    if fig._layoutbox is not None:
        w_pad, h_pad, wspace, hspace = fig.get_constrained_layout_pads(
            relative=True
        )
        figlb = fig._layoutbox
        text._layoutbox = layoutbox.LayoutBox(
            parent=figlb, artist=text, name=figlb.name+'.textinfo'
        )
        # stack the text on bottom of all the children.
        for child in figlb.children:
            if child is not text._layoutbox:
                layoutbox.vstack(
                    [child, text._layoutbox],
                    padding=h_pad*2., strength='required'
                )


def get_common_arguments(
    params, *, v_θ_scale="linear", initial_conditions, no_v_φ_offest=False
):
    """
    Return a list containing what to plot based on a particular set of
    parameters
    """
    args = [COMMON_ARGUMENTS[param] for param in params]

    if "v_θ" in params:
        args[ODEIndex.v_θ]["scale"] = v_θ_scale
        args[ODEIndex.v_θ]["extras"] = []
    if not no_v_φ_offest and "v_φ" in params:
        args[ODEIndex.v_φ]["offset"] = sqrt(initial_conditions.norm_kepler_sq)

    return args


def single_solution_plotter(func):
    """
    Pulls out common elements of plots which take a single solution
    """
    @wraps(func)
    def plot_wrapper(
        h5file, solution, *args, title=None, filename=None, figargs=None,
        mpl_style=DEFAULT_MPL_STYLE, **kwargs
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
            version_text = fig.text(0, 0.01, ds_version)
            constrain_text(fig, version_text)
            return fig
    return plot_wrapper


def multiple_solution_plotter(func):
    """
    Pulls out common elements of plots which take multiple solutions
    """
    @wraps(func)
    def plot_wrapper(
        solution_pairs, *args, title=None, figargs=None,
        mpl_style=DEFAULT_MPL_STYLE, **kwargs
    ):
        """
        Wraps plot functions
        """
        title_list = []

        if figargs is None:
            figargs = {}

        def solution_loader(solns):
            for run, name, filename in solns:
                if filename is None:
                    filename = "{}:{}".format(
                        run.config_filename, run.config_input.label
                    )

                title_list.append("{}:{}".format(filename, name))

                yield get_solutions(run, name)

        with use_style(mpl_style):
            fig = plt.figure(constrained_layout=True, **figargs)
            func(fig, solution_loader(solution_pairs), *args, **kwargs)
            if title is None:
                fig.suptitle('\n'.join(title_list))
            else:
                fig.suptitle(title)
            return fig
    return plot_wrapper


def analysis_func_wrapper_multisolution(func):
    """
    Wrapper for main analysis functions which take multiple solutions
    """
    @wraps(func)
    def wrapper(solutions_pairs, *args, **kwargs):
        """
        wrapper for analysis_func_wrapper_multisolution
        """
        def file_loader(pairs):
            for filename, index_str in pairs:
                with h5open(filename, registries) as soln_file:
                    yield soln_file["run"], index_str, filename
        return func(file_loader(solutions_pairs), *args, **kwargs)

    return wrapper


def common_plotting_options(parser):
    """
    Common plotting options to control appearance and output
    """
    parser.add_argument("--show", action="store_true", default=False)
    parser.add_argument("--filename")
    parser.add_argument("--figsize", nargs=2)
    parser.add_argument("--stop")
    parser.add_argument("--linestyle")
    parser.add_argument("--title")
    parser.add_argument("--style")
    return parser


def get_common_plot_args(args):
    """
    Extract the common plot options into the correct variables
    """
    figargs = {}
    if args.get("figsize") is not None:
        figargs["figsize"] = (
            int(args["figsize"][0]),
            int(args["figsize"][1])
        )

    return {
        "show": args.get("show", False),
        "plot_filename": args.get("filename"),
        "figargs": figargs,
        "stop": str_to_float(args.get("stop", "90")),
        "linestyle": args.get("linestyle", "-"),
        "title": args.get("title"),
        "mpl_style": args.get("style", DEFAULT_MPL_STYLE),
    }


def analyse_main_wrapper(
    cmd_description, cmd_parser_func, cmd_parser_splitters=None
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
        @wraps(cmd)
        def wrap_analysis_main(argv=None):
            """
            Actual main function for analysis code, deals with parsers and
            logging
            """
            if argv is None:
                argv = sys.argv[1:]
            cmd_args = {}
            parser = argparse.ArgumentParser(
                description=cmd_description,
                argument_default=argparse.SUPPRESS,
            )
            parser.add_argument(
                '--version', action='version', version='%(prog)s ' + ds_version
            )
            parser.add_argument("soln_file")
            parser.add_argument("soln_range")
            logging_options(parser)
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
    cmd_description, cmd_parser_func, cmd_parser_splitters=None
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
        @wraps(cmd)
        def wrap_analysis_main(argv=None):
            """
            Actual main function for analysis code, deals with parsers and
            logging
            """
            if argv is None:
                argv = sys.argv[1:]
            cmd_args = {}
            parser = argparse.ArgumentParser(
                description=cmd_description,
                argument_default=argparse.SUPPRESS,
            )
            parser.add_argument(
                '--version', action='version', version='%(prog)s ' + ds_version
            )
            parser.add_argument(
                "-r", "--runs", action="append", nargs=2,
                metavar=("soln_file", "soln_range"), dest="soln_pairs",
            )
            logging_options(parser)
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
    cmd_description, cmd_parser_func, cmd_parser_splitters=None
):
    """
    Wrapper for main entry point for analysis functions which use multiple
    solutions
    """
    def decorator(cmd):
        """
        decorator for analyse_main_wrapper
        """
        @wraps(cmd)
        def wrap_analysis_main(argv=None):
            """
            Actual main function for analysis code, deals with parsers and
            logging
            """
            if argv is None:
                argv = sys.argv[1:]
            cmd_args = {}
            parser = argparse.ArgumentParser(
                description=cmd_description,
                argument_default=argparse.SUPPRESS,
            )
            parser.add_argument(
                '--version', action='version', version='%(prog)s ' + ds_version
            )
            parser.add_argument("soln_file")
            logging_options(parser)
            cmd_parser = cmd_parser_func(parser)
            args = vars(cmd_parser.parse_args(argv))
            for name, func in cmd_parser_splitters.items():
                cmd_args[name] = func(args)
            with log_handler(args):
                with redirected_warnings(), redirected_logging():
                    with h5open(args["soln_file"], registries) as soln_file:
                        solutions = soln_file["run"].solutions.values()
                        return cmd(
                            *solutions, **cmd_args
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
        with h5open(filename, registries) as soln_file:
            return func(soln_file["run"], *args, filename=filename, **kwargs)
    return wrapper


def get_scale_height(solution):
    """
    Get the scale height of the solution
    """
    return solution.solution_input.c_s_on_v_k


def get_sonic_point(solution):
    """
    Get the angle at which the sonic point occurs
    """
    if solution.t_roots is not None:
        return solution.t_roots[0]
    fit = interp1d(
        solution.solution[:, ODEIndex.v_θ],
        solution.angles,
        fill_value="extrapolate",
    )
    return fit(1.0)


class AnalysisError(DiscSolverError):
    """
    Error class for problems with analysis routines
    """
    pass


def distinct_color_map(size):
    """
    Generate a list of unique colours for matplotlib line/scatter plots
    """
    if size > len(TABLEAU_COLORS):
        return XKCD_COLORS.keys()
    return TABLEAU_COLORS.keys()


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
