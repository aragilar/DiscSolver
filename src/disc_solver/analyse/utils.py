# -*- coding: utf-8 -*-
"""
Utils for analysis code
"""
import argparse
from functools import wraps
from sys import exit

from logbook.compat import redirected_warnings, redirected_logging
from h5preserve import open

from ..file_format import registries
from ..logging import log_handler, logging_options
from ..utils import fspath, str_to_float


def single_solution_plotter(func):
    """
    Pulls out common elements of plots which take a single solution
    """
    @wraps(func)
    def plot_wrapper(h5file, solution, *args, title=None, **kwargs):
        """
        Wraps plot functions
        """
        if solution is None:
            solution = "0"
        fig = func(
            get_solutions(h5file, solution), *args, **kwargs
        )
        if title is None:
            fig.suptitle("{}:{}:{}".format(
                h5file.config_filename, h5file.config_input.label, solution
            ))
        else:
            fig.suptitle(title)
        return fig
    return plot_wrapper


def get_solutions(soln_file, soln_range):
    """
    Get solutions based on range
    """
    if soln_range is None:
        soln_range = "0"
    elif soln_range == "final":
        return soln_file.final_solution
    return soln_file.solutions[soln_range]


def common_plotting_options(parser):
    """
    Common plotting options to control appearance and output
    """
    parser.add_argument("--show", action="store_true", default=False)
    parser.add_argument("--filename")
    parser.add_argument("--figsize", nargs=2)
    parser.add_argument("--stop")
    parser.add_argument("--linestyle")
    return parser


def get_common_plot_args(args):
    """
    Extract the common plot options into the correct variables
    """
    figargs = {}
    if args.get("figsize") is not None:
        figargs["figsize"] = args["figsize"]

    return {
        "show": args.get("show", False),
        "plot_filename": args.get("filename"),
        "figargs": figargs,
        "stop": str_to_float(args.get("stop", "90")),
        "linestyle": args.get("linestyle", "-"),
    }


def savefig(fig, file, **kwargs):
    """
    Savefig wrapper
    """
    fig.savefig(fspath(file), **kwargs)


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
        def wrap_analysis_main():
            """
            Actual main function for analysis code, deals with parsers and
            logging
            """
            cmd_args = {}
            parser = argparse.ArgumentParser(
                description=cmd_description,
                argument_default=argparse.SUPPRESS,
            )
            parser.add_argument("soln_file")
            parser.add_argument("soln_range")
            logging_options(parser)
            cmd_parser = cmd_parser_func(parser)
            args = vars(cmd_parser.parse_args())
            for name, func in cmd_parser_splitters.items():
                cmd_args[name] = func(args)
            with log_handler(args):
                with redirected_warnings(), redirected_logging():
                    exit(cmd(
                        args["soln_file"],
                        soln_range=args["soln_range"],
                        **cmd_args
                    ))

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
        with open(filename, registries) as soln_file:
            return func(soln_file["run"], *args, **kwargs)
    return wrapper
