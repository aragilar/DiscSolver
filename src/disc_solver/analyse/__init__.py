# -*- coding: utf-8 -*-
"""
Analysis component of Disc Solver
"""

import argparse

from logbook.compat import redirected_warnings, redirected_logging

from h5py import File

from . import commands
from .plot_functions import (
    plot_options, deriv_plot_options, params_plot_options
)

from ..file_format import wrap_hdf5_file
from ..logging import logging_options, log_handler
from ..utils import cli_to_var
from ..utils import allvars as vars


def analyse_main(output_file=None, **kwargs):
    """
    Main function to analyse solution
    """
    if output_file is None:
        output_file, kwargs = analyse_parser()
    else:
        kwargs["quiet"] = True

    with log_handler(kwargs), redirected_warnings(), redirected_logging():
        with File(output_file) as f:
            soln_file = wrap_hdf5_file(f)
            command = getattr(commands, cli_to_var(kwargs["command"]))
            if command:
                command(soln_file, kwargs)
            else:
                raise NotImplementedError(kwargs["command"])


def analyse_parser():
    """
    CLI Parser for analyse
    """
    parser = argparse.ArgumentParser(description='Analyser for DiscSolver')
    parser.add_argument("output_file")

    logging_options(parser)

    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    show_parser = subparsers.add_parser("show")
    plot_options(show_parser)
    add_range(show_parser)

    plot_parser = subparsers.add_parser("plot")
    plot_parser.add_argument("plot_filename")
    plot_parser.add_argument("--figsize", nargs=2)
    plot_options(plot_parser)
    add_range(plot_parser)

    info_parser = subparsers.add_parser("info")
    info_parser.add_argument("--input", action="store_true", default=False)
    info_parser.add_argument(
        "--initial-conditions", action="store_true", default=False
    )
    info_parser.add_argument(
        "--sonic-points", action="store_true", default=False
    )
    add_range(info_parser)

    deriv_show_parser = subparsers.add_parser("deriv-show")
    deriv_plot_options(deriv_show_parser)
    add_range(deriv_show_parser)

    check_taylor_parser = subparsers.add_parser("check-taylor")
    check_taylor_parser.add_argument(
        "--show-values", action="store_true", default=False
    )
    add_range(check_taylor_parser)

    params_show_parser = subparsers.add_parser("params-show")
    params_plot_options(params_show_parser)
    add_range(params_show_parser)

    plot_acc_parser = subparsers.add_parser("plot-acc")
    add_range(plot_acc_parser)

    output_file = parser.parse_args().output_file

    args = vars(parser.parse_args())
    return output_file, args


def add_range(parser):
    """
    Add support for grabbing a range of solutions
    """
    parser.add_argument("soln_range")
