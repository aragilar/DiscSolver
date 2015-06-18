# -*- coding: utf-8 -*-
"""
Analysis component of Disc Solver
"""

import argparse
from types import SimpleNamespace

from logbook.compat import redirected_warnings, redirected_logging

import numpy as np
import h5py

from .plot_functions import (
    plot_options, deriv_plot_options, params_plot_options
)
from . import commands

from ..logging import logging_options, log_handler
from ..utils import cli_to_var


from ..solve.config import define_conditions  # ## NEEDS REMOVAL ## #


def analyse_main(output_file=None, **kwargs):
    """
    Main function to analyse solution
    """
    if output_file is None:
        output_file, kwargs = analyse_parser()
    else:
        kwargs["quiet"] = True

    with log_handler(kwargs), redirected_warnings(), redirected_logging():
        with h5py.File(output_file) as f:
            inp = SimpleNamespace(**f.attrs)
            cons = define_conditions(inp)
            angles = np.array(f["angles"])
            soln = np.array(f["solution"])
            kwargs["internal_data"] = f["internal_data"]
            command = getattr(commands, cli_to_var(kwargs["command"]))
            if command:
                command(inp, cons, angles, soln, kwargs)
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

    plot_parser = subparsers.add_parser("plot")
    plot_parser.add_argument("plot_filename")
    plot_options(plot_parser)

    info_parser = subparsers.add_parser("info")
    info_parser.add_argument("--input", action="store_true", default=False)
    info_parser.add_argument(
        "--initial-conditions", action="store_true", default=False
    )
    info_parser.add_argument(
        "--sound-ratio", action="store_true", default=False
    )
    info_parser.add_argument(
        "--sonic-points", action="store_true", default=False
    )

    deriv_show_parser = subparsers.add_parser("deriv-show")
    deriv_plot_options(deriv_show_parser)

    check_taylor_parser = subparsers.add_parser("check-taylor")
    check_taylor_parser.add_argument(
        "--show-values", action="store_true", default=False
    )

    params_show_parser = subparsers.add_parser("params-show")
    params_plot_options(params_show_parser)

    output_file = parser.parse_args().output_file

    args = vars(parser.parse_args())
    return output_file, args
