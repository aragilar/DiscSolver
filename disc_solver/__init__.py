# -*- coding: utf-8 -*-
"""
Solver for PHD Project
"""

__version__ = "0.1"

import argparse
import tempfile
from types import SimpleNamespace

import arrow
import logbook
from logbook.compat import redirected_warnings, redirected_logging

import numpy as np
import h5py

from .analyse import plot_options, deriv_plot_options, params_plot_options
from .analyse import commands as analyse_commands
from .config import define_conditions, get_input
from .logging import logging_options, log_handler
from .solution import solution
from .utils import cli_to_var

log = logbook.Logger(__name__)


def solution_main(output_file=None, ismain=True):
    """
    Main function to generate solution
    """
    if ismain:
        parser = argparse.ArgumentParser(description='Solver for DiscSolver')
        parser.add_argument("conffile")
        logging_options(parser)
        args = vars(parser.parse_args())
        conffile = args["conffile"]
    else:
        args = {
            "quiet": True,
        }
        conffile = None
    gen_file_name = True if output_file is None else False
    with log_handler(args), redirected_warnings(), redirected_logging():
        inps = get_input(conffile)
        for inp in inps:
            cons = define_conditions(inp)

            angles, soln, internal_data = solution(
                cons.angles, cons.init_con, inp.β, cons.c_s,
                cons.norm_kepler_sq, cons.η_O, cons.η_A, cons.η_H,
                max_steps=inp.max_steps,
                taylor_stop_angle=inp.taylor_stop_angle
            )

            if gen_file_name:
                output_file = inp.label + str(arrow.now()) + ".hdf5"
            with h5py.File(output_file) as f:
                f['angles'] = angles
                f['solution'] = soln
                f.attrs.update(vars(inp))
                f.create_group('internal_data')
                f['internal_data'].update(internal_data)


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
            inp = SimpleNamespace(**f.attrs)  # pylint: disable=star-args
            cons = define_conditions(inp)
            angles = np.array(f["angles"])
            soln = np.array(f["solution"])
            kwargs["internal_data"] = f["internal_data"]
            command = getattr(analyse_commands, cli_to_var(kwargs["command"]))
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


def main():
    """
    The main function
    """
    with tempfile.NamedTemporaryFile() as output_file:
        solution_main(output_file=output_file.name, ismain=False)
        analyse_main(output_file=output_file.name, command="show")
        analyse_main(
            output_file=output_file.name, command="plot",
            plot_filename="plot.png"
        )
