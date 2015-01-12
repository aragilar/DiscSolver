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

import matplotlib as mpl
mpl.use("Qt4Agg")
mpl.rcParams["backend.qt4"] = "PySide"
import matplotlib.pyplot as plt

from .analyse import generate_plot, info
from .config import define_conditions, get_input
from .logging import logging_options, log_handler
from .solution import solution

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
    with log_handler(args), redirected_warnings(), redirected_logging():
        inps = get_input(conffile)
        for inp in inps:
            cons = define_conditions(inp)

            angles, soln = solution(
                cons.angles, cons.init_con, inp.β, cons.c_s,
                cons.norm_kepler_sq, cons.η_O, cons.η_A, cons.η_H,
                max_steps=inp.max_steps,
                taylor_stop_angle=inp.taylor_stop_angle
            )

            if not output_file:
                output_file = str(arrow.now())
            with h5py.File(output_file) as f:
                f['angles'] = angles
                f['solution'] = soln
                f.attrs.update(vars(inp))


def analyse_main(output_file=None, **kwargs):
    """
    Main function to analyse solution
    """
    if output_file is None:
        output_file, kwargs = analyse_parser(kwargs)
    make_plot = kwargs.get("show") or kwargs.get("output_fig_file")

    with redirected_warnings(), redirected_logging():
        with h5py.File(output_file) as f:
            inp = SimpleNamespace(**f.attrs)  # pylint: disable=star-args
            cons = define_conditions(inp)
            angles = np.array(f["angles"])
            soln = np.array(f["solution"])
            if make_plot:
                fig = generate_plot(angles, soln, inp, cons)
            if kwargs.get("show"):
                plt.show()
            if kwargs.get("output_fig_file"):
                fig.savefig(kwargs.get("output_fig_file"))
            if kwargs.get("info"):
                info(inp, cons)


def analyse_parser(kwargs):
    """
    CLI Parser for analyse
    """
    parser = argparse.ArgumentParser(description='Analyser for DiscSolver')
    parser.add_argument("output_file")

    if not kwargs:
        parser.add_argument(
            "--show", "-s", action="store_true", default=False
        )
        parser.add_argument("--output_fig_file", "-o")
        parser.add_argument(
            "--info", "-i", action="store_true", default=False
        )

    output_file = parser.parse_args().output_file

    if not kwargs:
        kwargs = vars(parser.parse_args())
    return output_file, kwargs


def main():
    """
    The main function
    """
    with tempfile.NamedTemporaryFile() as output_file:
        solution_main(output_file=output_file.name, ismain=False)
        analyse_main(
            output_file=output_file.name, show=True,
            output_fig_file="plot.png"
        )
