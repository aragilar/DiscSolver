# -*- coding: utf-8 -*-
"""
Solver for PHD Project
"""

__version__ = "0.1"

import argparse
from pprint import pprint
import signal
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
from .solution import solution

log = logbook.Logger(__name__)


def stack_info(frame):
    """Walk up stack printing line and locals"""
    print('f_code', frame.f_code)
    print('f_locals:')
    pprint(frame.f_locals)
    print()
    if frame.f_back:
        stack_info(frame.f_back)


def sig_handler(signum, frame):
    """Custom signal handler"""
    print(signum)
    stack_info(frame)


def solution_main(output_file=None, get_config_file=True):
    """
    Main function to generate solution
    """
    inp, = get_input(get_file=get_config_file)
    if output_file:
        inp.output_file = output_file
    cons = define_conditions(inp)

    signal.signal(signal.SIGINT, sig_handler)

    null_handler = logbook.NullHandler()
    with null_handler.applicationbound():
        angles, soln = solution(
            cons.angles, cons.init_con, inp.β, cons.c_s, cons.norm_kepler_sq,
            cons.η_O, cons.η_A, cons.η_H, max_steps=inp.max_steps,
            taylor_stop_angle=inp.taylor_stop_angle
        )

    with h5py.File(inp.output_file) as f:
        grp = f.create_group(str(arrow.now()))
        grp['angles'] = angles
        grp['solution'] = soln
        grp.attrs.update(vars(inp))


def analyse_main(output_file=None, **kwargs):
    """
    Main function to analyse solution
    """
    if output_file is None:
        output_file, kwargs = analyse_parser(kwargs)
    make_plot = kwargs.get("show") or kwargs.get("output_fig_file")

    with redirected_warnings(), redirected_logging():
        with h5py.File(output_file) as f:
            for grp in f.values():
                inp = SimpleNamespace(**grp.attrs)  # pylint: disable=star-args
                cons = define_conditions(inp)
                angles = np.array(grp["angles"])
                soln = np.array(grp["solution"])
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
    null_handler = logbook.NullHandler()
    with null_handler.applicationbound():
        with tempfile.NamedTemporaryFile() as output_file:
            solution_main(output_file=output_file.name, get_config_file=False)
            analyse_main(
                output_file=output_file.name, show=True,
                output_fig_file="plot.png"
            )
