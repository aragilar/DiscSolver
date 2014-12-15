# -*- coding: utf-8 -*-
"""
Solver for PHD Project
"""

__version__ = "0.1"

import argparse
import tempfile
from types import SimpleNamespace

import logbook
import arrow

import numpy as np
import h5py

import matplotlib as mpl
mpl.use("Qt4Agg")
mpl.rcParams["backend.qt4"] = "PySide"
import matplotlib.pyplot as plt

from .analyse import generate_plot
from .solution import solution
from .config import define_conditions, get_input

log = logbook.Logger(__name__)


def solution_main(output_file=None):
    """
    Main function to generate solution
    """
    inp = get_input()
    if output_file:
        inp.output_file = output_file
    cons = define_conditions(inp)

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
        parser = argparse.ArgumentParser(description='Analyser for DiscSolver')
        parser.add_argument("output_file")

        if not kwargs:
            parser.add_argument("--show", action="store_true", default=False)
            parser.add_argument("--output_fig_file")

        output_file = parser.parse_args().output_file

        if not kwargs:
            kwargs = vars(parser.parse_args())

    with h5py.File(output_file) as f:
        for grp in f.values():
            inp = SimpleNamespace(**grp.attrs)  # pylint: disable=star-args
            cons = define_conditions(inp)
            angles = np.array(grp["angles"])
            soln = np.array(grp["solution"])
            fig = generate_plot(
                angles, soln, cons.B_norm, cons.v_norm, cons.ρ_norm
            )

            if kwargs.get("show"):
                plt.show()
            if kwargs.get("output_fig_file"):
                fig.savefig(kwargs.get("output_fig_file"))


def main():
    """
    The main function
    """
    with tempfile.NamedTemporaryFile() as output_file:
        solution_main(output_file=output_file.name)
        analyse_main(
            output_file=output_file.name, show=True,
            output_fig_file="plot.png"
        )
