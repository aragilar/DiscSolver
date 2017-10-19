# -*- coding: utf-8 -*-
"""
Validate-input-plot command for DiscSolver
"""
import argparse
from pathlib import Path

from logbook.compat import redirected_warnings, redirected_logging
from h5preserve import open as h5open
import matplotlib.pyplot as plt
from corner import corner

from .. import __version__ as ds_version
from ..file_format import registries
from ..logging import log_handler, logging_options

from .utils import savefig

DIMENSIONS = [
    "γ", "v_rin_on_c_s", "c_s_on_v_k", "v_a_on_c_s",
]
DIM_RANGES = [
    (0, 0.01),
    (0, 2.5),
    (0.01, 0.08),
    (0, 2),
]


def validate_input_plot_main():
    """
    Entry point for ds-validate-input-plot
    """
    parser = argparse.ArgumentParser(
        description="plot inputs",
        argument_default=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--version', action='version', version='%(prog)s ' + ds_version
    )
    parser.add_argument("filenames")
    parser.add_argument("--show", action="store_true", default=False)
    parser.add_argument("--filename")
    parser.add_argument("--figsize", nargs=2)
    logging_options(parser)

    args = vars(parser.parse_args())

    with log_handler(args), redirected_warnings(), redirected_logging():
        figargs = {}
        if args.get("figsize") is not None:
            figargs["figsize"] = args["figsize"]

        kwargs = {
            "show": args.get("show", False),
            "plot_filename": args.get("filename"),
            "figargs": figargs,
        }

        with Path(args["filenames"]).open() as files:
            input_files = [Path(file.strip()) for file in files]

        validate_input_plot(input_files, **kwargs)


def validate_input_plot(soln, *, plot_filename=None, show=False, figargs):
    """
    Plot solution to file
    """
    fig = generate_plot(soln, figargs=figargs)

    if plot_filename is not None:
        savefig(fig, plot_filename)
    if show:
        plt.show()
    plt.close(fig)


def generate_plot(files, figargs):
    """
    Generate plot, with enough freedom to be able to format fig
    """
    data = []
    for file in files:
        with h5open(file, registries) as f:
            run = f["run"]
            data.extend([
                get_attrs(soln.solution_input, DIMENSIONS)
                for soln in run.solutions.values()
            ])
    fig = corner(
        data, range=DIM_RANGES, bins=50, plot_contours=False,
        labels=DIMENSIONS, **figargs
    )
    return fig


def get_attrs(obj, attrs):
    """
    Get all the attrs in `attrs` from obj
    """
    return [getattr(obj, attr) for attr in attrs]
