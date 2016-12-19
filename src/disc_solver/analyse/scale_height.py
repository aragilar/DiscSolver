# -*- coding: utf-8 -*-
"""
Scale-height command and associated code
"""
import argparse
from pathlib import Path

from logbook.compat import redirected_warnings, redirected_logging
from h5preserve import open as h5open
import matplotlib.pyplot as plt

from ..file_format import registries
from ..logging import log_handler, logging_options
from .utils import get_scale_height, get_sonic_point, savefig


def scale_height_main():
    """
    Entry point for ds-scale-height
    """
    parser = argparse.ArgumentParser(
        description="plot scale heights",
        argument_default=argparse.SUPPRESS,
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

        scale_height(input_files, **kwargs)


def scale_height(files, *, show, plot_filename, figargs):
    """
    Plot the height of the sonic point vs the height of the scale height
    """
    fig = generate_plot(files, figargs=figargs)
    if plot_filename is not None:
        savefig(fig, plot_filename)
    if show:
        plt.show()
    plt.close(fig)


def generate_plot(files, *, figargs):
    """
    Generate plot of the height of the sonic point vs the height of the scale
    height
    """
    scale_heights = []
    sonic_points = []
    for file in files:
        with h5open(file, registries) as f:
            run = f["run"]
        scale_heights.append(get_scale_height(run.final_solution))
        sonic_points.append(get_sonic_point(run.final_solution))

    fig, ax = plt.subplots(tight_layout=True, **figargs)
    ax.plot(scale_heights, sonic_points, ".")
    ax.set_xlabel("scale height")
    ax.set_ylabel("sonic point")
    return fig
