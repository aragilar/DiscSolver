# -*- coding: utf-8 -*-
"""
Solver for PHD Project
"""

import tempfile
from pathlib import Path

from .solve import solve
from .analyse import analyse_main

PLOT_FILE = "plot.png"


def main():
    """
    The main function
    """
    for method in ("step",):
        with tempfile.TemporaryDirectory() as workdir:
            h5file = solve(
                output_dir=Path(workdir), sonic_method=method,
                config_file=None, output_file=None
            )
            analyse_main(output_file=h5file, command="show")
            analyse_main(
                output_file=h5file, command="plot",
                plot_filename=Path(workdir, PLOT_FILE),
            )
            analyse_main(
                output_file=h5file, command="info", input=True,
                initial_conditions=True, sound_ratio=True, sonic_points=True,
            )
            analyse_main(output_file=h5file, command="deriv_show")
            analyse_main(output_file=h5file, command="check_taylor")
            analyse_main(output_file=h5file, command="params_show")

from ._version import version as __version__  # noqa
