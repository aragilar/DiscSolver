# -*- coding: utf-8 -*-
"""
Solver for PHD Project
"""

import tempfile

from .solve import solution_main
from .analyse import analyse_main


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

from ._version import version as __version__  # noqa
