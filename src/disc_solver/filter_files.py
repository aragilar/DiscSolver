# -*- coding: utf-8 -*-
"""
Solver component of Disc Solver
"""
import argparse
from pathlib import Path
from sys import stdin

import logbook
from logbook.compat import redirected_warnings, redirected_logging

from h5preserve import open as h5open

from . import __version__ as ds_version
from .file_format import registries
from .logging import logging_options, log_handler
from .utils import ODEIndex

log = logbook.Logger(__name__)


def get_level(file):
    """
    Get value for level
    """
    with h5open(file, registries) as f:
        return max(f["run"].final_solution.solution[:, ODEIndex.v_θ])


def level_wrapper(output_path, level_list):
    """
    level generator based on input
    """
    def output_wrapper(output_file):
        """
        wrap output file for filter
        """
        f = output_file.open("a")

        def filter_func(path):
            """
            function for filter
            """
            print(str(path), file=f)
        return filter_func

    return {
        float(item[0]): output_wrapper(Path(output_path, item[1]))
        for item in level_list
    }


def filter_files(*, files, levels):
    """
    Filter files based on a property
    """
    for file in files:
        file_level = get_level(file)
        for level, func in levels.items():
            if file_level >= level:
                yield func(file)


def main():
    """
    Entry point for ds-filter-files
    """
    parser = argparse.ArgumentParser(
        description='Config Generator for DiscSolver'
    )
    parser.add_argument(
        '--version', action='version', version='%(prog)s ' + ds_version
    )
    parser.add_argument("--output-path", default=".")
    parser.add_argument("--level", action="append", nargs=2)
    logging_options(parser)
    args = vars(parser.parse_args())
    with log_handler(args), redirected_warnings(), redirected_logging():
        for output in filter_files(
            files=(Path(f.strip()) for f in stdin),
            levels=level_wrapper(args["output_path"], args["level"]),
        ):
            print(output)
