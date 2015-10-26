# -*- coding: utf-8 -*-
"""
Solver component of Disc Solver
"""
import argparse

import arrow
import logbook
from logbook.compat import redirected_warnings, redirected_logging

from h5py import File

from .config import get_input, step_input
from .solution import create_soln_splitter, solver_generator
from .stepper import (
    binary_searcher, stepper_creator, writer_generator, cleanup_generator
)

from ..file_format import wrap_hdf5_file
from ..logging import logging_options, log_handler
from ..utils import allvars as vars

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
        config_inp = get_input(conffile=conffile)
        step_func = step_input(config_inp)
        if gen_file_name:
            output_file = config_inp.label + str(arrow.now()) + ".hdf5"
        with File(output_file) as f:
            soln_file = wrap_hdf5_file(f, version="latest", new=True)
            soln_file.root.config_input = config_inp
            soln_file.root.config_filename = str(conffile)
            writer = writer_generator(soln_file)
            cleanup = cleanup_generator(soln_file, writer)
            binary_searcher(
                solver_generator(), cleanup,
                stepper_creator(
                    writer, step_func,
                    create_soln_splitter("v_θ_deriv")
                ),
                config_inp,
            )
