# -*- coding: utf-8 -*-
"""
Solver component of Disc Solver
"""
import argparse

import arrow
import logbook
from logbook.compat import redirected_warnings, redirected_logging

from h5preserve import open

from .config import (
    get_input_from_conffile, config_input_to_soln_input
)
from .stepper import (
    binary_searcher, stepper_creator, writer_generator, cleanup_generator,
    create_soln_splitter, solver_generator, step_input,
)

from ..file_format import registries, Run
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
        config_inp = get_input_from_conffile(conffile=conffile)
        step_func = step_input()
        if gen_file_name:
            output_file = config_inp.label + str(arrow.now()) + ".hdf5"
        run = Run(config_input=config_inp, config_filename=str(conffile))
        writer = writer_generator(run)
        cleanup = cleanup_generator(run, writer)
        binary_searcher(
            solver_generator(), cleanup,
            stepper_creator(
                writer, step_func,
                create_soln_splitter("v_θ_deriv")
            ),
            config_input_to_soln_input(config_inp),
        )
        with open(output_file, registries) as f:
            f["run"] = run
