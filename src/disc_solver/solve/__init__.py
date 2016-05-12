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
from .jumper import solver as jumper_solver
from .single import solver as single_solver

from ..file_format import registries, Run
from ..logging import logging_options, log_handler
from ..utils import allvars as vars

log = logbook.Logger(__name__)


def solution_main(output_file=None, ismain=True, sonic_method=None):
    """
    Main function to generate solution
    """
    if ismain:
        parser = argparse.ArgumentParser(description='Solver for DiscSolver')
        parser.add_argument("conffile")
        parser.add_argument(
            "--sonic-method", choices=("step", "jump", "single"),
            default="step",
        )
        logging_options(parser)
        args = vars(parser.parse_args())
        conffile = args["conffile"]
        sonic_method = args["sonic_method"]
    else:
        args = {
            "quiet": True,
        }
        conffile = None
    gen_file_name = True if output_file is None else False
    with log_handler(args), redirected_warnings(), redirected_logging():
        config_inp = get_input_from_conffile(conffile=conffile)
        run = Run(config_input=config_inp, config_filename=str(conffile))

        if gen_file_name:
            output_file = config_inp.label + str(arrow.now()) + ".hdf5"
        if sonic_method == "step":
            step_func = step_input()
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
        elif sonic_method == "jump":
            jumper_solver(config_input_to_soln_input(config_inp), run)
        elif sonic_method == "single":
            single_solver(config_input_to_soln_input(config_inp), run)
        else:
            raise RuntimeError("No method chosen to cross sonic point")

        with open(output_file, registries) as f:
            f["run"] = run
