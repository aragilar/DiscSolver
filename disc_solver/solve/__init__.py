# -*- coding: utf-8 -*-
"""
Solver component of Disc Solver
"""
import argparse

import arrow
import logbook
from logbook.compat import redirected_warnings, redirected_logging

from .config import define_conditions, get_input
from .solution import solution

from ..hdf5_wrapper import soln_open
from ..logging import logging_options, log_handler

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
        inps = get_input(conffile)
        for inp in inps:
            log.notice(inp.label)
            cons = define_conditions(inp)

            angles, soln, internal_data, soln_props = solution(
                cons.angles, cons.init_con, inp.β, cons.c_s,
                cons.norm_kepler_sq, cons.η_O, cons.η_A, cons.η_H,
                max_steps=inp.max_steps,
                taylor_stop_angle=inp.taylor_stop_angle
            )
            current_time = str(arrow.now())
            if gen_file_name:
                output_file = inp.label + current_time + ".hdf5"
            with soln_open(output_file) as f:
                f.angles = angles
                f.solution = soln
                f.config_input = inp
                f.initial_conditions = cons
                f.config_filename = conffile
                f.config_label = inp.label
                f.time = current_time
                f.solution_properties = soln_props
                f.internal_data = internal_data
