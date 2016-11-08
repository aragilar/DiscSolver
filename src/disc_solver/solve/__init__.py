# -*- coding: utf-8 -*-
"""
Solver component of Disc Solver
"""
import argparse
from pathlib import Path

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
from .dae_single import solver as dae_single_solver

from ..file_format import registries, Run
from ..logging import logging_options, log_handler
from ..utils import allvars as vars, expanded_path

log = logbook.Logger(__name__)


def solve(
    *, output_file, sonic_method, config_file, output_dir, store_internal
):
    """
    Main function to generate solution
    """
    config_input = get_input_from_conffile(config_file=config_file)
    run = Run(config_input=config_input, config_filename=str(config_file))

    if output_file is None:
        output_file = Path(config_input.label + str(arrow.now()) + ".hdf5")
    output_file = expanded_path(output_dir / output_file)

    if sonic_method == "step":
        step_func = step_input()
        writer = writer_generator(run)
        cleanup = cleanup_generator(run, writer)
        binary_searcher(
            solver_generator(store_internal=store_internal), cleanup,
            stepper_creator(
                writer, step_func,
                create_soln_splitter("v_θ_deriv")
            ),
            config_input_to_soln_input(config_input),
        )
    elif sonic_method == "jump":
        jumper_solver(
            config_input_to_soln_input(config_input), run,
            store_internal=store_internal,
        )
    elif sonic_method == "single":
        single_solver(
            config_input_to_soln_input(config_input), run,
            store_internal=store_internal,
        )
    elif sonic_method == "dae_single":
        dae_single_solver(
            config_input_to_soln_input(config_input), run,
            store_internal=store_internal,
        )
    else:
        raise RuntimeError("No method chosen to cross sonic point")

    with open(output_file, registries) as f:
        f["run"] = run

    return output_file


def main():
    """
    Entry point for ds-soln
    """
    parser = argparse.ArgumentParser(description='Solver for DiscSolver')
    parser.add_argument("config_file")
    parser.add_argument(
        "--sonic-method", choices=("step", "jump", "single", "dae_single"),
        default="single",
    )
    parser.add_argument("--output-file")
    parser.add_argument(
        "--output-dir", default=".",
    )
    internal_store_group = parser.add_mutually_exclusive_group()
    internal_store_group.add_argument(
        "--store-internal", action='store_true', default=True
    )
    internal_store_group.add_argument(
        "--no-store-internal", action='store_false', dest="store_internal",
    )
    logging_options(parser)
    args = vars(parser.parse_args())

    config_file = expanded_path(args["config_file"])
    output_dir = expanded_path(args["output_dir"])
    sonic_method = args["sonic_method"]
    output_file = args.get("output_file", None)
    store_internal = args.get("store_internal", True)

    with log_handler(args), redirected_warnings(), redirected_logging():
        print(solve(
            output_file=output_file, sonic_method=sonic_method,
            config_file=config_file, output_dir=output_dir,
            store_internal=store_internal,
        ))
