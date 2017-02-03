# -*- coding: utf-8 -*-
"""
Solver component of Disc Solver
"""
import argparse
from pathlib import Path

import arrow
import logbook
from logbook.compat import redirected_warnings, redirected_logging

from h5preserve import open as h5open

from .config import (
    get_input_from_conffile, config_input_to_soln_input
)
from .stepper import solver as stepper_solver
from .single import solver as single_solver
from .dae_single import solver as dae_single_solver
from .mcmc import solver as mcmc_solver
from .sonic_minimise import solver as sonic_minimise_solver
from .utils import add_solver_arguments

from .. import __version__ as ds_version
from ..file_format import registries, Run
from ..logging import log_handler
from ..utils import expanded_path

log = logbook.Logger(__name__)


def solve(
    *, output_file, sonic_method, config_file, output_dir, store_internal
):
    """
    Main function to generate solution
    """
    config_input = get_input_from_conffile(config_file=config_file)
    run = Run(
        config_input=config_input,
        config_filename=str(config_file),
        disc_solver_version=ds_version,
    )

    if output_file is None:
        output_file = Path(config_input.label + str(arrow.now()) + ".hdf5")
    output_file = expanded_path(output_dir / output_file)

    with h5open(output_file, registries) as f:
        f["run"] = run
        if sonic_method == "step":
            stepper_solver(
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
        elif sonic_method == "mcmc":
            mcmc_solver(
                config_input_to_soln_input(config_input), run,
                store_internal=store_internal,
            )
        elif sonic_method == "sonic_minimise":
            sonic_minimise_solver(
                config_input_to_soln_input(config_input), run,
                store_internal=store_internal,
            )
        else:
            raise RuntimeError("No method chosen to cross sonic point")

    return output_file


def main():
    """
    Entry point for ds-soln
    """
    parser = argparse.ArgumentParser(description='Solver for DiscSolver')
    add_solver_arguments(parser)
    parser.add_argument("config_file")

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
