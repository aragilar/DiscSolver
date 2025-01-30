# -*- coding: utf-8 -*-
"""
hdf5runner module
"""
from multiprocessing import current_process
from pathlib import Path

import arrow
from pympler.asizeof import asizeof

from h5preserve import open as h5open

from . import SONIC_METHOD_MAP
from .utils import (
    SolverError, add_solver_arguments, get_csv_inputs, validate_overrides,
    add_overrides, add_worker_arguments,
)

from .. import __version__ as ds_version
from ..file_format import registries, Run, ConfigInput
from ..float_handling import float_type
from ..utils import expanded_path, main_entry_point_wrapper, nicer_mp_pool


# pylint: disable=too-few-public-methods
class SolutionFinder:
    """
    Generate best solution finder
    """
    def __init__(
        self, *, sonic_method, store_internal, output_file=None,
        output_dir, overrides, **kwargs
    ):
        self.sonic_method = sonic_method
        self.store_internal = store_internal
        self.output_file = output_file
        self.output_dir = output_dir
        self.overrides = overrides
        self.kwargs = kwargs

    def __call__(self, input_dict):
        """
        Function to be mapped over
        """
        sonic_solver = SONIC_METHOD_MAP.get(self.sonic_method)
        if sonic_solver is None:
            raise SolverError("No method chosen to cross sonic point")

        soln_filename = input_dict.pop("filename", None)
        soln_range = input_dict.pop("solution_name", None)

        process_name = current_process().name
        config_input = add_overrides(
            config_input=ConfigInput(**input_dict), overrides=self.overrides
        )
        print(f"Config is {config_input}")
        soln_input = config_input.to_soln_input()
        run = Run(
            config_input=config_input,
            config_filename=None,
            disc_solver_version=ds_version,
            float_type=str(float_type),
            sonic_method=self.sonic_method,
            use_E_r=soln_input.use_E_r,
            based_on_solution_filename=soln_filename,
            based_on_solution_solution_name=soln_range,
        )

        if self.output_file is None:
            output_file = Path(
                config_input.label + str(arrow.now()) +
                str(process_name) + ".hdf5"
            )
        else:
            output_file = self.output_file
        output_file = expanded_path(self.output_dir / output_file)

        print(f"Initial run size in {process_name} is {asizeof(run)}")

        try:
            with h5open(output_file, registries, mode='x') as f:
                f["run"] = run
                succeeded = sonic_solver(
                    soln_input, run,
                    store_internal=self.store_internal, **self.kwargs
                )
                run.finalise()
        except SolverError as e:
            print(str(e))
            return None

        print(f"Final run size in {process_name} is {asizeof(run)}")

        return succeeded
# pylint: enable=too-few-public-methods


def hdf5runner(
    *, output_file=None, input_file, nworkers=None, output_dir,
    store_internal, sonic_method, overrides, label='', **kwargs
):
    """
    Find the best solution for the inputs given in the csv file.
    """
    inputs = get_csv_inputs(input_file, label=label)

    with nicer_mp_pool(nworkers) as pool:
        for result in pool.imap(SolutionFinder(
            output_file=output_file, output_dir=output_dir,
            sonic_method=sonic_method, store_internal=store_internal,
            overrides=overrides, **kwargs
        ), inputs):
            if not result:
                print("Solver failed for input")


@main_entry_point_wrapper(description='hdf5runner for DiscSolver')
def main(argv, parser):
    """
    Entry point for ds-hdf5runner
    """
    add_solver_arguments(parser)
    add_worker_arguments(parser)
    parser.add_argument("input_file", type=expanded_path)
    parser.add_argument('-l', "--label", default='', type=str)

    args = parser.parse_args(argv)

    overrides = validate_overrides(args.override)

    hdf5runner(
        output_dir=args.output_dir, input_file=args.input_file,
        nworkers=args.nworkers, sonic_method=args.sonic_method,
        store_internal=args.store_internal, overrides=overrides,
        label=args.label,
    )
