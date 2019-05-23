# -*- coding: utf-8 -*-
"""
hdf5runner module
"""
import argparse
from multiprocessing import Pool, current_process
from pathlib import Path
from warnings import warn

import arrow
from pympler.asizeof import asizeof

from h5preserve import open as h5open

from . import SONIC_METHOD_MAP
from .utils import (
    SolverError, add_solver_arguments, get_csv_inputs, validate_overrides,
    add_overrides,
)

from .. import __version__ as ds_version
from ..file_format import registries, Run, ConfigInput
from ..float_handling import float_type
from ..utils import expanded_path


# pylint: disable=too-few-public-methods
class SolutionFinder:
    """
    Generate best solution finder
    """
    def __init__(
        self, *, sonic_method, store_internal, output_file=None,
        output_dir, use_E_r, overrides, **kwargs
    ):
        self.sonic_method = sonic_method
        self.store_internal = store_internal
        self.output_file = output_file
        self.output_dir = output_dir
        self.use_E_r = use_E_r
        self.overrides = overrides
        self.kwargs = kwargs

    def __call__(self, input_dict):
        """
        Function to be mapped over
        """
        sonic_solver = SONIC_METHOD_MAP.get(self.sonic_method)
        if sonic_solver is None:
            raise SolverError("No method chosen to cross sonic point")

        process_name = current_process().name
        config_input = add_overrides(
            config_input=ConfigInput(**input_dict), overrides=self.overrides
        )

        run = Run(
            config_input=config_input,
            config_filename=None,
            disc_solver_version=ds_version,
            float_type=str(float_type),
            sonic_method=self.sonic_method,
            use_E_r=self.use_E_r,
        )

        if self.output_file is None:
            output_file = Path(
                config_input.label + str(arrow.now()) +
                str(process_name) + ".hdf5"
            )
        else:
            output_file = self.output_file
        output_file = expanded_path(self.output_dir / output_file)

        print(f"Run size in {process_name} is {asizeof(run)}")

        try:
            with h5open(output_file, registries, mode='x') as f:
                f["run"] = run
                succeeded = sonic_solver(
                    config_input.to_soln_input(), run,
                    store_internal=self.store_internal, **self.kwargs
                )
        except SolverError as e:
            warn(str(e))
            return None

        print(f"Run size in {process_name} is {asizeof(run)}")

        return succeeded
# pylint: enable=too-few-public-methods


def hdf5runner(
    *, output_file=None, input_file, nworkers=None, use_E_r, output_dir,
    store_internal, sonic_method, overrides, label='', **kwargs
):
    """
    Find the best solution for the inputs given in the csv file.
    """
    inputs = get_csv_inputs(input_file, label=label)

    with Pool(nworkers) as pool:
        for result in pool.imap(SolutionFinder(
            output_file=output_file, output_dir=output_dir,
            sonic_method=sonic_method, store_internal=store_internal,
            use_E_r=use_E_r, overrides=overrides, **kwargs
        ), inputs):
            if not result:
                warn("Solver failed for input")


def main():
    """
    Entry point for ds-hdf5runner
    """
    parser = argparse.ArgumentParser(description='Solver for DiscSolver')
    add_solver_arguments(parser)
    parser.add_argument("input_file")
    parser.add_argument('-n', "--nworkers", default=None, type=int)
    parser.add_argument('-l', "--label", default='', type=str)

    args = parser.parse_args()

    output_dir = expanded_path(args.output_dir)
    overrides = validate_overrides(args.override)

    hdf5runner(
        output_dir=output_dir, input_file=args.input_file,
        nworkers=args.nworkers, sonic_method=args.sonic_method,
        store_internal=args.store_internal, overrides=overrides,
        use_E_r=args.use_E_r, label=args.label,
    )
