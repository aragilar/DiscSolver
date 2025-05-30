# -*- coding: utf-8 -*-
"""
csvrunner module
"""
from csv import DictWriter
from multiprocessing import current_process

from pympler.asizeof import asizeof

from . import SONIC_METHOD_MAP
from .utils import (
    SolverError, add_solver_arguments, get_csv_inputs, validate_overrides,
    add_overrides, add_worker_arguments, CSVWriterHelper,
)

from .. import __version__ as ds_version
from ..file_format import Run, ConfigInput, SOLUTION_INPUT_FIELDS
from ..float_handling import float_type
from ..utils import open_or_stream, main_entry_point_wrapper, nicer_mp_pool


# pylint: disable=too-few-public-methods
class SolutionFinder:
    """
    Generate best solution finder
    """
    def __init__(
        self, *, sonic_method, store_internal, overrides, **kwargs
    ):
        self.sonic_method = sonic_method
        self.store_internal = store_internal
        self.kwargs = kwargs
        self.overrides = overrides

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

        print(f"Run size in {process_name} is {asizeof(run)}")

        try:
            succeeded = sonic_solver(
                soln_input, run, store_internal=self.store_internal,
                **self.kwargs
            )
        except SolverError as e:
            print(str(e))
            return None

        print(f"Run size in {process_name} is {asizeof(run)}")

        if succeeded:
            if run.final_solution is None:
                print(run)
                return None
            return run.final_solution.solution_input.asdict()
        return None
# pylint: enable=too-few-public-methods


def csvrunner(
    *, output_file, input_file, nworkers=None, store_internal,
    sonic_method, overrides, label='', **kwargs
):
    """
    Find the best solution for the inputs given in the csv file.
    """
    inputs = get_csv_inputs(input_file, label=label)

    with open_or_stream(output_file, mode='a') as out:
        helper = CSVWriterHelper(out)
        csvwriter = DictWriter(
            helper, fieldnames=SOLUTION_INPUT_FIELDS, dialect="unix",
        )
        helper.add_metadata(dict(
            nworkers=nworkers, store_internal=store_internal,
            sonic_method=sonic_method,
        ))
        csvwriter.writeheader()
        out.flush()
        with nicer_mp_pool(nworkers) as pool:
            for best_input in pool.imap(SolutionFinder(
                sonic_method=sonic_method, store_internal=store_internal,
                overrides=overrides, **kwargs
            ), inputs):
                if best_input is None:
                    print("No final solution found for input")
                else:
                    csvwriter.writerow(best_input)
                    out.flush()


@main_entry_point_wrapper(description='csvrunner for DiscSolver')
def main(argv, parser):
    """
    Entry point for ds-csvrunner
    """
    add_solver_arguments(
        parser, store_internal=False, sonic_method="step", output_file='-',
    )
    add_worker_arguments(parser)
    parser.add_argument("input_file")

    args = parser.parse_args(argv)

    overrides = validate_overrides(args.override)

    csvrunner(
        input_file=args.input_file, nworkers=args.nworkers,
        sonic_method=args.sonic_method, store_internal=args.store_internal,
        overrides=overrides, output_file=args.output_file,
    )
