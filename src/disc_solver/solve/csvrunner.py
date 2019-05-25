# -*- coding: utf-8 -*-
"""
csvrunner module
"""

from csv import DictWriter
from multiprocessing import Pool, current_process
from warnings import warn

from pympler.asizeof import asizeof

from . import SONIC_METHOD_MAP
from .utils import SolverError, get_csv_inputs, add_worker_arguments

from .. import __version__ as ds_version
from ..file_format import Run, ConfigInput, SOLUTION_INPUT_FIELDS
from ..float_handling import float_type
from ..utils import open_or_stream, main_entry_point_wrapper


# pylint: disable=too-few-public-methods
class SolutionFinder:
    """
    Generate best solution finder
    """
    def __init__(self, *, sonic_method='step', store_internal=False, **kwargs):
        self.sonic_method = sonic_method
        self.store_internal = store_internal
        self.kwargs = kwargs

    def __call__(self, input_dict):
        """
        Function to be mapped over
        """
        run = Run(
            config_input=None,
            config_filename=None,
            disc_solver_version=ds_version,
            float_type=str(float_type),
            sonic_method=self.sonic_method,
            use_E_r=False,
        )
        process_name = current_process().name
        print(f"Run size in {process_name} is {asizeof(run)}")
        soln_input = ConfigInput(**input_dict).to_soln_input()
        sonic_solver = SONIC_METHOD_MAP.get(self.sonic_method)
        if sonic_solver is None:
            raise SolverError("No method chosen to cross sonic point")

        try:
            succeeded = sonic_solver(
                soln_input, run, store_internal=self.store_internal,
                **self.kwargs
            )
        except SolverError as e:
            warn(str(e))
            return None

        print(f"Run size in {process_name} is {asizeof(run)}")

        if succeeded:
            return run.final_solution.solution_input.asdict()
        return None
# pylint: enable=too-few-public-methods


def csvrunner(*, output_file, input_file, nworkers=None, **kwargs):
    """
    Find the best solution for the inputs given in the csv file.
    """
    inputs = get_csv_inputs(input_file)

    with open_or_stream(output_file, mode='a') as out:
        csvwriter = DictWriter(
            out, fieldnames=SOLUTION_INPUT_FIELDS, dialect="unix",
        )
        csvwriter.writeheader()
        out.flush()
        with Pool(nworkers) as pool:
            for best_input in pool.imap(SolutionFinder(**kwargs), inputs):
                if best_input is None:
                    warn("No final solution found for input")
                else:
                    csvwriter.writerow(best_input)
                    out.flush()


@main_entry_point_wrapper(description='csvrunner for DiscSolver')
def main(argv, parser):
    """
    Entry point for ds-csvrunner
    """
    add_worker_arguments(parser)
    parser.add_argument("input_file")
    parser.add_argument("--output-file", default='-')

    args = parser.parse_args(argv)

    csvrunner(
        output_file=args.output_file,
        input_file=args.input_file,
        nworkers=args.nworkers,
    )
