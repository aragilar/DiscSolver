# -*- coding: utf-8 -*-
"""
csvrunner module
"""

import argparse
from csv import DictWriter, DictReader, Sniffer
from multiprocessing import Pool
from warnings import warn

from . import SONIC_METHOD_MAP
from .utils import SolverError

from .. import __version__ as ds_version
from ..file_format import Run, ConfigInput, SOLUTION_INPUT_FIELDS
from ..float_handling import float_type
from ..utils import open_or_stream


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

        if succeeded:
            return run.final_solution.solution_input.asdict()
        return None
# pylint: enable=too-few-public-methods


def add_labels(seq):
    """
    Add labels
    """
    new_seq = []
    for d in seq:
        d['label'] = ''
        new_seq.append(d)
    return new_seq


def has_csv_header(file):
    """
    Checks if csv file has header
    """
    has_header = Sniffer().has_header(file.readline())
    file.seek(0)
    return has_header


def csvrunner(*, output_file, input_file, nworkers=None, **kwargs):
    """
    Find the best solution for the inputs given in the csv file.
    """
    with open(input_file) as infile:
        has_header = has_csv_header(infile)
        inputs = add_labels(DictReader(
            infile, fieldnames=SOLUTION_INPUT_FIELDS, dialect="unix",
        ))

    if has_header:
        inputs = inputs[1:]

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


def main():
    """
    Entry point for ds-csvrunner
    """
    parser = argparse.ArgumentParser(description='Solver for DiscSolver')
    parser.add_argument("input_file")
    parser.add_argument("--output-file", default='-')
    parser.add_argument('-n', "--nworkers", default=None, type=int)

    args = parser.parse_args()

    csvrunner(
        output_file=args.output_file,
        input_file=args.input_file,
        nworkers=args.nworkers,
    )
