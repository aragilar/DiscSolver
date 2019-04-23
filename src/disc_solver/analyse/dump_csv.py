# -*- coding: utf-8 -*-
"""
dump command and associated code
"""
from csv import DictWriter

from attr import asdict

from ..file_format import SOLUTION_INPUT_FIELDS
from ..utils import get_solutions
from .utils import (
    analyse_main_wrapper, analysis_func_wrapper, open_or_stream,
)


def dump_parser(parser):
    """
    Add arguments for dump command to parser
    """
    parser.add_argument("--file", default='-')
    parser.add_argument("--with-header", action="store_true", default=False)
    return parser


def get_dump_args(args):
    """
    Parse dump args
    """
    return {
        "output_file": args.get("file", '-'),
        "with_header": args.get("with_header", False),
    }


@analyse_main_wrapper(
    "dump dumper for output from DiscSolver",
    dump_parser,
    cmd_parser_splitters={
        "dump_args": get_dump_args,
    }
)
def dump_main(soln_file, *, soln_range, dump_args):
    """
    Entry point for ds-dump
    """
    # pylint: disable=missing-kwoa
    return dump_csv(
        soln_file, soln_range=soln_range, **dump_args
    )


@analysis_func_wrapper
def dump_csv(soln_file, *, soln_range, output_file, with_header, **kwargs):
    """
    Dump csv of input to output_file
    """
    # pylint: disable=unused-argument
    soln_instance = get_solutions(soln_file, soln_range)
    inp = soln_instance.solution_input
    with open_or_stream(output_file, mode='a'):
        csvwriter = DictWriter(
            output_file, fieldnames=SOLUTION_INPUT_FIELDS, dialect="unix",
        )
        if with_header:
            csvwriter.writeheader()
        csvwriter.writerow(asdict(inp))
