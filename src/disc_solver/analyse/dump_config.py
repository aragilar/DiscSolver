# -*- coding: utf-8 -*-
"""
dump command and associated code
"""
from ..utils import get_solutions, open_or_stream
from .utils import analyse_main_wrapper, analysis_func_wrapper


def dump_parser(parser):
    """
    Add arguments for dump command to parser
    """
    parser.add_argument("--file", default='-')
    return parser


def get_dump_args(args):
    """
    Parse dump args
    """
    return {
        "output_file": args.get("file", '-'),
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
    Entry point for ds-dumpcfg
    """
    # pylint: disable=missing-kwoa
    return dump_cfg(
        soln_file, soln_range=soln_range, **dump_args
    )


@analysis_func_wrapper
def dump_cfg(soln_file, *, soln_range, output_file, **kwargs):
    """
    Dump cfg of input to output_file
    """
    # pylint: disable=unused-argument
    soln_instance = get_solutions(soln_file, soln_range)
    inp = soln_instance.solution_input
    with open_or_stream(output_file, mode='w') as out:
        inp.to_config_input(soln_file.config_input.label).to_conf_file(out)
