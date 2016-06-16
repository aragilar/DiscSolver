# -*- coding: utf-8 -*-
"""
Info command and associated code
"""
from sys import stdin

import numpy as np
from numpy import degrees

from ..utils import (
    is_supersonic, find_in_array, get_normalisation, allvars as vars
)
from .utils import get_solutions, analyse_main_wrapper, analysis_func_wrapper

INPUT_FORMAT = " {: <20}: {}"
INIT_FORMAT = " {: <20}: {}"
OTHER_FORMAT = " {: <20}: {}"


def info_parser(parser):
    """
    Add arguments for info command to parser
    """
    parser.add_argument("group")
    return parser


@analyse_main_wrapper(
    "Info dumper for output from DiscSolver",
    info_parser,
    cmd_parser_splitters={
        "group", lambda args: args["group"]
    }
)
def info_main(soln_file, *, group, soln_range):
    """
    Entry point for ds-info
    """
    return info(
        soln_file, group=group, soln_range=soln_range, output_file=stdin
    )


@analysis_func_wrapper
def info(soln_file, *, group, soln_range, output_file):
    """
    Output info about the solution
    """
    soln_instance = get_solutions(soln_file, soln_range)
    if group == "run":
        print("run properties:", file=output_file)
        print(
            "label: {}".format(soln_file.config_input.label),
            file=output_file,
        )
        print(
            "config filename: {}".format(soln_file.config_filename),
            file=output_file
        )
        print(
            "number of solutions: {}".format(len(soln_file.solutions)),
            file=output_file
        )
    elif group == "status":
        print(
            "ODE return flag: {!s}".format(soln_instance.flag),
            file=output_file
        )
        print(
            "Coordinate System: {!s}".format(soln_instance.coordinate_system),
            file=output_file
        )
    else:
        inp = soln_instance.solution_input
        init_con = soln_instance.initial_conditions
        v_norm = get_normalisation(inp)["v_norm"]  # need to fix config here
        c_s = init_con.c_s * v_norm
        if group == "input":
            print("input settings:", file=output_file)
            for name, value in vars(inp).items():
                print(INPUT_FORMAT.format(name, value), file=output_file)
        elif group == "initial_conditions":
            print("initial conditions:", file=output_file)
            for name, value in vars(init_con).items():
                print(INIT_FORMAT.format(name, value), file=output_file)
        elif group == "sonic_points":
            soln = soln_instance.solution
            angles = soln_instance.angles
            zero_soln = np.zeros(len(soln))
            v = np.array([zero_soln, zero_soln, soln[:, 5]])
            slow_index = find_in_array(is_supersonic(
                v.T, soln[:, 0:3], soln[:, 6], c_s, "slow"
            ), True)
            alfven_index = find_in_array(is_supersonic(
                v.T, soln[:, 0:3], soln[:, 6], c_s, "alfven"
            ), True)
            fast_index = find_in_array(is_supersonic(
                v.T, soln[:, 0:3], soln[:, 6], c_s, "fast"
            ), True)
            print(OTHER_FORMAT.format(
                "slow sonic point",
                degrees(angles[slow_index]) if slow_index else None
            ), file=output_file)
            print(OTHER_FORMAT.format(
                "alfven sonic point",
                degrees(angles[alfven_index]) if alfven_index else None
            ), file=output_file)
            print(OTHER_FORMAT.format(
                "fast sonic point",
                degrees(angles[fast_index]) if fast_index else None
            ), file=output_file)
        else:
            print("Cannot find {}.".format(group), file=output_file)
