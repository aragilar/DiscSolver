# -*- coding: utf-8 -*-
"""
Info command and associated code
"""
from sys import stdout

import numpy as np
from numpy import degrees

from ..utils import (
    is_supersonic, find_in_array, get_normalisation, ODEIndex,
    MAGNETIC_INDEXES, get_solutions,
)
from .utils import (
    analyse_main_wrapper, analysis_func_wrapper, get_sonic_point,
    get_scale_height, AnalysisError,
)
from ..solve.mcmc import get_logprob_of_soln

INPUT_FORMAT = " {: <20}: {}"
INIT_FORMAT = " {: <20}: {}"
OTHER_FORMAT = " {: <20}: {}"


def info_parser(parser):
    """
    Add arguments for info command to parser
    """
    parser.add_argument("group", choices=[
        "run", "status", "input", "initial-conditions", "sonic-points",
        "crosses-points", "sonic-on-scale", "solutions",
    ])
    return parser


@analyse_main_wrapper(
    "Info dumper for output from DiscSolver",
    info_parser,
    cmd_parser_splitters={
        "group": lambda args: args["group"]
    }
)
def info_main(soln_file, *, group, soln_range, filename):
    """
    Entry point for ds-info
    """
    return info(
        soln_file, group=group, soln_range=soln_range, output_file=stdout,
        filename=filename,
    )


@analysis_func_wrapper
def info(soln_file, *, group, soln_range, output_file, filename):
    """
    Output info about the solution
    """
    soln_instance = get_solutions(soln_file, soln_range)
    if group == "run":
        print("run properties:", file=output_file)
        print("filename: {}".format(filename), file=output_file)
        print(
            "label: {}".format(soln_file.config_input.label), file=output_file,
        )
        print(
            "config filename: {}".format(soln_file.config_filename),
            file=output_file
        )
        print(
            "sonic method: {}".format(soln_file.sonic_method),
            file=output_file,
        )
        print(
            "using E_r: {}".format(soln_file.use_E_r), file=output_file,
        )
        print(
            "number of solutions: {}".format(len(soln_file.solutions)),
            file=output_file,
        )
        print("config:", file=output_file)
        for name, value in vars(soln_file.config_input).items():
            print(INPUT_FORMAT.format(name, value), file=output_file)
    elif group == "status":
        print(
            "ODE return flag: {!s}".format(soln_instance.flag),
            file=output_file
        )
        print(
            "Coordinate System: {!s}".format(soln_instance.coordinate_system),
            file=output_file
        )
    elif group == "sonic-on-scale":
        print("{}: {}".format(
            soln_file.config_input.label,
            get_sonic_point(soln_instance) / get_scale_height(soln_instance)
        ))
    elif group == "solutions":
        for name, _ in sort_solutions(soln_file.solutions):
            print(name, file=output_file)
    else:
        inp = soln_instance.solution_input
        init_con = soln_instance.initial_conditions
        v_norm = get_normalisation(inp)["v_norm"]  # need to fix config here
        c_s = v_norm
        if group == "input":
            print("input settings:", file=output_file)
            for name, value in vars(inp).items():
                print(INPUT_FORMAT.format(name, value), file=output_file)
        elif group == "initial-conditions":
            print("initial conditions:", file=output_file)
            for name, value in vars(init_con).items():
                print(INIT_FORMAT.format(name, value), file=output_file)
        else:
            soln = soln_instance.solution
            angles = soln_instance.angles
            zero_soln = np.zeros(len(soln))
            v = np.array([zero_soln, zero_soln, soln[:, ODEIndex.v_θ]])
            slow_index = find_in_array(is_supersonic(
                v.T, soln[:, MAGNETIC_INDEXES], soln[:, ODEIndex.ρ],
                c_s, "slow"
            ), True)
            alfven_index = find_in_array(is_supersonic(
                v.T, soln[:, MAGNETIC_INDEXES], soln[:, ODEIndex.ρ],
                c_s, "alfven"
            ), True)
            fast_index = find_in_array(is_supersonic(
                v.T, soln[:, MAGNETIC_INDEXES], soln[:, ODEIndex.ρ],
                c_s, "fast"
            ), True)

            if group == "crosses-points":
                if fast_index:
                    print(
                        "{}: fast".format(soln_file.config_input.label),
                        file=output_file
                    )
                elif alfven_index:
                    print(
                        "{}: alfven".format(soln_file.config_input.label),
                        file=output_file
                    )
                elif slow_index:
                    print(
                        "{}: slow".format(soln_file.config_input.label),
                        file=output_file
                    )
                else:
                    print(
                        "{}: none".format(soln_file.config_input.label),
                        file=output_file
                    )
            elif group == "sonic-points":
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
                raise AnalysisError("Cannot find {}.".format(group))


def sort_solutions(solutions):
    """
    Sort solutions based on logprob of solution
    """
    return sorted(
        solutions.items(),
        key=lambda soln: get_logprob_of_soln(soln[1])
    )
