# -*- coding: utf-8 -*-
"""
Info command and associated code
"""
from sys import stdout

from ..utils import get_solutions
from .utils import (
    analyse_main_wrapper, analysis_func_wrapper, get_sonic_point,
    get_scale_height, AnalysisError, get_all_sonic_points,
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
def info_main(soln_file, *, group, soln_range):
    """
    Entry point for ds-info
    """
    # pylint: disable=missing-kwoa
    return info(
        soln_file, group=group, soln_range=soln_range, output_file=stdout,
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
            "DiscSolver version: {}".format(soln_file.disc_solver_version),
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
        if group == "input":
            print("input settings:", file=output_file)
            for name, value in vars(inp).items():
                print(INPUT_FORMAT.format(name, value), file=output_file)
        elif group == "initial-conditions":
            print("initial conditions:", file=output_file)
            for name, value in vars(init_con).items():
                print(INIT_FORMAT.format(name, value), file=output_file)
        else:
            slow_θ, sonic_θ, alfven_θ, fast_θ = get_all_sonic_points(
                soln_instance
            )

            if group == "crosses-points":
                if fast_θ is not None:
                    print(
                        "{}: fast".format(soln_file.config_input.label),
                        file=output_file
                    )
                elif alfven_θ is not None:
                    print(
                        "{}: alfven".format(soln_file.config_input.label),
                        file=output_file
                    )
                elif sonic_θ is not None:
                    print(
                        "{}: sonic".format(soln_file.config_input.label),
                        file=output_file
                    )
                elif slow_θ is not None:
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
                    "slow sonic point", slow_θ
                ), file=output_file)
                print(OTHER_FORMAT.format(
                    "sonic point", sonic_θ
                ), file=output_file)
                print(OTHER_FORMAT.format(
                    "alfven sonic point", alfven_θ
                ), file=output_file)
                print(OTHER_FORMAT.format(
                    "fast sonic point", fast_θ
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
