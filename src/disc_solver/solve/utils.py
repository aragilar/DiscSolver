# -*- coding: utf-8 -*-
"""
Utility function and classes for solver associated code
"""
from csv import DictReader, Sniffer

import attr
import logbook

from numpy import (
    any as np_any,
    diff,
    abs as np_abs,
    sqrt as np_sqrt,
)

from scikits.odes.sundials.cvode import StatusEnum

from ..file_format import CONFIG_FIELDS, SOLUTION_INPUT_FIELDS
from ..utils import (
    ODEIndex, DiscSolverError, MHD_Wave_Index, mhd_wave_speeds,
    MAGNETIC_INDEXES, expanded_path
)

log = logbook.Logger(__name__)


def error_handler(error_code, module, func, msg, user_data):
    """ drop all CVODE/IDA messages """
    # pylint: disable=unused-argument
    pass


def gen_mhd_sonic_point_rootfn(mhd_wave_type):
    """
    Finds specified mhd sonic point
    """
    mhd_wave_type = MHD_Wave_Index[mhd_wave_type]

    def rootfn(θ, params, out):
        """
        root function to find mhd sonic point
        """
        # pylint: disable=unused-argument
        wave_speeds = np_sqrt(mhd_wave_speeds(
            params[MAGNETIC_INDEXES], params[ODEIndex.ρ], 1
        ))
        out[0] = wave_speeds[mhd_wave_type] - params[ODEIndex.v_θ]
        return 0
    return rootfn


def validate_solution(soln):
    """
    Check that the solution returned is valid, even if the ode solver returned
    success
    """
    if soln.flag != StatusEnum.SUCCESS:
        return soln, False
    v_θ = soln.solution[:, ODEIndex.v_θ]
    ρ = soln.solution[:, ODEIndex.ρ]
    if np_any(v_θ < 0):
        return soln, False
    if np_any(diff(v_θ) < 0):
        return soln, False
    if np_any(ρ < 0):
        return soln, False
    return soln, True


def onroot_continue(*args):
    """
    Always continue after finding root
    """
    # pylint: disable=unused-argument
    return 0


def onroot_stop(*args):
    """
    Always stop after finding root
    """
    # pylint: disable=unused-argument
    return 1


def ontstop_continue(*args):
    """
    Always continue after finding tstop
    """
    # pylint: disable=unused-argument
    return 0


def ontstop_stop(*args):
    """
    Always stop after finding tstop
    """
    # pylint: disable=unused-argument
    return 1


def closest(a, max_val):
    """
    Return closest value in a to max_val
    """
    return a[closest_index(a, max_val)]


def closest_less_than(a, max_val):
    """
    Return closest value in a to max_val which is less than max_val
    """
    return a[closest_less_than_index(a, max_val)]


def closest_index(a, max_val):
    """
    Return index of closest value in a to max_val
    """
    return np_abs(a - max_val).argmin()


def closest_less_than_index(a, max_val):
    """
    Return index of closest value in a to max_val which is less than max_val
    """
    less_than = a < max_val
    return closest_index(a[less_than], max_val)


def add_solver_arguments(
    parser, *, store_internal=True, sonic_method='single'
):
    """
    Add common parser arguments for solver
    """
    parser.add_argument(
        "--sonic-method", choices=(
            "step", "single", "mcmc", "sonic_root", "hydrostatic", "mod_hydro",
            "fast_crosser",
        ), default=sonic_method,
    )
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--output-dir", default=".", type=expanded_path)
    internal_store_group = parser.add_mutually_exclusive_group()
    internal_store_group.add_argument(
        "--store-internal", action='store_true', default=store_internal
    )
    internal_store_group.add_argument(
        "--no-store-internal", action='store_false', dest="store_internal",
    )
    parser.add_argument("--override", action='append', nargs=2, default=[])
    parser.add_argument("--use-E_r", action='store_true', default=False)


def add_worker_arguments(parser):
    """
    Add arguments related to workers
    """
    parser.add_argument('-n', "--nworkers", default=None, type=int)


def velocity_stop_generator(target_velocity):
    """
    return function to stop at target velocity
    """
    def rootfn(θ, params, out):
        """
        root function to stop at required velocity
        """
        # pylint: disable=unused-argument
        out[0] = target_velocity - params[ODEIndex.v_θ]
        return 0
    return rootfn


def rad_to_scaled(obj, θ_scale):
    """
    Convert from radians to sonic scaled angle
    """
    if obj is None:
        return None
    return obj / θ_scale


def scaled_to_rad(obj, θ_scale):
    """
    Convert from sonic scaled angle to radians
    """
    if obj is None:
        return None
    return obj * θ_scale


class SolverError(DiscSolverError):
    """
    Error class for problems with solver routines
    """
    pass


def add_overrides(*, overrides, config_input):
    """
    Create new instance of `ConfigInput` with `overrides` added.
    """
    if overrides is None:
        return config_input
    log.info("overrides is {}".format(overrides))
    return attr.evolve(config_input, **overrides)


def validate_overrides(overrides):
    """
    Validate overrides passed as auguments
    """
    clean_overrides = {}
    for name, value in overrides:
        if name in CONFIG_FIELDS:
            clean_overrides[name] = value
        else:
            raise SolverError("Override incorrect: no such field {}".format(
                name
            ))
    return clean_overrides


def add_labels(seq, *, label=''):
    """
    Add labels
    """
    new_seq = []
    for d in seq:
        d['label'] = label
        new_seq.append(d)
    return new_seq


def has_csv_header(file):
    """
    Checks if csv file has header
    """
    has_header = Sniffer().has_header(file.readline())
    file.seek(0)
    return has_header


def get_csv_inputs(input_file, label=''):
    """
    Get inputs from csv file
    """
    with open(input_file) as infile:
        has_header = has_csv_header(infile)
        inputs = add_labels(DictReader(
            infile, fieldnames=SOLUTION_INPUT_FIELDS, dialect="unix",
        ), label=label)

    if has_header:
        inputs = inputs[1:]

    return inputs
