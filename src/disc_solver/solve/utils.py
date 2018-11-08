# -*- coding: utf-8 -*-
"""
Utility function and classes for solver associated code
"""

import attr

from numpy import (
    any as np_any,
    diff,
    abs as np_abs,
    sqrt as np_sqrt,
)

from scikits.odes.sundials.cvode import StatusEnum

from .. import __version__ as ds_version
from ..file_format import CONFIG_FIELDS
from ..logging import logging_options
from ..utils import (
    ODEIndex, DiscSolverError, MHD_Wave_Index, mhd_wave_speeds,
    MAGNETIC_INDEXES,
)


def error_handler(error_code, module, func, msg, user_data):
    """ drop all CVODE/IDA messages """
    # pylint: disable=unused-argument
    pass


def gen_sonic_point_rootfn(c_s):
    """
    Finds acoustic sonic point
    """
    def rootfn(θ, params, out):
        """
        root function to find acoustic sonic point
        """
        # pylint: disable=unused-argument
        out[0] = c_s - params[ODEIndex.v_θ]
        return 0
    return rootfn


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


def add_solver_arguments(parser):
    """
    Add common parser arguments for solver
    """
    parser.add_argument(
        '--version', action='version', version='%(prog)s ' + ds_version
    )
    parser.add_argument(
        "--sonic-method", choices=(
            "step", "single", "mcmc", "sonic_root", "hydrostatic", "mod_hydro",
        ), default="single",
    )
    parser.add_argument("--output-file")
    parser.add_argument(
        "--output-dir", default=".",
    )
    internal_store_group = parser.add_mutually_exclusive_group()
    internal_store_group.add_argument(
        "--store-internal", action='store_true', default=True
    )
    internal_store_group.add_argument(
        "--no-store-internal", action='store_false', dest="store_internal",
    )
    parser.add_argument("--override", action='append', nargs=2, default=[])
    parser.add_argument("--use-E_r", action='store_true', default=False)
    logging_options(parser)


def velocity_stop_generator(soln_input):
    """
    return function to stop at target velocity
    """
    def rootfn(θ, params, out):
        """
        root function to stop at required velocity
        """
        # pylint: disable=unused-argument
        out[0] = soln_input.target_velocity - params[ODEIndex.v_θ]
        return 0
    return rootfn, 1


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
