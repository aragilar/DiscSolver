# -*- coding: utf-8 -*-
"""
Utility function and classes for solver associated code
"""
from collections import defaultdict
from csv import DictReader, Sniffer, get_dialect

import attr
import logbook

from numpy import (
    any as np_any,
    diff,
    abs as np_abs,
    sqrt as np_sqrt,
    max as np_max,
)
from scipy.interpolate import interp1d

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
    parser, *, store_internal=True, sonic_method='single', output_file=None,
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
    parser.add_argument("--output-file", default=output_file)
    parser.add_argument("--output-dir", default=".", type=expanded_path)
    internal_store_group = parser.add_mutually_exclusive_group()
    internal_store_group.add_argument(
        "--store-internal", action='store_true', default=store_internal
    )
    internal_store_group.add_argument(
        "--no-store-internal", action='store_false', dest="store_internal",
    )
    parser.add_argument("--override", action='append', nargs=2, default=[])


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


def filter_csv_columns(seq, *, columns):
    """
    Only pass through specified columns
    """
    new_seq = []
    for d in seq:
        new_seq.append({k: v for k, v in d.items() if k in columns})
    return new_seq


class CSVReaderHelper:
    """
    Helper class to handle comments in csv files
    """
    def __init__(self, file, *, comment='#'):
        self.file = file
        self.comment = comment
        self._comments = []

    def __iter__(self):
        self._comments = []
        for line in self.file:
            filted_line = self._filter(line)
            if filted_line is not None:
                yield filted_line

    def _filter(self, line):
        """
        Filter function which collects comments, and returns data lines
        """
        if line.strip().startswith(self.comment):
            self._add_to_comments(line)
            return None
        return line

    def _add_to_comments(self, line):
        """
        Add line to comments stored
        """
        self._comments.append(line.strip()[len(self.comment):])

    def has_csv_header(self):
        """
        Checks if csv file has header
        """
        self.file.seek(0)
        has_header = Sniffer().has_header(next(iter(self)))
        self.file.seek(0)
        return has_header

    def get_dialect(self, *args, lines=5, **kwargs):
        """
        Wrap csv.Sniffer.sniff to handle comments
        """
        self.file.seek(0)
        dialect = Sniffer().sniff(
            [line for line, _ in zip(self, range(lines))],
            *args, **kwargs
        )
        self.file.seek(0)
        return dialect

    @property
    def comments(self):
        """
        Comments found whilst reading csv
        """
        return tuple(self._comments)


class CSVWriterHelper:
    """
    Helper class to add comments to csv files
    """
    def __init__(self, file, *, comment='#', dialect="unix"):
        self.file = file
        self.comment = comment
        self._dialect = get_dialect(dialect)

    @property
    def dialect(self):
        """
        CSV dialect to use when writing
        """
        return self._dialect

    def write(self, *args, **kwargs):
        """
        Wrapper for calling write on the wrapped file
        """
        return self.file.write(*args, **kwargs)

    def add_comment(self, comment, marker=None):
        """
        Add comment to file
        """
        comment_marker = marker or self.comment
        print(
            comment_marker + comment, file=self.file,
            end=self.dialect.lineterminator,
        )

    def add_metadata(self, mapping):
        """
        Add metadata to file
        """
        for key, val in mapping.items():
            self.add_comment(f"{key}={val}")


def get_csv_inputs(input_file, label=''):
    """
    Get inputs from csv file
    """
    with open(input_file) as infile:
        helper = CSVReaderHelper(infile)
        if helper.has_csv_header():
            inputs = filter_csv_columns(add_labels(
                DictReader(helper, dialect="unix"), label=label
            ), columns=CONFIG_FIELDS)
        else:
            inputs = filter_csv_columns(add_labels(DictReader(
                helper, fieldnames=SOLUTION_INPUT_FIELDS, dialect="unix",
            ), label=label), columns=CONFIG_FIELDS)

    return inputs


def deduplicate_and_interpolate(x, y, **kwargs):
    """
    As interp1d has issues with duplicates, fix x and y so that it works
    """
    dedup_dict = defaultdict(list)
    for x_i, y_i in zip(x, y):
        dedup_dict[x_i].append(y_i)
    fixed_x, fixed_y = zip(*[
        (x_i, np_max(y_i)) for x_i, y_i in dedup_dict.items()
    ])
    return interp1d(fixed_x, fixed_y, **kwargs)
