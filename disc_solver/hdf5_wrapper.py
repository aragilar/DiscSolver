"""
Wrapper around h5py and hdf5 to manage different versions of output files from
Disc Solver
"""

from collections import namedtuple

import numpy as np
import h5py

from ._version import version as generator_version


def soln_open(f, **kwargs):
    """
    Open an hdf5 file containing Disc Solver data
    """
    file_version = OLDEST_CLASS(f, **kwargs).version
    if file_version is None:
        return NEWEST_CLASS(f, **kwargs)
    soln_class = SOLUTION_FILE_CLASSES[file_version]
    return soln_class(f, **kwargs)


def dataset_to_type(dset):
    """
    Convert h5py datasets to python/numpy types
    """
    if dset.shape == (1,):
        return dset.dtype.type(dset)
    return np.copy(dset)


class SolutionFileBase:
    """
    Base class for SolutionFile type hdf5 wrappers
    """
    _file = None
    _filename = None
    _require_close = True
    _version = "0"
    _new_file = False
    SolutionProperties = namedtuple("SolutionProperties", [])
    InitialConditions = namedtuple("InitialConditions", [])
    ConfigInput = namedtuple("ConfigInput", [])
    InternalData = namedtuple("InternalData", [])

    def __init__(self, f, **kwargs):
        if isinstance(f, h5py.File):
            self._file = f
            self._filename = f.filename
            self._require_close = False
            self._new_file = self._is_new_file(f)
        elif hasattr(f, "name"):
            self._filename = f.name
        else:
            self._filename = f
        self._kwargs = kwargs

    def __enter__(self):
        if self._file is None:
            self._file = h5py.File(self._filename, **self._kwargs)
            if self._new_file:
                self._file.attrs["version"] = self._version
                self._file.attrs["generator_version"] = generator_version
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._require_close:
            return self._file.__exit__(exc_type, exc_value, traceback)
        return False  # Don't handle any exceptions

    @staticmethod
    def _is_new_file(f):
        """
        Check if `f` is an empty/new hdf5 file
        """
        if not f.attrs.keys() and not f.keys():
            return True
        return False

    @property
    def version(self):
        """
        Solution file version
        """
        if self._file is None:
            with h5py.File(self._filename, **self._kwargs) as f:
                return f.attrs.get("version")
        return self._file.attrs.get("version")

    @property
    def name(self):
        """
        Name of file on filesystem
        """
        return self._filename

    @property
    def generator_version(self):
        """
        Version of Disc Solver used to create file
        """
        return self._file.attrs["generator_version"]


class SolutionFileV1(SolutionFileBase):
    """
    Version 1 of SolutionFile
    """
    _version = "1"

    SolutionProperties = namedtuple("SolutionProperties", [
        "flag", "coordinate_system",
    ])

    InitialConditions = namedtuple("InitialConditions", [
        "v_norm", "B_norm", "diff_norm", "ρ_norm", "norm_kepler_sq", "c_s",
        "η_O", "η_A", "η_H", "init_con", "angles",
    ])

    ConfigInput = namedtuple("ConfigInput", [
        "label", "start", "stop", "taylor_stop_angle", "radius",
        "scale_height_vs_radius", "central_mass", "β", "v_rin_on_v_k", "B_θ",
        "η_O", "η_H", "η_A", "c_s", "ρ", "max_steps", "num_angles",
    ])

    InternalData = namedtuple("InternalData", [
        "derivs",
        "params",
        "angles",
        "v_r_normal",
        "v_φ_normal",
        "ρ_normal",
        "v_r_taylor",
        "v_φ_taylor",
        "ρ_taylor",
    ])

    @property
    def angles(self):
        """
        Angles at which solution was found
        """
        return np.copy(self._file["angles"])

    @angles.setter
    def angles(self, vals):
        """
        Angles at which solution was found
        """
        self._file["angles"] = vals

    @property
    def solution(self):
        """
        The solution found
        """
        return np.copy(self._file["solution"])

    @solution.setter
    def solution(self, vals):
        """
        The solution found
        """
        self._file["solution"] = vals

    @property
    def initial_conditions(self):
        """
        Initial conditions used
        """
        return self.InitialConditions(**{
            key: dataset_to_type(val)
            for key, val in self._file["initial conditions"].items()
        })

    @initial_conditions.setter
    def initial_conditions(self, vals):
        """
        Initial conditions used
        """
        if self._file.get("initial conditions") is None:
            self._file.create_group("initial conditions")
        self._file["initial conditions"].update(vars(vals))

    @property
    def config_input(self):
        """
        The inputs for the solution
        """
        return self.ConfigInput(**self._file["config input"].attrs)

    @config_input.setter
    def config_input(self, vals):
        """
        The inputs for the solution
        """
        if self._file.get("config input") is None:
            self._file.create_group("config input")
        self._file["config input"].attrs.update(vars(vals))

    @property
    def internal_data(self):
        """
        Data internal to the solution
        """
        return self.InternalData({
            key: np.copy(val)
            for key, val in self._file["internal data"].items()
        })

    @internal_data.setter
    def internal_data(self, vals):
        """
        Data internal to the solution
        """
        if self._file.get("internal data") is None:
            self._file.create_group("internal data")
        self._file["internal data"].update(vars(vals))

    @property
    def config_filename(self):
        """
        Filename of the config file used
        """
        return self._file.attrs["config filename"]

    @config_filename.setter
    def config_filename(self, vals):
        """
        Filename of the config file used
        """
        self._file.attrs["config filename"] = str(vals)

    @property
    def config_label(self):
        """
        Name of the type of solution
        """
        return self._file.attrs["config label"]

    @config_label.setter
    def config_label(self, vals):
        """
        Name of the type of solution
        """
        self._file.attrs["config label"] = vals

    @property
    def time(self):
        """
        Time when solution finished
        """
        return self._file.attrs["time"]

    @time.setter
    def time(self, vals):
        """
        Time when solution finished
        """
        self._file.attrs["time"] = vals

    @property
    def solution_properties(self):
        """
        Properties of the solution
        """
        return self.SolutionProperties(**self._file["solution properties"])

    @solution_properties.setter
    def solution_properties(self, vals):
        """
        Properties of the solution
        """
        if self._file.get("solution properties") is None:
            self._file.create_group("solution properties")
        self._file["solution properties"].update(vars(vals))


SOLUTION_FILE_CLASSES = {
    klass._version: klass for klass in [  # pylint: disable=protected-access
        SolutionFileV1,
    ]
}

OLDEST_CLASS = SOLUTION_FILE_CLASSES[min(SOLUTION_FILE_CLASSES)]
NEWEST_CLASS = SOLUTION_FILE_CLASSES[max(SOLUTION_FILE_CLASSES)]
