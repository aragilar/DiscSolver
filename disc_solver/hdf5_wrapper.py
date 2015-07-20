"""
Wrapper around h5py and hdf5 to manage different versions of output files from
Disc Solver
"""

import numpy as np

from hdf5_wrapper import HDF5Wrapper

from ._version import version  # pylint: disable=unused-import


class SolutionFileBase(
    HDF5Wrapper, filetype="Solution File", generator_version=version
):
    """
    Base class for SolutionFile type hdf5 wrappers
    """
    pass


class SolutionFileV1(SolutionFileBase, version=1):
    """
    Version 1 of SolutionFile
    """
    angles = np.ndarray
    solution = np.ndarray
    config_filename = str
    config_label = str
    time = str

    solution_properties = {
        "flag": int,
        "coordinate_system": str,
    }

    initial_conditions = {
        "v_norm": float,
        "B_norm": float,
        "diff_norm": float,
        "ρ_norm": float,
        "norm_kepler_sq": float,
        "c_s": float,
        "η_O": float,
        "η_A": float,
        "η_H": float,
        "init_con": np.ndarray,
        "angles": np.ndarray,
    }

    config_input = {
        "label": str,
        "start": float,
        "stop": float,
        "taylor_stop_angle": float,
        "radius": float,
        "scale_height_vs_radius": float,
        "central_mass": float,
        "β": float,
        "v_rin_on_v_k": float,
        "B_θ": float,
        "η_O": float,
        "η_H": float,
        "η_A": float,
        "c_s": float,
        "ρ": float,
        "max_steps": float,
        "num_angles": float,
    }

    internal_data = {
        "derivs": np.ndarray,
        "params": np.ndarray,
        "angles": np.ndarray,
        "v_r_normal": np.ndarray,
        "v_φ_normal": np.ndarray,
        "ρ_normal": np.ndarray,
        "v_r_taylor": np.ndarray,
        "v_φ_taylor": np.ndarray,
        "ρ_taylor": np.ndarray,
    }


class SolutionFileV2(SolutionFileBase, version=2):
    """
    Version 2 of SolutionFile
    """
    angles = np.ndarray
    solution = np.ndarray
    config_filename = str
    config_label = str
    time = str

    solution_properties = {
        "flag": int,
        "coordinate_system": str,
    }

    initial_conditions = {
        "v_norm": float,
        "B_norm": float,
        "diff_norm": float,
        "ρ_norm": float,
        "norm_kepler_sq": float,
        "c_s": float,
        "η_O": float,
        "η_A": float,
        "η_H": float,
        "init_con": np.ndarray,
        "angles": np.ndarray,
    }

    config_input = {
        "label": str,
        "start": float,
        "stop": float,
        "taylor_stop_angle": float,
        "radius": float,
        "central_mass": float,
        "β": float,
        "B_θ": float,
        "ρ": float,
        "v_rin_on_c_s": float,
        "η_O": float,
        "η_H": float,
        "η_A": float,
        "max_steps": float,
        "num_angles": float,
    }

    internal_data = {
        "derivs": np.ndarray,
        "params": np.ndarray,
        "angles": np.ndarray,
        "v_r_normal": np.ndarray,
        "v_φ_normal": np.ndarray,
        "ρ_normal": np.ndarray,
        "v_r_taylor": np.ndarray,
        "v_φ_taylor": np.ndarray,
        "ρ_taylor": np.ndarray,
    }


soln_open = SolutionFileBase.open
NEWEST_CLASS = SolutionFileBase.newest()
OLDEST_CLASS = SolutionFileBase.oldest()
