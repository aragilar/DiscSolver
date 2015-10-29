"""
Definition of version 4 of the solution file format
"""

# pylint: skip-file

from numpy import array
from h5schemaesqe import HDF5Group


version_4_schema = HDF5Group(
    angles=array,
    solution=array,
    config_filename=str,
    config_label=str,
    time=str,

    solution_properties=HDF5Group(
        flag=int,
        coordinate_system=str,
    ),

    initial_conditions=HDF5Group(
        norm_kepler_sq=float,
        c_s=float,
        η_O=float,
        η_A=float,
        η_H=float,
        β=float,
        init_con=array,
        angles=array,
    ),

    config_input=HDF5Group(
        start=float,
        stop=float,
        taylor_stop_angle=float,
        max_steps=float,
        num_angles=float,
        label=str,
        relative_tolerance=float,
        absolute_tolerance=float,
        β=float,
        v_rin_on_c_s=float,
        v_a_on_c_s=float,
        c_s_on_v_k=float,
        η_O=float,
        η_H=float,
        η_A=float,
    ),

    internal_data=HDF5Group(
        derivs=array,
        params=array,
        angles=array,
        v_r_normal=array,
        v_φ_normal=array,
        ρ_normal=array,
        v_r_taylor=array,
        v_φ_taylor=array,
        ρ_taylor=array,
    ),
)
