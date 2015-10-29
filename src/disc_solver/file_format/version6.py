"""
Definition of version 6 of the solution file format
"""

# pylint: skip-file

from numpy import array
from scikits.odes.sundials.cvode import StatusEnum
from h5schemaesqe import HDF5Link, HDF5Group, HDF5MultiGroup


version_6_schema = HDF5Group(
    config_input=HDF5Group(
        start=str,
        stop=str,
        taylor_stop_angle=str,
        max_steps=str,
        num_angles=str,
        label=str,
        relative_tolerance=str,
        absolute_tolerance=str,
        β=str,
        v_rin_on_c_s=str,
        v_a_on_c_s=str,
        c_s_on_v_k=str,
        η_O=str,
        η_H=str,
        η_A=str,
    ),
    config_filename=str,
    time=str,
    final_solution=HDF5Link(),
    solutions=HDF5MultiGroup(
        "solution", HDF5Group(
            soln_input=HDF5Group(
                start=float,
                stop=float,
                taylor_stop_angle=float,
                max_steps=int,
                num_angles=int,
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
            angles=array,
            solution=array,
            flag=lambda x: StatusEnum(int(array(x))),
            coordinate_system=str,
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
        ),
    ),
)
