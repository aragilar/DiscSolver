"""
Defines the dumpers for the data strutures
"""
from numpy import asarray

from h5preserve import GroupContainer

from ._containers import (
    Solution, SolutionInput, ConfigInput, Problems, InternalData,
    DAEInternalData, DAEInitialConditions, Run, InitialConditions,
)
from ._utils import ds_registry


# pylint: disable=missing-docstring

@ds_registry.dumper(DAEInternalData, "DAEInternalData", version=1)
def _dae_internal_dump(internal_data):
    # pylint: disable=protected-access
    internal_data._finalise()
    # pylint: enable=protected-access
    return GroupContainer(
        derivs=internal_data.derivs,
        params=internal_data.params,
        angles=internal_data.angles,
        residuals=internal_data.residuals,
        problems=internal_data.problems,
    )


@ds_registry.dumper(InternalData, "InternalData", version=2)
def _internal_dump2(internal_data):
    # pylint: disable=protected-access
    internal_data._finalise()
    # pylint: enable=protected-access
    return GroupContainer(
        derivs=internal_data.derivs,
        params=internal_data.params,
        angles=internal_data.angles,
        v_r_normal=internal_data.v_r_normal,
        v_φ_normal=internal_data.v_φ_normal,
        ρ_normal=internal_data.ρ_normal,
        v_r_taylor=internal_data.v_r_taylor,
        v_φ_taylor=internal_data.v_φ_taylor,
        ρ_taylor=internal_data.ρ_taylor,
        problems=internal_data.problems,
    )


@ds_registry.dumper(DAEInitialConditions, "DAEInitialConditions", version=1)
def _dae_initial_dump(initial_conditions):
    return GroupContainer(
        attrs={
            "norm_kepler_sq": initial_conditions.norm_kepler_sq,
            "a_0": initial_conditions.a_0,
            "γ": initial_conditions.γ,
            "init_con": initial_conditions.init_con,
            "deriv_init_con": initial_conditions.deriv_init_con,
        }, angles=initial_conditions.angles,
    )


@ds_registry.dumper(InitialConditions, "InitialConditions", version=3)
def _initial_dump(initial_conditions):
    return GroupContainer(
        attrs={
            "norm_kepler_sq": initial_conditions.norm_kepler_sq,
            "a_0": initial_conditions.a_0,
            "η_O": initial_conditions.η_O,
            "η_A": initial_conditions.η_A,
            "η_H": initial_conditions.η_H,
            "γ": initial_conditions.γ,
            "init_con": initial_conditions.init_con,
        }, angles=initial_conditions.angles,
    )


@ds_registry.dumper(Solution, "Solution", version=2)
def _solution_dumper(solution):
    return GroupContainer(
        attrs={
            "flag": solution.flag,
            "coordinate_system": solution.coordinate_system,
        },
        angles=solution.angles,
        solution=solution.solution,
        internal_data=solution.internal_data,
        initial_conditions=solution.initial_conditions,
        t_roots=solution.t_roots,
        y_roots=solution.y_roots,
        solution_input=solution.solution_input,
        derivatives=solution.derivatives,
    )


@ds_registry.dumper(ConfigInput, "ConfigInput", version=4)
def _config_dumper(config_input):
    return GroupContainer(
        attrs={
            "start": config_input.start,
            "stop": config_input.stop,
            "taylor_stop_angle": config_input.taylor_stop_angle,
            "max_steps": config_input.max_steps,
            "num_angles": config_input.num_angles,
            "label": config_input.label,
            "relative_tolerance": config_input.relative_tolerance,
            "absolute_tolerance": config_input.absolute_tolerance,
            "γ": config_input.γ,
            "v_rin_on_c_s": config_input.v_rin_on_c_s,
            "v_a_on_c_s": config_input.v_a_on_c_s,
            "c_s_on_v_k": config_input.c_s_on_v_k,
            "η_O": config_input.η_O,
            "η_H": config_input.η_H,
            "η_A": config_input.η_A,
            "η_derivs": config_input.η_derivs,
        }, jump_before_sonic=config_input.jump_before_sonic
    )


@ds_registry.dumper(SolutionInput, "SolutionInput", version=4)
def _input_dumper(solution_input):
    return GroupContainer(
        attrs={
            "start": solution_input.start,
            "stop": solution_input.stop,
            "taylor_stop_angle": solution_input.taylor_stop_angle,
            "max_steps": solution_input.max_steps,
            "num_angles": solution_input.num_angles,
            "relative_tolerance": solution_input.relative_tolerance,
            "absolute_tolerance": solution_input.absolute_tolerance,
            "γ": solution_input.γ,
            "v_rin_on_c_s": solution_input.v_rin_on_c_s,
            "v_a_on_c_s": solution_input.v_a_on_c_s,
            "c_s_on_v_k": solution_input.c_s_on_v_k,
            "η_O": solution_input.η_O,
            "η_H": solution_input.η_H,
            "η_A": solution_input.η_A,
            "η_derivs": solution_input.η_derivs,
        }, jump_before_sonic=solution_input.jump_before_sonic
    )


@ds_registry.dumper(Run, "Run", version=1)
def _run_dumper(run):
    return GroupContainer(
        time=run.time,
        config_filename=run.config_filename,
        config_input=run.config_input,
        final_solution=run.final_solution,
        solutions=run.solutions,
    )


@ds_registry.dumper(Problems, "Problems", version=1)
def _problems_dumper(problems):
    group = GroupContainer()
    for key, item in problems.items():
        group[key] = asarray([
            s.encode("utf8") for s in item
        ])
    return group

# pylint: enable=missing-docstring
