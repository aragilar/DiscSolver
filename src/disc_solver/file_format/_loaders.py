"""
Defines the loaders for the data strutures
"""

from ._containers import (
    Solution, SolutionInput, ConfigInput, Problems, InternalData,
    Run, InitialConditions,
    JacobianData, Solutions
)
from ._utils import ds_registry, _str_β_to_γ


# pylint: disable=missing-docstring

@ds_registry.loader("InternalData", version=1)
def _internal_load(group):
    return InternalData(
        derivs=group["derivs"]["data"],
        params=group["params"]["data"],
        angles=group["angles"]["data"],
        v_r_normal=group["v_r_normal"]["data"],
        v_φ_normal=group["v_φ_normal"]["data"],
        ρ_normal=group["ρ_normal"]["data"],
        v_r_taylor=group["v_r_taylor"]["data"],
        v_φ_taylor=group["v_φ_taylor"]["data"],
        ρ_taylor=group["ρ_taylor"]["data"],
        jacobians=None,
    )


@ds_registry.loader("InternalData", version=2)
def _internal_load2(group):
    return InternalData(
        derivs=group["derivs"]["data"],
        params=group["params"]["data"],
        angles=group["angles"]["data"],
        v_r_normal=group["v_r_normal"]["data"],
        v_φ_normal=group["v_φ_normal"]["data"],
        ρ_normal=group["ρ_normal"]["data"],
        v_r_taylor=group["v_r_taylor"]["data"],
        v_φ_taylor=group["v_φ_taylor"]["data"],
        ρ_taylor=group["ρ_taylor"]["data"],
        problems=group["problems"],
        jacobians=None,
    )


@ds_registry.loader("InternalData", version=3)
def _internal_load3(group):
    return InternalData(
        derivs=group["derivs"]["data"],
        params=group["params"]["data"],
        angles=group["angles"]["data"],
        v_r_normal=group["v_r_normal"]["data"],
        v_φ_normal=group["v_φ_normal"]["data"],
        ρ_normal=group["ρ_normal"]["data"],
        v_r_taylor=group["v_r_taylor"]["data"],
        v_φ_taylor=group["v_φ_taylor"]["data"],
        ρ_taylor=group["ρ_taylor"]["data"],
        problems=group["problems"],
        jacobian_data=group["jacobian_data"],
    )


@ds_registry.loader("InitialConditions", version=3)
def _initial_load3(group):
    return InitialConditions(
        norm_kepler_sq=group.attrs["norm_kepler_sq"],
        a_0=group.attrs["a_0"],
        η_O=group.attrs["η_O"],
        η_A=group.attrs["η_A"],
        η_H=group.attrs["η_H"],
        γ=group.attrs["γ"],
        init_con=group.attrs["init_con"],
        angles=group["angles"]["data"],
    )


@ds_registry.loader("InitialConditions", version=2)
def _initial_load(group):
    return InitialConditions(
        norm_kepler_sq=group.attrs["norm_kepler_sq"],
        a_0=group.attrs["a_0"],
        η_O=group.attrs["η_O"],
        η_A=group.attrs["η_A"],
        η_H=group.attrs["η_H"],
        γ=5/4 - group.attrs["β"],
        init_con=group.attrs["init_con"],
        angles=group["angles"]["data"],
    )


@ds_registry.loader("InitialConditions", version=1)
def _initial_load_old(group):
    return InitialConditions(
        norm_kepler_sq=group.attrs["norm_kepler_sq"],
        a_0=float('nan'),
        η_O=group.attrs["η_O"],
        η_A=group.attrs["η_A"],
        η_H=group.attrs["η_H"],
        γ=5/4 - group.attrs["β"],
        init_con=group.attrs["init_con"],
        angles=group["angles"]["data"],
    )


@ds_registry.loader("Solution", version=1)
def _solution_loader(group):
    if group["t_roots"] is None:
        t_roots = None
    else:
        t_roots = group["t_roots"]["data"]
    if group["y_roots"] is None:
        y_roots = None
    else:
        y_roots = group["y_roots"]["data"]

    return Solution(
        flag=group.attrs["flag"],
        coordinate_system=group.attrs["coordinate_system"],
        angles=group["angles"]["data"],
        solution=group["solution"]["data"],
        internal_data=group["internal_data"],
        initial_conditions=group["initial_conditions"],
        solution_input=group["solution_input"],
        t_roots=t_roots,
        y_roots=y_roots,
        derivatives=None,
        sonic_point=None,
        sonic_point_values=None,
    )


@ds_registry.loader("Solution", version=2)
def _solution_loader_2(group):
    if group["t_roots"] is None:
        t_roots = None
    else:
        t_roots = group["t_roots"]["data"]
    if group["y_roots"] is None:
        y_roots = None
    else:
        y_roots = group["y_roots"]["data"]

    return Solution(
        flag=group.attrs["flag"],
        coordinate_system=group.attrs["coordinate_system"],
        angles=group["angles"]["data"],
        solution=group["solution"]["data"],
        internal_data=group["internal_data"],
        initial_conditions=group["initial_conditions"],
        solution_input=group["solution_input"],
        t_roots=t_roots,
        y_roots=y_roots,
        derivatives=group["derivatives"],
        sonic_point=None,
        sonic_point_values=None,
    )


@ds_registry.loader("Solution", version=3)
def _solution_loader_3(group):
    if group["t_roots"] is None:
        t_roots = None
    else:
        t_roots = group["t_roots"]["data"]
    if group["y_roots"] is None:
        y_roots = None
    else:
        y_roots = group["y_roots"]["data"]

    if group["sonic_point"] is None:
        sonic_point = None
    else:
        sonic_point = group["sonic_point"]["data"]
    if group["sonic_point_values"] is None:
        sonic_point_values = None
    else:
        sonic_point_values = group["sonic_point_values"]["data"]

    return Solution(
        flag=group.attrs["flag"],
        coordinate_system=group.attrs["coordinate_system"],
        angles=group["angles"]["data"],
        solution=group["solution"]["data"],
        internal_data=group["internal_data"],
        initial_conditions=group["initial_conditions"],
        solution_input=group["solution_input"],
        t_roots=t_roots,
        y_roots=y_roots,
        sonic_point=sonic_point,
        sonic_point_values=sonic_point_values,
        derivatives=group["derivatives"],
    )


@ds_registry.loader("ConfigInput", version=1)
def _config_loader(group):
    return ConfigInput(
        start=group.attrs["start"],
        stop=group.attrs["stop"],
        taylor_stop_angle=group.attrs["taylor_stop_angle"],
        max_steps=group.attrs["max_steps"],
        num_angles=group.attrs["num_angles"],
        label=group.attrs["label"],
        relative_tolerance=group.attrs["relative_tolerance"],
        absolute_tolerance=group.attrs["absolute_tolerance"],
        split_method="v_θ_deriv",
        γ=_str_β_to_γ(group.attrs["β"]),
        v_rin_on_c_s=group.attrs["v_rin_on_c_s"],
        v_a_on_c_s=group.attrs["v_a_on_c_s"],
        c_s_on_v_k=group.attrs["c_s_on_v_k"],
        η_O=group.attrs["η_O"],
        η_H=group.attrs["η_H"],
        η_A=group.attrs["η_A"],
    )


@ds_registry.loader("ConfigInput", version=2)
def _config_loader2(group):
    return ConfigInput(
        start=group.attrs["start"],
        stop=group.attrs["stop"],
        taylor_stop_angle=group.attrs["taylor_stop_angle"],
        max_steps=group.attrs["max_steps"],
        num_angles=group.attrs["num_angles"],
        label=group.attrs["label"],
        relative_tolerance=group.attrs["relative_tolerance"],
        absolute_tolerance=group.attrs["absolute_tolerance"],
        split_method="v_θ_deriv",
        γ=_str_β_to_γ(group.attrs["β"]),
        v_rin_on_c_s=group.attrs["v_rin_on_c_s"],
        v_a_on_c_s=group.attrs["v_a_on_c_s"],
        c_s_on_v_k=group.attrs["c_s_on_v_k"],
        η_O=group.attrs["η_O"],
        η_H=group.attrs["η_H"],
        η_A=group.attrs["η_A"],
        jump_before_sonic=group["jump_before_sonic"]
    )


@ds_registry.loader("ConfigInput", version=3)
def _config_loader3(group):
    return ConfigInput(
        start=group.attrs["start"],
        stop=group.attrs["stop"],
        taylor_stop_angle=group.attrs["taylor_stop_angle"],
        max_steps=group.attrs["max_steps"],
        num_angles=group.attrs["num_angles"],
        label=group.attrs["label"],
        relative_tolerance=group.attrs["relative_tolerance"],
        absolute_tolerance=group.attrs["absolute_tolerance"],
        split_method="v_θ_deriv",
        γ=_str_β_to_γ(group.attrs["β"]),
        v_rin_on_c_s=group.attrs["v_rin_on_c_s"],
        v_a_on_c_s=group.attrs["v_a_on_c_s"],
        c_s_on_v_k=group.attrs["c_s_on_v_k"],
        η_O=group.attrs["η_O"],
        η_H=group.attrs["η_H"],
        η_A=group.attrs["η_A"],
        jump_before_sonic=group["jump_before_sonic"],
        η_derivs=group.attrs["η_derivs"],
    )


@ds_registry.loader("ConfigInput", version=4)
def _config_loader4(group):
    return ConfigInput(
        start=group.attrs["start"],
        stop=group.attrs["stop"],
        taylor_stop_angle=group.attrs["taylor_stop_angle"],
        max_steps=group.attrs["max_steps"],
        num_angles=group.attrs["num_angles"],
        label=group.attrs["label"],
        relative_tolerance=group.attrs["relative_tolerance"],
        absolute_tolerance=group.attrs["absolute_tolerance"],
        γ=group.attrs["γ"],
        v_rin_on_c_s=group.attrs["v_rin_on_c_s"],
        v_a_on_c_s=group.attrs["v_a_on_c_s"],
        c_s_on_v_k=group.attrs["c_s_on_v_k"],
        η_O=group.attrs["η_O"],
        η_H=group.attrs["η_H"],
        η_A=group.attrs["η_A"],
        jump_before_sonic=group["jump_before_sonic"],
        η_derivs=group.attrs["η_derivs"],
        nwalkers=None,
        iterations=None,
        threads=None,
        target_velocity=None,
        split_method=None,
    )


@ds_registry.loader("ConfigInput", version=5)
def _config_loader5(group):
    return ConfigInput(
        start=group.attrs["start"],
        stop=group.attrs["stop"],
        taylor_stop_angle=group.attrs["taylor_stop_angle"],
        max_steps=group.attrs["max_steps"],
        num_angles=group.attrs["num_angles"],
        label=group.attrs["label"],
        relative_tolerance=group.attrs["relative_tolerance"],
        absolute_tolerance=group.attrs["absolute_tolerance"],
        γ=group.attrs["γ"],
        nwalkers=group.attrs["nwalkers"],
        iterations=group.attrs["iterations"],
        threads=group.attrs["threads"],
        split_method="v_θ_deriv",
        v_rin_on_c_s=group.attrs["v_rin_on_c_s"],
        v_a_on_c_s=group.attrs["v_a_on_c_s"],
        c_s_on_v_k=group.attrs["c_s_on_v_k"],
        η_O=group.attrs["η_O"],
        η_H=group.attrs["η_H"],
        η_A=group.attrs["η_A"],
        jump_before_sonic=group["jump_before_sonic"],
        η_derivs=group.attrs["η_derivs"],
    )


@ds_registry.loader("ConfigInput", version=6)
def _config_loader6(group):
    return ConfigInput(
        start=group.attrs["start"],
        stop=group.attrs["stop"],
        taylor_stop_angle=group.attrs["taylor_stop_angle"],
        max_steps=group.attrs["max_steps"],
        num_angles=group.attrs["num_angles"],
        label=group.attrs["label"],
        relative_tolerance=group.attrs["relative_tolerance"],
        absolute_tolerance=group.attrs["absolute_tolerance"],
        γ=group.attrs["γ"],
        nwalkers=group.attrs["nwalkers"],
        iterations=group.attrs["iterations"],
        threads=group.attrs["threads"],
        target_velocity=group.attrs["target_velocity"],
        split_method="v_θ_deriv",
        v_rin_on_c_s=group.attrs["v_rin_on_c_s"],
        v_a_on_c_s=group.attrs["v_a_on_c_s"],
        c_s_on_v_k=group.attrs["c_s_on_v_k"],
        η_O=group.attrs["η_O"],
        η_H=group.attrs["η_H"],
        η_A=group.attrs["η_A"],
        jump_before_sonic=group["jump_before_sonic"],
        η_derivs=group.attrs["η_derivs"],
    )


@ds_registry.loader("ConfigInput", version=7)
def _config_loader7(group):
    return ConfigInput(
        start=group.attrs["start"],
        stop=group.attrs["stop"],
        taylor_stop_angle=group.attrs["taylor_stop_angle"],
        max_steps=group.attrs["max_steps"],
        num_angles=group.attrs["num_angles"],
        label=group.attrs["label"],
        relative_tolerance=group.attrs["relative_tolerance"],
        absolute_tolerance=group.attrs["absolute_tolerance"],
        γ=group.attrs["γ"],
        nwalkers=group.attrs["nwalkers"],
        iterations=group.attrs["iterations"],
        threads=group.attrs["threads"],
        target_velocity=group.attrs["target_velocity"],
        split_method=group.attrs["split_method"],
        v_rin_on_c_s=group.attrs["v_rin_on_c_s"],
        v_a_on_c_s=group.attrs["v_a_on_c_s"],
        c_s_on_v_k=group.attrs["c_s_on_v_k"],
        η_O=group.attrs["η_O"],
        η_H=group.attrs["η_H"],
        η_A=group.attrs["η_A"],
        jump_before_sonic=group["jump_before_sonic"],
        η_derivs=group.attrs["η_derivs"],
    )


@ds_registry.loader("ConfigInput", version=8)
def _config_loader8(group):
    return ConfigInput(
        start=group.attrs["start"],
        stop=group.attrs["stop"],
        taylor_stop_angle=group.attrs["taylor_stop_angle"],
        max_steps=group.attrs["max_steps"],
        num_angles=group.attrs["num_angles"],
        label=group.attrs["label"],
        relative_tolerance=group.attrs["relative_tolerance"],
        absolute_tolerance=group.attrs["absolute_tolerance"],
        γ=group.attrs["γ"],
        nwalkers=group.attrs["nwalkers"],
        iterations=group.attrs["iterations"],
        threads=group.attrs["threads"],
        target_velocity=group.attrs["target_velocity"],
        split_method=group.attrs["split_method"],
        v_rin_on_c_s=group.attrs["v_rin_on_c_s"],
        v_a_on_c_s=group.attrs["v_a_on_c_s"],
        c_s_on_v_k=group.attrs["c_s_on_v_k"],
        η_O=group.attrs["η_O"],
        η_H=group.attrs["η_H"],
        η_A=group.attrs["η_A"],
        jump_before_sonic=group["jump_before_sonic"],
        η_derivs=group.attrs["η_derivs"],
        use_taylor_jump=group.attrs["use_taylor_jump"],
    )


@ds_registry.loader("SolutionInput", version=1)
def _input_loader(group):
    return SolutionInput(
        start=group.attrs["start"],
        stop=group.attrs["stop"],
        taylor_stop_angle=group.attrs["taylor_stop_angle"],
        max_steps=group.attrs["max_steps"],
        num_angles=group.attrs["num_angles"],
        relative_tolerance=group.attrs["relative_tolerance"],
        absolute_tolerance=group.attrs["absolute_tolerance"],
        split_method="v_θ_deriv",
        γ=5/4 - group.attrs["β"],
        v_rin_on_c_s=group.attrs["v_rin_on_c_s"],
        v_a_on_c_s=group.attrs["v_a_on_c_s"],
        c_s_on_v_k=group.attrs["c_s_on_v_k"],
        η_O=group.attrs["η_O"],
        η_H=group.attrs["η_H"],
        η_A=group.attrs["η_A"],
    )


@ds_registry.loader("SolutionInput", version=2)
def _input_loader2(group):
    if group["jump_before_sonic"] is None:
        jump_before_sonic = None
    else:
        jump_before_sonic = group["jump_before_sonic"]
    return SolutionInput(
        start=group.attrs["start"],
        stop=group.attrs["stop"],
        taylor_stop_angle=group.attrs["taylor_stop_angle"],
        max_steps=group.attrs["max_steps"],
        num_angles=group.attrs["num_angles"],
        relative_tolerance=group.attrs["relative_tolerance"],
        absolute_tolerance=group.attrs["absolute_tolerance"],
        split_method="v_θ_deriv",
        γ=5/4 - group.attrs["β"],
        v_rin_on_c_s=group.attrs["v_rin_on_c_s"],
        v_a_on_c_s=group.attrs["v_a_on_c_s"],
        c_s_on_v_k=group.attrs["c_s_on_v_k"],
        η_O=group.attrs["η_O"],
        η_H=group.attrs["η_H"],
        η_A=group.attrs["η_A"],
        jump_before_sonic=jump_before_sonic
    )


@ds_registry.loader("SolutionInput", version=3)
def _input_loader3(group):
    if group["jump_before_sonic"] is None:
        jump_before_sonic = None
    else:
        jump_before_sonic = group["jump_before_sonic"]
    return SolutionInput(
        start=group.attrs["start"],
        stop=group.attrs["stop"],
        taylor_stop_angle=group.attrs["taylor_stop_angle"],
        max_steps=group.attrs["max_steps"],
        num_angles=group.attrs["num_angles"],
        relative_tolerance=group.attrs["relative_tolerance"],
        absolute_tolerance=group.attrs["absolute_tolerance"],
        split_method="v_θ_deriv",
        γ=5/4 - group.attrs["β"],
        v_rin_on_c_s=group.attrs["v_rin_on_c_s"],
        v_a_on_c_s=group.attrs["v_a_on_c_s"],
        c_s_on_v_k=group.attrs["c_s_on_v_k"],
        η_O=group.attrs["η_O"],
        η_H=group.attrs["η_H"],
        η_A=group.attrs["η_A"],
        jump_before_sonic=jump_before_sonic,
        η_derivs=group.attrs["η_derivs"],
    )


@ds_registry.loader("SolutionInput", version=4)
def _input_loader4(group):
    if group["jump_before_sonic"] is None:
        jump_before_sonic = None
    else:
        jump_before_sonic = group["jump_before_sonic"]
    return SolutionInput(
        start=group.attrs["start"],
        stop=group.attrs["stop"],
        taylor_stop_angle=group.attrs["taylor_stop_angle"],
        max_steps=group.attrs["max_steps"],
        num_angles=group.attrs["num_angles"],
        relative_tolerance=group.attrs["relative_tolerance"],
        absolute_tolerance=group.attrs["absolute_tolerance"],
        split_method="v_θ_deriv",
        γ=group.attrs["γ"],
        v_rin_on_c_s=group.attrs["v_rin_on_c_s"],
        v_a_on_c_s=group.attrs["v_a_on_c_s"],
        c_s_on_v_k=group.attrs["c_s_on_v_k"],
        η_O=group.attrs["η_O"],
        η_H=group.attrs["η_H"],
        η_A=group.attrs["η_A"],
        jump_before_sonic=jump_before_sonic,
        η_derivs=group.attrs["η_derivs"],
        nwalkers=None,
        iterations=None,
        threads=None,
        target_velocity=None,
    )


@ds_registry.loader("SolutionInput", version=5)
def _input_loader5(group):
    if group["jump_before_sonic"] is None:
        jump_before_sonic = None
    else:
        jump_before_sonic = group["jump_before_sonic"]
    return SolutionInput(
        start=group.attrs["start"],
        stop=group.attrs["stop"],
        taylor_stop_angle=group.attrs["taylor_stop_angle"],
        max_steps=group.attrs["max_steps"],
        num_angles=group.attrs["num_angles"],
        relative_tolerance=group.attrs["relative_tolerance"],
        absolute_tolerance=group.attrs["absolute_tolerance"],
        split_method="v_θ_deriv",
        γ=group.attrs["γ"],
        nwalkers=group.attrs["nwalkers"],
        iterations=group.attrs["iterations"],
        threads=group.attrs["threads"],
        v_rin_on_c_s=group.attrs["v_rin_on_c_s"],
        v_a_on_c_s=group.attrs["v_a_on_c_s"],
        c_s_on_v_k=group.attrs["c_s_on_v_k"],
        η_O=group.attrs["η_O"],
        η_H=group.attrs["η_H"],
        η_A=group.attrs["η_A"],
        jump_before_sonic=jump_before_sonic,
        η_derivs=group.attrs["η_derivs"],
    )


@ds_registry.loader("SolutionInput", version=6)
def _input_loader6(group):
    if group["jump_before_sonic"] is None:
        jump_before_sonic = None
    else:
        jump_before_sonic = group["jump_before_sonic"]
    return SolutionInput(
        start=group.attrs["start"],
        stop=group.attrs["stop"],
        taylor_stop_angle=group.attrs["taylor_stop_angle"],
        max_steps=group.attrs["max_steps"],
        num_angles=group.attrs["num_angles"],
        relative_tolerance=group.attrs["relative_tolerance"],
        absolute_tolerance=group.attrs["absolute_tolerance"],
        γ=group.attrs["γ"],
        nwalkers=group.attrs["nwalkers"],
        iterations=group.attrs["iterations"],
        threads=group.attrs["threads"],
        target_velocity=group.attrs["target_velocity"],
        split_method="v_θ_deriv",
        v_rin_on_c_s=group.attrs["v_rin_on_c_s"],
        v_a_on_c_s=group.attrs["v_a_on_c_s"],
        c_s_on_v_k=group.attrs["c_s_on_v_k"],
        η_O=group.attrs["η_O"],
        η_H=group.attrs["η_H"],
        η_A=group.attrs["η_A"],
        jump_before_sonic=jump_before_sonic,
        η_derivs=group.attrs["η_derivs"],
    )


@ds_registry.loader("SolutionInput", version=7)
def _input_loader7(group):
    if group["jump_before_sonic"] is None:
        jump_before_sonic = None
    else:
        jump_before_sonic = group["jump_before_sonic"]
    return SolutionInput(
        start=group.attrs["start"],
        stop=group.attrs["stop"],
        taylor_stop_angle=group.attrs["taylor_stop_angle"],
        max_steps=group.attrs["max_steps"],
        num_angles=group.attrs["num_angles"],
        relative_tolerance=group.attrs["relative_tolerance"],
        absolute_tolerance=group.attrs["absolute_tolerance"],
        γ=group.attrs["γ"],
        nwalkers=group.attrs["nwalkers"],
        iterations=group.attrs["iterations"],
        threads=group.attrs["threads"],
        target_velocity=group.attrs["target_velocity"],
        split_method=group.attrs["split_method"],
        v_rin_on_c_s=group.attrs["v_rin_on_c_s"],
        v_a_on_c_s=group.attrs["v_a_on_c_s"],
        c_s_on_v_k=group.attrs["c_s_on_v_k"],
        η_O=group.attrs["η_O"],
        η_H=group.attrs["η_H"],
        η_A=group.attrs["η_A"],
        jump_before_sonic=jump_before_sonic,
        η_derivs=group.attrs["η_derivs"],
    )


@ds_registry.loader("SolutionInput", version=8)
def _input_loader8(group):
    if group["jump_before_sonic"] is None:
        jump_before_sonic = None
    else:
        jump_before_sonic = group["jump_before_sonic"]
    return SolutionInput(
        start=group.attrs["start"],
        stop=group.attrs["stop"],
        taylor_stop_angle=group.attrs["taylor_stop_angle"],
        max_steps=group.attrs["max_steps"],
        num_angles=group.attrs["num_angles"],
        relative_tolerance=group.attrs["relative_tolerance"],
        absolute_tolerance=group.attrs["absolute_tolerance"],
        γ=group.attrs["γ"],
        nwalkers=group.attrs["nwalkers"],
        iterations=group.attrs["iterations"],
        threads=group.attrs["threads"],
        target_velocity=group.attrs["target_velocity"],
        split_method=group.attrs["split_method"],
        v_rin_on_c_s=group.attrs["v_rin_on_c_s"],
        v_a_on_c_s=group.attrs["v_a_on_c_s"],
        c_s_on_v_k=group.attrs["c_s_on_v_k"],
        η_O=group.attrs["η_O"],
        η_H=group.attrs["η_H"],
        η_A=group.attrs["η_A"],
        jump_before_sonic=jump_before_sonic,
        η_derivs=group.attrs["η_derivs"],
        use_taylor_jump=group.attrs["use_taylor_jump"],
    )


@ds_registry.loader("Run", version=1)
def _run_loader(group):
    return Run(
        config_input=group["config_input"],
        config_filename=group["config_filename"],
        time=group["time"],
        final_solution=group["final_solution"],
        solutions=group["solutions"],
        disc_solver_version=None,
        float_type=None,
        sonic_method="unknown",
    )


@ds_registry.loader("Run", version=2)
def _run_loader2(group):
    return Run(
        config_input=group["config_input"],
        config_filename=group["config_filename"],
        time=group["time"],
        final_solution=group["final_solution"],
        solutions=group["solutions"],
        disc_solver_version=group["disc_solver_version"],
        float_type=None,
        sonic_method="unknown",
    )


@ds_registry.loader("Run", version=3)
def _run_loader3(group):
    return Run(
        config_input=group["config_input"],
        config_filename=group["config_filename"],
        time=group["time"],
        final_solution=group["final_solution"],
        solutions=group["solutions"],
        disc_solver_version=group["disc_solver_version"],
        float_type=group["float_type"],
        sonic_method="unknown",
    )


@ds_registry.loader("Run", version=4)
def _run_loader4(group):
    return Run(
        config_input=group["config_input"],
        config_filename=group["config_filename"],
        time=group["time"],
        final_solution=group["final_solution"],
        solutions=group["solutions"],
        disc_solver_version=group["disc_solver_version"],
        float_type=group["float_type"],
        sonic_method=group["sonic_method"],
    )


@ds_registry.loader("Run", version=5)
def _run_loader5(group):
    return Run(
        config_input=group["config_input"],
        config_filename=group["config_filename"],
        time=group["time"],
        final_solution=group["final_solution"],
        solutions=group["solutions"],
        disc_solver_version=group["disc_solver_version"],
        float_type=group["float_type"],
        sonic_method=group["sonic_method"],
        use_E_r=group["use_E_r"],
    )


@ds_registry.loader("Problems", version=1)
def _problems_loader(group):
    problems = Problems()
    for key, item in group.items():
        problems[key] = [
            s.decode("utf8") for s in item["data"]
        ]
    return problems


@ds_registry.loader("JacobianData", version=1)
def _jacobian_data_loader(group):
    return JacobianData(
        derivs=group["derivs"]["data"],
        params=group["params"]["data"],
        angles=group["angles"]["data"],
        jacobians=group["jacobians"]["data"],
    )


@ds_registry.loader("Solutions", version=1)
def _solutions_loader(group):
    return Solutions(**group)

# pylint: enable=missing-docstring
