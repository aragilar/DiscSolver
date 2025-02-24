"""
Defines the loaders for the data strutures
"""

from ._containers import (
    Solution, SolutionInput, ConfigInput, Problems, InternalData, Run,
    InitialConditions, JacobianData, Solutions, MCMCVars,
)
from ._utils import ds_registry, _str_β_to_γ, ensure_fully_loaded


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


@ds_registry.loader("Solution", version=4)
def _solution_loader_4(group):
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
        is_post_shock_only=group.attrs["is_post_shock_only"],
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
        use_E_r=None,
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
        jump_before_sonic=group["jump_before_sonic"],
        use_E_r=None,
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
        use_E_r=None,
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
        use_E_r=None,
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
        use_E_r=None,
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
        use_E_r=None,
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
        use_E_r=None,
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
        use_E_r=None,
    )


@ds_registry.loader("ConfigInput", version=9)
def _config_loader9(group):
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
        mcmc_vars=group["mcmc_vars"],
        use_E_r=None,
    )


@ds_registry.loader("ConfigInput", version=10)
def _config_loader10(group):
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
        mcmc_vars=group["mcmc_vars"],
        v_θ_sonic_crit=group["v_θ_sonic_crit"],
        use_E_r=None,
    )


@ds_registry.loader("ConfigInput", version=11)
def _config_loader11(group):
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
        mcmc_vars=group["mcmc_vars"],
        v_θ_sonic_crit=group["v_θ_sonic_crit"],
        after_sonic=group["after_sonic"],
        use_E_r=None,
    )


@ds_registry.loader("ConfigInput", version=12)
def _config_loader12(group):
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
        mcmc_vars=group["mcmc_vars"],
        v_θ_sonic_crit=group["v_θ_sonic_crit"],
        after_sonic=group["after_sonic"],
        sonic_interp_size=group["sonic_interp_size"],
        interp_range=group.attrs["interp_range"],
        use_E_r=None,
    )


@ds_registry.loader("ConfigInput", version=13)
def _config_loader13(group):
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
        mcmc_vars=group["mcmc_vars"],
        v_θ_sonic_crit=group["v_θ_sonic_crit"],
        after_sonic=group["after_sonic"],
        sonic_interp_size=group["sonic_interp_size"],
        interp_range=ensure_fully_loaded(group["interp_range"]),
        interp_slice=group["interp_slice"],
        use_E_r=None,
    )


@ds_registry.loader("ConfigInput", version=14)
def _config_loader14(group):
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
        mcmc_vars=group["mcmc_vars"],
        v_θ_sonic_crit=group["v_θ_sonic_crit"],
        after_sonic=group["after_sonic"],
        sonic_interp_size=group["sonic_interp_size"],
        interp_range=ensure_fully_loaded(group["interp_range"]),
        interp_slice=group["interp_slice"],
        use_E_r=group.attrs["use_E_r"],
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
        use_E_r=None,
    )


@ds_registry.loader("SolutionInput", version=2)
def _input_loader2(group):
    if group["jump_before_sonic"] is None:
        jump_before_sonic = None
    else:
        jump_before_sonic = group["jump_before_sonic"]["data"]
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
        use_E_r=None,
    )


@ds_registry.loader("SolutionInput", version=3)
def _input_loader3(group):
    if group["jump_before_sonic"] is None:
        jump_before_sonic = None
    else:
        jump_before_sonic = group["jump_before_sonic"]["data"]
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
        use_E_r=None,
    )


@ds_registry.loader("SolutionInput", version=4)
def _input_loader4(group):
    if group["jump_before_sonic"] is None:
        jump_before_sonic = None
    else:
        jump_before_sonic = group["jump_before_sonic"]["data"]
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
        use_E_r=None,
    )


@ds_registry.loader("SolutionInput", version=5)
def _input_loader5(group):
    if group["jump_before_sonic"] is None:
        jump_before_sonic = None
    else:
        jump_before_sonic = group["jump_before_sonic"]["data"]
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
        use_E_r=None,
    )


@ds_registry.loader("SolutionInput", version=6)
def _input_loader6(group):
    if group["jump_before_sonic"] is None:
        jump_before_sonic = None
    else:
        jump_before_sonic = group["jump_before_sonic"]["data"]
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
        use_E_r=None,
    )


@ds_registry.loader("SolutionInput", version=7)
def _input_loader7(group):
    if group["jump_before_sonic"] is None:
        jump_before_sonic = None
    else:
        jump_before_sonic = group["jump_before_sonic"]["data"]
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
        use_E_r=None,
    )


@ds_registry.loader("SolutionInput", version=8)
def _input_loader8(group):
    if group["jump_before_sonic"] is None:
        jump_before_sonic = None
    else:
        jump_before_sonic = group["jump_before_sonic"]["data"]
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
        use_E_r=None,
    )


@ds_registry.loader("SolutionInput", version=9)
def _input_loader9(group):
    if group["jump_before_sonic"] is None:
        jump_before_sonic = None
    else:
        jump_before_sonic = group["jump_before_sonic"]["data"]
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
        mcmc_vars=group["mcmc_vars"],
        use_E_r=None,
    )


@ds_registry.loader("SolutionInput", version=10)
def _input_loader10(group):
    if group["jump_before_sonic"] is None:
        jump_before_sonic = None
    else:
        jump_before_sonic = group["jump_before_sonic"]["data"]
    if group["v_θ_sonic_crit"] is None:
        v_θ_sonic_crit = None
    else:
        v_θ_sonic_crit = group["v_θ_sonic_crit"]["data"]
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
        mcmc_vars=group["mcmc_vars"],
        v_θ_sonic_crit=v_θ_sonic_crit,
        use_E_r=None,
    )


@ds_registry.loader("SolutionInput", version=11)
def _input_loader11(group):
    if group["jump_before_sonic"] is None:
        jump_before_sonic = None
    else:
        jump_before_sonic = group["jump_before_sonic"]["data"]
    if group["v_θ_sonic_crit"] is None:
        v_θ_sonic_crit = None
    else:
        v_θ_sonic_crit = group["v_θ_sonic_crit"]["data"]
    if group["after_sonic"] is None:
        after_sonic = None
    else:
        after_sonic = group["after_sonic"]["data"]
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
        mcmc_vars=group["mcmc_vars"],
        v_θ_sonic_crit=v_θ_sonic_crit,
        after_sonic=after_sonic,
        use_E_r=None,
    )


@ds_registry.loader("SolutionInput", version=12)
def _input_loader12(group):
    if group["jump_before_sonic"] is None:
        jump_before_sonic = None
    else:
        jump_before_sonic = group["jump_before_sonic"]["data"]
    if group["v_θ_sonic_crit"] is None:
        v_θ_sonic_crit = None
    else:
        v_θ_sonic_crit = group["v_θ_sonic_crit"]["data"]
    if group["after_sonic"] is None:
        after_sonic = None
    else:
        after_sonic = group["after_sonic"]["data"]
    if group["sonic_interp_size"] is None:
        sonic_interp_size = None
    else:
        sonic_interp_size = group["sonic_interp_size"]["data"]
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
        mcmc_vars=group["mcmc_vars"],
        v_θ_sonic_crit=v_θ_sonic_crit,
        after_sonic=after_sonic,
        sonic_interp_size=sonic_interp_size,
        interp_range=group.attrs["interp_range"],
        use_E_r=None,
    )


@ds_registry.loader("SolutionInput", version=13)
def _input_loader13(group):
    if group["jump_before_sonic"] is None:
        jump_before_sonic = None
    else:
        jump_before_sonic = group["jump_before_sonic"]["data"]
    if group["v_θ_sonic_crit"] is None:
        v_θ_sonic_crit = None
    else:
        v_θ_sonic_crit = group["v_θ_sonic_crit"]["data"]
    if group["after_sonic"] is None:
        after_sonic = None
    else:
        after_sonic = group["after_sonic"]["data"]
    if group["sonic_interp_size"] is None:
        sonic_interp_size = None
    else:
        sonic_interp_size = group["sonic_interp_size"]["data"]
    if group["interp_range"] is None:
        interp_range = None
    else:
        interp_range = ensure_fully_loaded(group["interp_range"])
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
        mcmc_vars=group["mcmc_vars"],
        v_θ_sonic_crit=v_θ_sonic_crit,
        after_sonic=after_sonic,
        sonic_interp_size=sonic_interp_size,
        interp_range=interp_range,
        interp_slice=group["interp_slice"],
        use_E_r=None,
    )


@ds_registry.loader("SolutionInput", version=14)
def _input_loader14(group):
    if group["jump_before_sonic"] is None:
        jump_before_sonic = None
    else:
        jump_before_sonic = group["jump_before_sonic"]["data"]
    if group["v_θ_sonic_crit"] is None:
        v_θ_sonic_crit = None
    else:
        v_θ_sonic_crit = group["v_θ_sonic_crit"]["data"]
    if group["after_sonic"] is None:
        after_sonic = None
    else:
        after_sonic = group["after_sonic"]["data"]
    if group["sonic_interp_size"] is None:
        sonic_interp_size = None
    else:
        sonic_interp_size = group["sonic_interp_size"]["data"]
    if group["interp_range"] is None:
        interp_range = None
    else:
        interp_range = ensure_fully_loaded(group["interp_range"])
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
        mcmc_vars=group["mcmc_vars"],
        v_θ_sonic_crit=v_θ_sonic_crit,
        after_sonic=after_sonic,
        sonic_interp_size=sonic_interp_size,
        interp_range=interp_range,
        interp_slice=group["interp_slice"],
        use_E_r=group.attrs["use_E_r"],
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
    use_E_r = group["use_E_r"]

    config_input = group["config_input"]
    if config_input.use_E_r is None:
        config_input.use_E_r = str(use_E_r)

    solutions = group["solutions"]
    for soln in solutions.values():
        if soln.solution_input.use_E_r is None:
            soln.solution_input.use_E_r = use_E_r

    final_solution = group["final_solution"]
    if final_solution is not None and (
        final_solution.solution_input.use_E_r is None
    ):
        final_solution.solution_input.use_E_r = use_E_r

    return Run(
        config_input=config_input,
        config_filename=group["config_filename"],
        time=group["time"],
        final_solution=final_solution,
        solutions=solutions,
        disc_solver_version=group["disc_solver_version"],
        float_type=group["float_type"],
        sonic_method=group["sonic_method"],
        use_E_r=use_E_r,
    )


@ds_registry.loader("Run", version=6)
def _run_loader6(group):
    use_E_r = group["use_E_r"]

    config_input = group["config_input"]
    if config_input.use_E_r is None:
        config_input.use_E_r = str(use_E_r)

    solutions = group["solutions"]
    for soln in solutions.values():
        if soln.solution_input.use_E_r is None:
            soln.solution_input.use_E_r = use_E_r

    final_solution = group["final_solution"]
    if final_solution is not None and (
        final_solution.solution_input.use_E_r is None
    ):
        final_solution.solution_input.use_E_r = use_E_r

    return Run(
        config_input=config_input,
        config_filename=group["config_filename"],
        time=group["time"],
        final_solution=final_solution,
        solutions=solutions,
        disc_solver_version=group["disc_solver_version"],
        float_type=group["float_type"],
        sonic_method=group["sonic_method"],
        use_E_r=use_E_r,
        is_post_shock_only=group.attrs["is_post_shock_only"],
        based_on_solution_filename=group["based_on_solution_filename"],
        based_on_solution_solution_name=group[
            "based_on_solution_solution_name"
        ],
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


@ds_registry.loader("MCMCVars", version=1)
def _mcmc_vars_loader(group):
    return MCMCVars(
        with_v_r=group["with_v_r"],
        with_v_a=group["with_v_a"],
        with_v_k=group["with_v_k"],
    )


@ds_registry.loader("slice", version=1)
def _slice_loader(group):
    return slice(
        ensure_fully_loaded(group["start"]),
        ensure_fully_loaded(group["stop"]),
        ensure_fully_loaded(group["step"]),
    )

# pylint: enable=missing-docstring
