# -*- coding: utf-8 -*-
"""
Compute and plot field-lines/trajectories.
"""
from numpy import (
    array, sin, cos, arctan2, degrees, tan,
)
from ..critical_points import get_all_sonic_points
from ..utils import ODEIndex, get_closest_value_sorted

from .utils import (
    single_solution_plotter, analyse_main_wrapper, analysis_func_wrapper,
    common_output_plot_options, common_plot_appearence_options,
    get_common_plot_appearence_args, get_common_plot_output_args,
    plot_output_wrapper, DEFAULT_MPL_STYLE,
)

DEFAULT_V_START_POSITION = (2, .1)
DEFAULT_B_START_POSITION = (1, 0)
DEFAULT_V_MAX_STEPS = 100000
DEFAULT_B_MAX_STEPS = 1000000
DEFAULT_V_STEP_SCALING = 1e-6
DEFAULT_B_STEP_SCALING = 1e-3


def compute_values(*, x, y, angles, r_vals, θ_vals, radial_power):
    """
    Get the x,y-axes values at `(x, y)` from `r_vals` and `θ_vals` computed at
    `angles` scaled by `radial_power` i.e. convert from spherical coordinate
    system to Cartesian coordinate system.
    """
    r_pos = x**2 + y**2
    θ_pos = arctan2(y, x)
    rad_coef = r_pos ** radial_power
    r_val = rad_coef * get_closest_value_sorted(xs=angles, ys=r_vals, x=θ_pos)
    θ_val = rad_coef * get_closest_value_sorted(xs=angles, ys=θ_vals, x=θ_pos)

    R_magnitude = r_val ** 2 + θ_val ** 2
    Θ_magnitude = arctan2(θ_val, r_val) + θ_pos

    x_val = R_magnitude * cos(Θ_magnitude)
    y_val = R_magnitude * sin(Θ_magnitude)

    return array([x_val, y_val])


def follow_trajectory(
    *, x_start, y_start, angles, r_vals, θ_vals, radial_power, max_steps=1000,
    step_scaling=1e-3
):
    """
    Compute the positions and values from `(x_start, y_start)` along the
    trajectory using `angles`, `r_vals`, `θ_vals` and `radial_power` to
    determine the values.

    Each step is scaled by `step_scaling` and no more than `max_steps` steps
    are taken.
    """
    values = []
    positions = []
    pos = array([x_start, y_start])
    for step in range(max_steps):
        try:
            vals = compute_values(
                x=pos[0], y=pos[1], angles=angles, r_vals=r_vals,
                θ_vals=θ_vals, radial_power=radial_power,
            )
        except IndexError:
            print("Errored at step with current scaling, dropping", step, pos)
            old_scaling = step_scaling
            step_scaling = old_scaling * 0.01
            print(
                "Old scaling was", old_scaling, "new scaling is", step_scaling
            )
            prev_vals = values[-1]
            prev_pos = positions[-1]
            pos = prev_pos + step_scaling * prev_vals
            # We've run out of solution to show
            try:
                vals = compute_values(
                    x=pos[0], y=pos[1], angles=angles, r_vals=r_vals,
                    θ_vals=θ_vals, radial_power=radial_power,
                )
            except IndexError:
                print("Errored at step with new scaling, stopping", step, pos)
                break
            values.append(vals)
            positions.append(pos)
        else:
            values.append(vals)
            positions.append(pos)
            # Add in scaling by r^2 to allow for bigger steps
            pos = pos + step_scaling * vals
    return array(values), array(positions)


def plot_parser(parser):
    """
    Add arguments for plot command to parser
    """
    common_plot_appearence_options(parser)
    common_output_plot_options(parser)
    parser.add_argument("--v-start-position", nargs=2)
    parser.add_argument("--v-max-steps", type=int, default=DEFAULT_V_MAX_STEPS)
    parser.add_argument(
        "--v-step-scaling", type=float, default=DEFAULT_V_STEP_SCALING,
    )
    parser.add_argument("--B-start-position", nargs=2)
    parser.add_argument("--B-max-steps", type=int, default=DEFAULT_B_MAX_STEPS)
    parser.add_argument(
        "--B-step-scaling", type=float, default=DEFAULT_B_STEP_SCALING,
    )
    return parser


def get_plot_args(args):
    """
    Parse plot args
    """
    v_start_position = args.get("v_start_position")
    if v_start_position is None:
        v_start_position = DEFAULT_V_START_POSITION
    else:
        v_start_position = (
            float(v_start_position[0]),
            float(v_start_position[1]),
        )
    B_start_position = args.get("B_start_position")
    if B_start_position is None:
        B_start_position = DEFAULT_B_START_POSITION
    else:
        B_start_position = (
            float(B_start_position[0]),
            float(B_start_position[1]),
        )
    v_max_steps = args.get("v_max_steps", DEFAULT_V_MAX_STEPS)
    v_step_scaling = args.get("v_step_scaling", DEFAULT_V_STEP_SCALING)
    B_max_steps = args.get("B_max_steps", DEFAULT_B_MAX_STEPS)
    B_step_scaling = args.get("B_step_scaling", DEFAULT_B_STEP_SCALING)

    return {
        "v_start_position": v_start_position,
        "v_max_steps": v_max_steps,
        "v_step_scaling": v_step_scaling,
        "B_start_position": B_start_position,
        "B_max_steps": B_max_steps,
        "B_step_scaling": B_step_scaling,
    }


@analyse_main_wrapper(
    "Main plotter for DiscSolver",
    plot_parser,
    cmd_parser_splitters={
        "common_plot_output_args": get_common_plot_output_args,
        "common_plot_appearence_args": get_common_plot_appearence_args,
        "plot_args": get_plot_args,
    }
)
def plot_main(
    soln, *, soln_range, common_plot_output_args, common_plot_appearence_args,
    plot_args
):
    """
    Entry point for ds-plot
    """
    return plot(
        soln, soln_range=soln_range, **common_plot_appearence_args,
        **common_plot_output_args, **plot_args
    )


@analysis_func_wrapper
def plot(
    soln, *, soln_range=None, plot_filename=None, show=False,
    figargs=None, title=None, close=True, filename,
    mpl_style=DEFAULT_MPL_STYLE, with_version=True,
    v_start_position=DEFAULT_V_START_POSITION,
    B_start_position=DEFAULT_B_START_POSITION, v_max_steps=DEFAULT_V_MAX_STEPS,
    B_max_steps=DEFAULT_B_MAX_STEPS, v_step_scaling=DEFAULT_V_STEP_SCALING,
    B_step_scaling=DEFAULT_B_STEP_SCALING,
):
    """
    Plot solution to file
    """
    # pylint: disable=too-many-function-args,unexpected-keyword-arg
    fig = generate_plot(
        soln, soln_range, figargs=figargs, title=title,
        filename=filename, mpl_style=mpl_style, with_version=with_version,
        v_start_position=v_start_position, v_max_steps=v_max_steps,
        v_step_scaling=v_step_scaling,
        B_start_position=B_start_position, B_max_steps=B_max_steps,
        B_step_scaling=B_step_scaling,
    )

    return plot_output_wrapper(
        fig, file=plot_filename, show=show, close=close
    )


@single_solution_plotter
def generate_plot(
    fig, soln, *,
    v_start_position=DEFAULT_V_START_POSITION,
    B_start_position=DEFAULT_B_START_POSITION,
    v_max_steps=DEFAULT_V_MAX_STEPS, B_max_steps=DEFAULT_B_MAX_STEPS,
    v_step_scaling=DEFAULT_V_STEP_SCALING,
    B_step_scaling=DEFAULT_B_STEP_SCALING,
    use_E_r=False, with_slow=False, with_sonic=True, with_alfven=False,
    with_fast=False, with_45_line=False, with_max_soln_angle=False,
    with_max_v_angle=True,
):
    """
    Generate plot, with enough freedom to be able to format fig
    """
    # pylint: disable=unused-argument
    solution = soln.solution
    angles = soln.angles
    soln_input = soln.solution_input

    slow_angle, sonic_angle, alfven_angle, fast_angle = get_all_sonic_points(
        soln
    )

    ax = fig.subplots()
    v_values, v_positions = follow_trajectory(
        x_start=float(v_start_position[0]), y_start=float(v_start_position[1]),
        angles=angles,
        r_vals=solution[:, ODEIndex.v_r],
        θ_vals=solution[:, ODEIndex.v_θ],
        radial_power=-1/2, max_steps=v_max_steps,
        step_scaling=v_step_scaling,
    )
    B_values, B_positions = follow_trajectory(
        x_start=float(B_start_position[0]), y_start=float(B_start_position[1]),
        angles=angles,
        r_vals=solution[:, ODEIndex.B_r],
        θ_vals=solution[:, ODEIndex.B_θ],
        radial_power=(soln_input.γ - (5/4)), max_steps=B_max_steps,
        step_scaling=B_step_scaling,
    )
    print("v", len(v_values), len(v_positions))
    print("B", len(B_values), len(B_positions))

    if with_slow and slow_angle is not None:
        ax.axline(
            (0, 0), slope=tan(slow_angle),
            label=f"Slow point angle ({degrees(slow_angle)}°)",
            color="C2",
        )

    if with_sonic and sonic_angle is not None:
        ax.axline(
            (0, 0), slope=tan(sonic_angle),
            label=f"Sonic point angle ({degrees(sonic_angle)}°)",
            color="C3",
        )

    if with_alfven and alfven_angle is not None:
        ax.axline(
            (0, 0), slope=tan(alfven_angle),
            label=f"alfven point angle ({degrees(alfven_angle)}°)",
            color="C4",
        )

    if with_fast and fast_angle is not None:
        ax.axline(
            (0, 0), slope=tan(fast_angle),
            label=f"fast point angle ({degrees(fast_angle)}°)",
            color="C5",
        )

    if with_45_line:
        ax.axline(
            (0, 0), slope=1,
            label="45°",
            color="C6",
        )

    if with_max_soln_angle:
        max_soln_angle = max(angles)
        ax.axline(
            (0, 0), slope=tan(max_soln_angle),
            label=f"Max angle in solution {degrees(max_soln_angle)}°",
            color="C7",
        )
    if with_max_v_angle:
        max_v_angle = max(arctan2(v_positions[:, 1], v_positions[:, 0]))
        ax.axline(
            (0, 0), slope=tan(max_v_angle),
            label=f"Max angle in v streamline {degrees(max_v_angle)}°",
            color="C8",
        )

    ax.plot(
        B_positions[:, 0], B_positions[:, 1], label="$B$ field line",
        color="C1",
    )
    ax.plot(
        v_positions[:, 0], v_positions[:, 1], label="$v$ streamline",
        color="C0",
    )

    ax.set_xlabel("Cylindrical radius (distance from origin, arb. units)")
    ax.set_ylabel("Cylindrical height (distance from origin, arb. units)")
    ax.legend(loc="upper left")
    # Don't show anything below the midplane
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.set_aspect('equal', adjustable='datalim')

    return fig
