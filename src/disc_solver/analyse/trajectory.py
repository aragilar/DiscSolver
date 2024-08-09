from numpy import array, sin, cos, arctan2, searchsorted, degrees, sqrt, argmax

from ..utils import (
    mhd_wave_speeds, MHD_Wave_Index, ODEIndex, MAGNETIC_INDEXES,
)

from .utils import (
    single_solution_plotter, analyse_main_wrapper, analysis_func_wrapper,
    common_plotting_options, get_common_plot_args, plot_output_wrapper,
    DEFAULT_MPL_STYLE, get_common_arguments, PlotOrdering,
)


def get_closest_value(*, xs, ys, x):
    return ys[searchsorted(xs, x)]


def compute_values(*, x, y, angles, r_vals, θ_vals, radial_power):
    r_pos = x**2 + y**2
    θ_pos = arctan2(y, x)
    rad_coef = r_pos ** radial_power
    #print(r_pos, degrees(θ_pos), rad_coef)
    r_val = rad_coef * get_closest_value(xs=angles, ys=r_vals, x=θ_pos)
    θ_val = rad_coef * get_closest_value(xs=angles, ys=θ_vals, x=θ_pos)

    R_magnitude = r_val ** 2 + θ_val ** 2
    Θ_magnitude = arctan2(θ_val, r_val) + θ_pos

    x_val = R_magnitude * cos(Θ_magnitude)
    y_val = R_magnitude * sin(Θ_magnitude)

    return array([x_val, y_val])


def follow_trajectory(
    *, x_start, y_start, angles, r_vals, θ_vals, radial_power, max_steps=1000,
    step_scaling=1e-3
):
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
            print("Errored at step", step, pos)
            # We've run out of solution to show
            break
        #print("out vals", vals, pos)
        values.append(vals)
        positions.append(pos)
        pos = pos + step_scaling * vals
    return array(values), array(positions)


def plot_parser(parser):
    """
    Add arguments for plot command to parser
    """
    common_plotting_options(parser)
    parser.add_argument("--with-slow", action='store_true', default=False)
    parser.add_argument("--with-alfven", action='store_true', default=False)
    parser.add_argument("--with-fast", action='store_true', default=False)
    parser.add_argument("--with-sonic", action='store_true', default=False)
    return parser


def get_plot_args(args):
    """
    Parse plot args
    """
    return {
        "with_slow": args.get("with_slow", False),
        "with_alfven": args.get("with_alfven", False),
        "with_fast": args.get("with_fast", False),
        "with_sonic": args.get("with_sonic", False),
    }


@analyse_main_wrapper(
    "Main plotter for DiscSolver",
    plot_parser,
    cmd_parser_splitters={
        "common_plot_args": get_common_plot_args,
        "plot_args": get_plot_args,
    }
)
def plot_main(soln, *, soln_range, common_plot_args, plot_args):
    """
    Entry point for ds-plot
    """
    return plot(soln, soln_range=soln_range, **common_plot_args, **plot_args)


@analysis_func_wrapper
def plot(
    soln, *, soln_range=None, plot_filename=None, show=False, linestyle='-',
    with_slow=False, with_alfven=False, with_fast=False, with_sonic=False,
    start=0, stop=90, figargs=None, title=None, close=True, filename,
    mpl_style=DEFAULT_MPL_STYLE, with_version=True,
):
    """
    Plot solution to file
    """
    # pylint: disable=too-many-function-args,unexpected-keyword-arg
    fig = generate_plot(
        soln, soln_range, linestyle=linestyle, with_slow=with_slow,
        with_alfven=with_alfven, with_fast=with_fast, with_sonic=with_sonic,
        start=start, stop=stop, figargs=figargs, title=title, filename=filename,
        mpl_style=mpl_style, with_version=with_version,
    )

    return plot_output_wrapper(
        fig, file=plot_filename, show=show, close=close
    )


@single_solution_plotter
def generate_plot(
    fig, soln, *, linestyle='-', with_slow=False, with_alfven=False,
    with_fast=False, with_sonic=False, start=0, stop=90,
    use_E_r=False, portrait=False, hide_roots=False, hide_sonic_angle=False,
    v_start_position=(2, .1), B_start_position=(1,0),
    v_max_steps=100000, B_max_steps=1000000, v_step_scaling=1e-6
):
    """
    Generate plot, with enough freedom to be able to format fig
    """
    solution = soln.solution
    angles = soln.angles
    soln_input = soln.solution_input
    sonic_angle = soln.sonic_point
    sonic_values = soln.sonic_point_values
    roots_angles = soln.t_roots
    roots_values = soln.y_roots

    wave_speeds = sqrt(mhd_wave_speeds(
        solution[:, MAGNETIC_INDEXES], solution[:, ODEIndex.ρ], 1
    ))

    indexes = (start <= degrees(angles)) & (degrees(angles) <= stop)

    ax = fig.subplots()
    v_values, v_positions = follow_trajectory(
        x_start=float(v_start_position[0]), y_start=float(v_start_position[1]),
        angles=angles,
        r_vals=solution[:, ODEIndex.v_r],
        θ_vals=solution[:,ODEIndex.v_θ],
        radial_power=-1/2, max_steps=v_max_steps,
        step_scaling=v_step_scaling,
    )
    B_values, B_positions = follow_trajectory(
        x_start=float(B_start_position[0]), y_start=float(B_start_position[1]),
        angles=angles,
        r_vals=solution[:, ODEIndex.B_r],
        θ_vals=solution[:,ODEIndex.B_θ],
        radial_power=(soln_input.γ - (5/4)), max_steps=B_max_steps,
    )
    print(len(v_values), len(v_positions))
    v_sonic_idx = argmax((v_values[:, 0] ** 2 + v_values[:, 1]**2) >= 1)
    v_sonic_pos = v_positions[v_sonic_idx]

    ax.plot(v_positions[:, 0], v_positions[:, 1], label="$v$ streamline")
    ax.plot(B_positions[:, 0], B_positions[:, 1], label="$B$ field line")
    ax.plot(v_sonic_pos[0], v_sonic_pos[1], label="sonic point", marker='x')

    ax.set_xlabel("Cylindrical radius (distance from origin, arb. units)")
    ax.set_ylabel("Cylindrical height (distance from origin, arb. units)")

    return fig
