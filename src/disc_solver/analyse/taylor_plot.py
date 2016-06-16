# -*- coding: utf-8 -*-
"""
Check-taylor command for DiscSolver
"""
import numpy as np
from numpy import degrees
import matplotlib.pyplot as plt

from .utils import single_solution_plotter, common_plotting_options


def check_taylor_parser(parser):
    """
    Add arguments for check-taylor command to parser
    """
    common_plotting_options(parser)
    parser.add_argument(
        "--show-values", action="store_true", default=False
    )
    return parser


@single_solution_plotter
def taylor_plot(
    soln, *, figargs=None, show_values=False
):
    """
    Compare derivatives from taylor series to full version
    """
    v_r_normal = soln.internal_data.v_r_normal
    v_φ_normal = soln.internal_data.v_φ_normal
    ρ_normal = soln.internal_data.ρ_normal
    v_r_taylor = soln.internal_data.v_r_taylor
    v_φ_taylor = soln.internal_data.v_φ_taylor
    ρ_taylor = soln.internal_data.ρ_taylor

    deriv_angles = soln.internal_data.angles
    fig, axes = plt.subplots(ncols=3, tight_layout=True, **figargs)
    if show_values:
        axes[0].plot(degrees(deriv_angles), v_r_normal)
        axes[0].plot(degrees(deriv_angles), v_r_taylor)
        axes[1].plot(degrees(deriv_angles), v_φ_normal)
        axes[1].plot(degrees(deriv_angles), v_φ_taylor)
        axes[2].plot(degrees(deriv_angles), ρ_normal)
        axes[2].plot(degrees(deriv_angles), ρ_taylor)
        axes[0].set_yscale("log")
        axes[1].set_yscale("log")
        axes[2].set_yscale("log")
    else:
        axes[0].plot(
            degrees(deriv_angles),
            np.abs(v_r_normal - v_r_taylor), '.'
        )
        axes[1].plot(
            degrees(deriv_angles),
            np.abs(v_φ_normal - v_φ_taylor), '.'
        )
        axes[2].plot(
            degrees(deriv_angles),
            np.abs(ρ_normal - ρ_taylor), '.'
        )
        axes[0].set_yscale("log")
        axes[1].set_yscale("log")
        axes[2].set_yscale("log")
    return fig
