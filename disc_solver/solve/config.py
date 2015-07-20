# -*- coding: utf-8 -*-
"""
Define input and environment for ode system
"""

import configparser
from math import pi, sqrt

import logbook

import numpy as np

from ..constants import G, AU, M_SUN
from ..hdf5_wrapper import NEWEST_CLASS
from ..utils import float_with_frac

namespace_container = NEWEST_CLASS.named_tuples
log = logbook.Logger(__name__)

DEFAULT_INPUT = dict(
    start=0,
    stop=5,
    taylor_stop_angle=0.01,

    # pick a radii, 1AU makes it easy to calculate
    radius=1,
    central_mass=1,

    # from BP82
    β=5/4,

    # B_θ is the equipartition field
    B_θ=18,  # G
    # ρ computed from Wardle 2007
    ρ=1.5e-9,  # g/cm^3

    v_rin_on_c_s=1,

    # from wardle 2007 for 1 AU
    # assume B ~ 1 G
    η_O=5e15,  # cm^2/s
    η_H=5e16,  # cm^2/s
    η_A=1e14,  # cm^2/s

    max_steps=10000,
    num_angles=10000,
)


class CaseDependentConfigParser(configparser.ConfigParser):
    # pylint: disable=too-many-ancestors
    """
    configparser.ConfigParser subclass that removes the case transform.
    """
    def optionxform(self, optionstr):
        return optionstr


def strdict(d):
    """ stringify a dictionary"""
    return {str(k): str(v) for k, v in d.items()}


def define_conditions(inp):
    """
    Compute initial conditions based on input
    """
    keplerian_velocity = sqrt(G * inp.central_mass / inp.radius)  # cm/s
    c_s = sqrt(inp.B_θ**2 / (8 * pi * inp.ρ))

    v_r = - inp.v_rin_on_c_s * c_s
    if v_r > 0:
        log.error("v_r > 0, v_r = {}".format(v_r))
        exit(1)

    v_θ = 0  # symmetry across disc
    B_r = 0  # symmetry across disc
    B_φ = 0  # symmetry across disc

    # Define normalisations
    v_norm = c_s
    B_norm = inp.B_θ
    diff_norm = v_norm * inp.radius
    ρ_norm = B_norm**2 / v_norm**2

    # Norm for use in v_φ
    v_r_normed = v_r / v_norm
    B_θ_normed = inp.B_θ / B_norm
    ρ_normed = inp.ρ / ρ_norm

    norm_kepler_sq = keplerian_velocity**2 / v_norm**2
    c_s = c_s / v_norm
    η_O = inp.η_O / diff_norm
    η_A = inp.η_A / diff_norm
    η_H = inp.η_H / diff_norm

    # solution for A * v_φ**2 + B * v_φ + C = 0
    A_v_φ = 1
    B_v_φ = (v_r_normed * η_H) / (2 * (η_O + η_A))
    C_v_φ = (
        v_r_normed**2 / 2 + 2 * inp.β * c_s**2 -
        norm_kepler_sq - B_θ_normed**2 * (
            v_r_normed / (η_O + η_A)
        ) / (4 * pi * ρ_normed)
    )
    log.debug("A_v_φ: {}".format(A_v_φ))
    log.debug("B_v_φ: {}".format(B_v_φ))
    log.debug("C_v_φ: {}".format(C_v_φ))

    v_φ_normed = - 1 / (2 * A_v_φ) * (
        B_v_φ - sqrt(B_v_φ**2 - 4 * A_v_φ * C_v_φ)
    )

    B_φ_prime_normed = (
        v_φ_normed * v_r_normed * 2 * pi * ρ_normed
    ) / B_θ_normed

    log.info("v_φ: {}".format(v_φ_normed * v_norm))
    log.info("B_φ_prime: {}".format(B_φ_prime_normed * B_norm))

    init_con = np.zeros(8)

    init_con[0] = B_r
    init_con[1] = B_φ
    init_con[2] = B_θ_normed
    init_con[3] = v_r_normed
    init_con[4] = v_φ_normed
    init_con[5] = v_θ
    init_con[6] = ρ_normed
    init_con[7] = B_φ_prime_normed

    angles = np.radians(np.linspace(inp.start, inp.stop, inp.num_angles))

    return namespace_container.initial_conditions(
        v_norm=v_norm, B_norm=B_norm, diff_norm=diff_norm, ρ_norm=ρ_norm,
        norm_kepler_sq=norm_kepler_sq, c_s=c_s, η_O=η_O, η_A=η_A, η_H=η_H,
        init_con=init_con, angles=angles,
    )


def get_input(conffile=None):
    """
    Get input values
    """
    config = CaseDependentConfigParser(defaults=strdict(DEFAULT_INPUT))
    if conffile:
        config.read_file(open(conffile))
    if config.sections():
        return [
            parse_config(section, config[section])
            for section in config.sections()
        ]
    return [parse_config("default", config.defaults())]


def parse_config(section_name, section):
    """
    Get the values from the config file for the run
    """
    inp = namespace_container.config_input(
        label=section_name,
        start=float(section.get("start")),
        stop=float(section.get("stop")),
        taylor_stop_angle=float(section.get("taylor_stop_angle")),
        radius=float(section.get("radius")) * AU,
        central_mass=float(section.get("central_mass")) * M_SUN,
        β=float_with_frac(section.get("β")),
        v_rin_on_c_s=float(section.get("v_rin_on_c_s")),
        B_θ=float(section.get("B_θ")),
        η_O=float(section.get("η_O")),
        η_H=float(section.get("η_H")),
        η_A=float(section.get("η_A")),
        ρ=float(section.get("ρ")),
        max_steps=int(section.get("max_steps")),
        num_angles=int(section.get("num_angles")),
    )
    return inp
