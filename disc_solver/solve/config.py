# -*- coding: utf-8 -*-
"""
Define input and environment for ode system
"""

import configparser
from math import pi, sqrt
from types import SimpleNamespace

import logbook

import numpy as np

from ..constants import G, AU, M_SUN, KM
from ..utils import float_with_frac

log = logbook.Logger(__name__)

DEFAULT_INPUT = dict(
    start=90,
    stop=85,
    taylor_stop_angle=89.99,

    # pick a radii, 1AU makes it easy to calculate
    radius=1,
    central_mass=1,

    β=5/4,

    v_rin_on_v_k=0.1,
    scale_height_vs_radius=0.01,

    # B_θ is the equipartition field
    B_θ=18,  # G

    # from wardle 2007 for 1 AU
    # assume B ~ 1 G
    η_O=5e15,  # cm^2/s
    η_H=5e16,  # cm^2/s
    η_A=1e14,  # cm^2/s

    c_s=0.99,  # actually in cm/s for conversions
    # ρ computed from Wardle 2007
    ρ=1.5e-9,  # g/cm^3

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
    cons = SimpleNamespace()
    keplerian_velocity = sqrt(G * inp.central_mass / inp.radius)  # cm/s

    cons.v_norm = inp.c_s
    cons.B_norm = inp.B_θ
    cons.diff_norm = cons.v_norm * inp.radius * inp.scale_height_vs_radius
    cons.ρ_norm = cons.B_norm**2 / (4 * pi * cons.v_norm**2)

    cons.norm_kepler_sq = keplerian_velocity**2 / cons.v_norm**2
    cons.c_s = inp.c_s / cons.v_norm
    cons.η_O = inp.η_O / cons.diff_norm
    cons.η_A = inp.η_A / cons.diff_norm
    cons.η_H = inp.η_H / cons.diff_norm

    v_r = - inp.v_rin_on_v_k * keplerian_velocity
    if v_r > 0:
        log.error("v_r > 0, v_r = {}".format(v_r))
        exit(1)

    v_θ = 0  # symmetry across disc
    B_r = 0  # symmetry across disc
    B_φ = 0  # symmetry across disc

    # norm for use in v_φ
    v_r_normed = v_r / cons.v_norm
    B_θ_normed = inp.B_θ / cons.B_norm
    ρ_normed = inp.ρ / cons.ρ_norm

    # solution for A * v_φ**2 + B * v_φ + C = 0
    A_v_φ = 1
    B_v_φ = - (v_r_normed * cons.η_H) / (2 * (cons.η_O + cons.η_A))
    C_v_φ = (
        v_r_normed**2 / 2 + 2 * inp.β * cons.c_s**2 -
        cons.norm_kepler_sq + B_θ_normed**2 * (
            2 * (inp.β - 1) - v_r_normed / (cons.η_O + cons.η_A)
        ) / (4 * pi * ρ_normed)
    )
    log.debug("A_v_φ: {}".format(A_v_φ))
    log.debug("B_v_φ: {}".format(B_v_φ))
    log.debug("C_v_φ: {}".format(C_v_φ))

    v_φ_normed = - 1/2 * (B_v_φ - sqrt(B_v_φ**2 - 4 * A_v_φ * C_v_φ))

    B_φ_prime_normed = (
        v_φ_normed * v_r_normed * 4 * pi * ρ_normed
    ) / (2 * B_θ_normed)

    log.info("v_φ: {}".format(v_φ_normed * cons.v_norm))
    log.info("B_φ_prime: {}".format(B_φ_prime_normed * cons.B_norm))

    init_con = np.zeros(8)

    init_con[0] = B_r
    init_con[1] = B_φ
    init_con[2] = B_θ_normed
    init_con[3] = v_r_normed
    init_con[4] = v_φ_normed
    init_con[5] = v_θ
    init_con[6] = ρ_normed
    init_con[7] = B_φ_prime_normed

    cons.init_con = init_con

    cons.angles = np.linspace(inp.start, inp.stop, inp.num_angles) / 180 * pi

    return cons


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
    inp = SimpleNamespace()
    inp.label = section_name
    inp.start = float(section.get("start"))
    inp.stop = float(section.get("stop"))
    inp.taylor_stop_angle = float(section.get("taylor_stop_angle"))
    inp.radius = float(section.get("radius")) * AU
    inp.scale_height_vs_radius = float(section.get("scale_height_vs_radius"))
    inp.central_mass = float(section.get("central_mass")) * M_SUN
    inp.β = float_with_frac(section.get("β"))
    inp.v_rin_on_v_k = float(section.get("v_rin_on_v_k"))
    inp.B_θ = float(section.get("B_θ"))
    inp.η_O = float(section.get("η_O"))
    inp.η_H = float(section.get("η_H"))
    inp.η_A = float(section.get("η_A"))
    inp.c_s = float(section.get("c_s")) * KM
    inp.ρ = float(section.get("ρ"))
    inp.max_steps = int(section.get("max_steps"))
    inp.num_angles = int(section.get("num_angles"))
    return inp
