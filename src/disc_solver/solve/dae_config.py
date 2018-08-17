# -*- coding: utf-8 -*-
"""
Define input and environment for dae system
"""

import logbook

import numpy as np
from numpy import sqrt

from ..file_format import DAEInitialConditions
from ..utils import ODEIndex

log = logbook.Logger(__name__)


def define_dae_conditions(inp, initial_conditions):
    """
    Compute initial conditions for dae
    """
    init_con = initial_conditions.init_con
    γ = initial_conditions.γ
    a_0 = initial_conditions.a_0

    v_r = init_con[ODEIndex.v_r]
    v_φ = init_con[ODEIndex.v_φ]
    ρ = init_con[ODEIndex.ρ]
    η_O = init_con[ODEIndex.η_O]
    η_A = init_con[ODEIndex.η_A]
    η_H = init_con[ODEIndex.η_H]

    deriv_ρ = 0
    deriv_B_θ = 0
    deriv_v_φ = 0
    deriv_v_r = 0
    deriv_B_φ_prime = 0

    deriv_v_θ = - 2 * γ * v_r
    deriv_B_φ = v_r * v_φ / (2 * a_0)
    deriv_B_r = γ - 1/4 + (deriv_B_φ * η_H - v_r) / (η_O + η_A)

    if inp.η_derivs:
        η_O_scale = η_O / sqrt(ρ)
        η_A_scale = η_A / sqrt(ρ)
        η_H_scale = η_H / sqrt(ρ)
    else:
        η_O_scale = 0
        η_A_scale = 0
        η_H_scale = 0

    deriv_ρ_scale = deriv_ρ / sqrt(ρ) / 2
    deriv_η_O = deriv_ρ_scale * η_O_scale
    deriv_η_A = deriv_ρ_scale * η_A_scale
    deriv_η_H = deriv_ρ_scale * η_H_scale

    deriv_init_con = np.zeros(11)

    deriv_init_con[ODEIndex.B_r] = deriv_B_r
    deriv_init_con[ODEIndex.B_φ] = deriv_B_φ
    deriv_init_con[ODEIndex.B_θ] = deriv_B_θ
    deriv_init_con[ODEIndex.v_r] = deriv_v_r
    deriv_init_con[ODEIndex.v_φ] = deriv_v_φ
    deriv_init_con[ODEIndex.v_θ] = deriv_v_θ
    deriv_init_con[ODEIndex.ρ] = deriv_ρ
    deriv_init_con[ODEIndex.B_φ_prime] = deriv_B_φ_prime
    deriv_init_con[ODEIndex.η_O] = deriv_η_O
    deriv_init_con[ODEIndex.η_A] = deriv_η_A
    deriv_init_con[ODEIndex.η_H] = deriv_η_H

    return DAEInitialConditions(
        init_con=init_con, deriv_init_con=deriv_init_con,
        norm_kepler_sq=initial_conditions.norm_kepler_sq, a_0=a_0, γ=γ,
        angles=initial_conditions.angles
    )
