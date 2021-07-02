# -*- coding: utf-8 -*-
"""
Computing J and E
"""
from numpy import tan as np_tan


def J_r_func(*, θ, B_φ, deriv_B_φ, tan=np_tan):
    """
    Compute J_r
    """
    return B_φ * tan(θ) - deriv_B_φ


def J_θ_func(*, γ, B_φ):
    """
    Compute J_θ
    """
    return - B_φ * (1/4 - γ)


def J_φ_func(*, γ, B_θ, deriv_B_r):
    """
    Compute J_φ
    """
    return deriv_B_r + B_θ * (1/4 - γ)


def E_r_func(
    *, v_θ, v_φ, B_θ, B_φ, J_r, J_θ, J_φ, η_O, η_A, η_H, b_r, b_θ, b_φ,
):
    """
    Compute E_r
    """
    return (
        η_O * J_r + η_H * (
            J_φ * b_θ - J_θ * b_φ
        ) - η_A * (
            J_φ * b_r * b_φ + J_θ * b_r * b_θ - J_r * (1 - b_r ** 2)
        )
    ) - v_φ * B_θ + v_θ * B_φ


def E_θ_func(
    *, v_r, v_φ, B_r, B_φ, J_r, J_θ, J_φ, η_O, η_A, η_H, b_r, b_θ, b_φ,
):
    """
    Compute E_θ
    """
    return (
        η_O * J_θ + η_H * (
            J_r * b_φ - J_φ * b_r
        ) - η_A * (
            J_r * b_r * b_θ + J_φ * b_θ * b_φ - J_θ * (1 - b_θ ** 2)
        )
    ) - v_r * B_φ + v_φ * B_r


def E_φ_func(
    *, v_r, v_θ, B_r, B_θ, J_r, J_θ, J_φ, η_O, η_A, η_H, b_r, b_θ, b_φ,
):
    """
    Compute E_φ
    """
    return (
        η_O * J_φ + η_H * (
            J_θ * b_r - J_r * b_θ
        ) - η_A * (
            J_θ * b_θ * b_φ + J_r * b_r * b_φ - J_φ * (1 - b_φ ** 2)
        )
    ) - v_θ * B_r + v_r * B_θ
