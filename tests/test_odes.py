
# -*- coding: utf-8 -*-

from math import pi, sqrt, tan, isinf
from types import SimpleNamespace
import unittest

import pytest
from pytest import approx

import hypothesis
from hypothesis import assume
import hypothesis.strategies as st
from hypothesis.extra.numpy import arrays

import logbook

import numpy as np

from disc_solver.constants import G
from disc_solver.utils import ODEIndex
from disc_solver.solve.solution import ode_system
from disc_solver.solve.dae_solution import dae_system

ODE_NUMBER = len(ODEIndex)
CONTINUTY_MAX_ERROR = 2.3283064365386963e-10
SOLENOID_MAX_ERROR = 7.2759576141834259e-12

add_initial_conditions = hypothesis.given(
    params=st.tuples(
        st.floats(allow_nan=False, allow_infinity=False),
        st.floats(allow_nan=False, allow_infinity=False),
        st.floats(allow_nan=False, allow_infinity=False),
        st.floats(allow_nan=False, allow_infinity=False),
        st.floats(allow_nan=False, allow_infinity=False),
        st.floats(allow_nan=False, allow_infinity=False),
        st.floats(allow_nan=False, allow_infinity=False),
        st.floats(allow_nan=False, allow_infinity=False),
        st.floats(allow_nan=False, allow_infinity=False),
        st.floats(allow_nan=False, allow_infinity=False),
        st.floats(allow_nan=False, allow_infinity=False),
    ),
    β=st.floats(allow_nan=False, allow_infinity=False),
    norm_kepler_sq=st.floats(allow_nan=False, allow_infinity=False),
    a_0=st.floats(allow_nan=False, allow_infinity=False),
    angle=st.floats(allow_nan=False, allow_infinity=False),
)


def initial_conditions_func(params, β, norm_kepler_sq, a_0, angle):
    values = SimpleNamespace()

    values.params = np.array(params, dtype=float)
    values.β = β
    values.norm_kepler_sq = norm_kepler_sq
    values.a_0 = a_0
    values.angle = angle
    values.γ = 5/4 - values.β

    return values


def validate_initial_conditions(initial_conditions):
    norm_kepler_sq = initial_conditions.norm_kepler_sq
    a_0 = initial_conditions.a_0
    β = initial_conditions.β
    γ = initial_conditions.γ
    angle = initial_conditions.angle
    B_r = initial_conditions.params[ODEIndex.B_r]
    B_φ = initial_conditions.params[ODEIndex.B_φ]
    B_θ = initial_conditions.params[ODEIndex.B_θ]
    v_r = initial_conditions.params[ODEIndex.v_r]
    v_φ = initial_conditions.params[ODEIndex.v_φ]
    v_θ = initial_conditions.params[ODEIndex.v_θ]
    ρ = initial_conditions.params[ODEIndex.ρ]
    B_φ_prime = initial_conditions.params[ODEIndex.B_φ_prime]
    η_O = initial_conditions.params[ODEIndex.η_O]
    η_A = initial_conditions.params[ODEIndex.η_A]
    η_H = initial_conditions.params[ODEIndex.η_H]

    assume(norm_kepler_sq > 0)
    assume(a_0 > 0)

    assume(ρ > 0)
    assume(v_θ > 0)
    assume(η_O > 0)
    assume(η_A >= 0)

    assume(B_r ** 2 + B_φ ** 2 + B_θ ** 2 > 0)

    assume(isinf(1 / ρ) is False)
    assume(isinf(1 / v_θ) is False)

    assume(isinf(β * β) is False)

    assume(isinf(B_r ** 2) is False)
    assume(isinf(B_θ ** 2) is False)
    assume(isinf(B_φ ** 2) is False)
    assume(isinf(v_r ** 2) is False)
    assume(isinf(v_θ ** 2) is False)
    assume(isinf(v_φ ** 2) is False)
    assume(isinf(ρ ** 2) is False)
    assume(isinf(B_φ_prime ** 2) is False)
    assume(isinf(η_O ** 2) is False)
    assume(isinf(η_A ** 2) is False)
    assume(isinf(η_H ** 2) is False)

    for q in initial_conditions.params:
        assume(isinf(q ** 2 / ρ) is False)
        assume(isinf(q ** 2/ v_θ) is False)

def rhs_func_func(initial_conditions):
    with logbook.NullHandler().applicationbound():
        rhs, internal_data = ode_system(
            γ=initial_conditions.γ,
            a_0=initial_conditions.a_0,
            norm_kepler_sq=initial_conditions.norm_kepler_sq,
            init_con=initial_conditions.params,
            with_taylor=False
        )
    return rhs


def derivs_func(rhs_func, initial_conditions):
    derivs = np.zeros(ODE_NUMBER)
    with logbook.NullHandler().applicationbound():
        rhs_func(
            initial_conditions.angle,
            initial_conditions.params,
            derivs
        )
    return derivs


def solution_func(derivs, initial_conditions):
    values = SimpleNamespace(deriv=SimpleNamespace())

    values.B_r = initial_conditions.params[ODEIndex.B_r]
    values.B_φ = initial_conditions.params[ODEIndex.B_φ]
    values.B_θ = initial_conditions.params[ODEIndex.B_θ]
    values.v_r = initial_conditions.params[ODEIndex.v_r]
    values.v_φ = initial_conditions.params[ODEIndex.v_φ]
    values.v_θ = initial_conditions.params[ODEIndex.v_θ]
    values.ρ = initial_conditions.params[ODEIndex.ρ]
    values.η_O = initial_conditions.params[ODEIndex.η_O]
    values.η_A = initial_conditions.params[ODEIndex.η_A]
    values.η_H = initial_conditions.params[ODEIndex.η_H]
    values.deriv.B_r = derivs[ODEIndex.B_r]
    values.deriv.B_φ = derivs[ODEIndex.B_φ]
    values.deriv.B_θ = derivs[ODEIndex.B_θ]
    values.deriv.v_r = derivs[ODEIndex.v_r]
    values.deriv.v_φ = derivs[ODEIndex.v_φ]
    values.deriv.v_θ = derivs[ODEIndex.v_θ]
    values.deriv.ρ = derivs[ODEIndex.ρ]
    values.deriv.B_φ_prime = derivs[ODEIndex.B_φ_prime]
    values.deriv.η_O = derivs[ODEIndex.η_O]
    values.deriv.η_A = derivs[ODEIndex.η_A]
    values.deriv.η_H = derivs[ODEIndex.η_H]

    B_mag = sqrt(values.B_r**2 + values.B_φ**2 + values.B_θ **2)
    values.norm_B_r, values.norm_B_φ, values.norm_B_θ = (
        values.B_r/B_mag, values.B_φ/B_mag, values.B_θ/B_mag)
    return values


def residual_func(derivs, initial_conditions):
    residual = np.zeros(ODE_NUMBER)
    with logbook.NullHandler().applicationbound():
        dae_rhs, internal_data = dae_system(
            γ=initial_conditions.γ,
            a_0=initial_conditions.a_0,
            norm_kepler_sq=initial_conditions.norm_kepler_sq,
            init_con=initial_conditions.params,
        )
        dae_rhs(
            initial_conditions.angle,
            initial_conditions.params,
            derivs,
            residual
        )
    return residual

@add_initial_conditions
def test_continuity(params, β, norm_kepler_sq, a_0, angle, regtest, test_info):
    initial_conditions = initial_conditions_func(
        params, β, norm_kepler_sq, a_0, angle
    )
    validate_initial_conditions(initial_conditions)
    solution = solution_func(
        derivs_func(rhs_func_func(initial_conditions), initial_conditions),
        initial_conditions
    )
    eqn = (
        (5/2 - 2 * initial_conditions.β) * solution.v_r +
        solution.deriv.v_θ + (
            solution.v_θ / solution.ρ
        ) * (
            solution.deriv.ρ - solution.ρ * tan(initial_conditions.angle)
        )
    )
    test_info(eqn)
    print(eqn, file=regtest)
    assert eqn == approx(0, abs=CONTINUTY_MAX_ERROR)


@add_initial_conditions
def test_solenoid(params, β, norm_kepler_sq, a_0, angle, regtest, test_info):
    initial_conditions = initial_conditions_func(
        params, β, norm_kepler_sq, a_0, angle
    )
    validate_initial_conditions(initial_conditions)
    solution = solution_func(
        derivs_func(rhs_func_func(initial_conditions), initial_conditions),
        initial_conditions
    )
    eqn = solution.deriv.B_θ - (
        (initial_conditions.β - 2) * solution.B_r + solution.B_θ * tan(initial_conditions.angle)
    )
    test_info(eqn)
    print(eqn, file=regtest)
    assert eqn == approx(0, abs=SOLENOID_MAX_ERROR)


@add_initial_conditions
def test_radial_momentum(params, β, norm_kepler_sq, a_0, angle, regtest, test_info):
    initial_conditions = initial_conditions_func(
        params, β, norm_kepler_sq, a_0, angle
    )
    validate_initial_conditions(initial_conditions)
    solution = solution_func(
        derivs_func(rhs_func_func(initial_conditions), initial_conditions),
        initial_conditions
    )
    eqn = (solution.v_θ * solution.deriv.v_r - 1/2 * solution.v_r**2 -
        solution.v_θ**2 - solution.v_φ**2 + initial_conditions.norm_kepler_sq -
        2 * initial_conditions.β - initial_conditions.a_0 / solution.ρ * (
            solution.B_θ * solution.deriv.B_r + (initial_conditions.β - 1) * (
                solution.B_θ**2 + solution.B_φ**2
            )
        )
    )
    test_info(eqn)
    print(eqn, file=regtest)
    assert eqn == approx(0)


@add_initial_conditions
def test_azimuthal_mometum(params, β, norm_kepler_sq, a_0, angle, regtest, test_info):
    initial_conditions = initial_conditions_func(
        params, β, norm_kepler_sq, a_0, angle
    )
    validate_initial_conditions(initial_conditions)
    solution = solution_func(
        derivs_func(rhs_func_func(initial_conditions), initial_conditions),
        initial_conditions
    )
    eqn = (solution.v_θ * solution.deriv.v_φ + 1/2 * solution.v_r * solution.v_φ -
        tan(initial_conditions.angle) * solution.v_θ * solution.v_φ - initial_conditions.a_0 / solution.ρ * (
            solution.B_θ * solution.deriv.B_φ +
            (1 - initial_conditions.β) * solution.B_r * solution.B_φ -
            tan(initial_conditions.angle) * solution.B_θ * solution.B_φ
        )
    )
    test_info(eqn)
    print(eqn, file=regtest)
    assert eqn == approx(0)


@add_initial_conditions
def test_polar_momentum(params, β, norm_kepler_sq, a_0, angle, regtest, test_info):
    initial_conditions = initial_conditions_func(
        params, β, norm_kepler_sq, a_0, angle
    )
    validate_initial_conditions(initial_conditions)
    solution = solution_func(
        derivs_func(rhs_func_func(initial_conditions), initial_conditions),
        initial_conditions
    )
    eqn = (solution.v_r * solution.v_θ / 2 + solution.v_θ * solution.deriv.v_θ +
        tan(initial_conditions.angle) * solution.v_φ ** 2 + solution.deriv.ρ / solution.ρ +
        initial_conditions.a_0 / solution.ρ * (
            (initial_conditions.β - 1) * solution.B_θ * solution.B_r +
            solution.B_r * solution.deriv.B_r + solution.B_φ * solution.deriv.B_φ -
            solution.B_φ ** 2 * tan(initial_conditions.angle)
        )
    )
    test_info(eqn)
    print(eqn, file=regtest)
    assert eqn == approx(0)


@add_initial_conditions
def test_polar_induction(params, β, norm_kepler_sq, a_0, angle, regtest, test_info):
    initial_conditions = initial_conditions_func(
        params, β, norm_kepler_sq, a_0, angle
    )
    validate_initial_conditions(initial_conditions)
    solution = solution_func(
        derivs_func(rhs_func_func(initial_conditions), initial_conditions),
        initial_conditions
    )
    eqn = (
        solution.v_θ * solution.B_r -
        solution.v_r * solution.B_θ + (
            solution.B_θ * (1 - initial_conditions.β) - solution.deriv.B_r
        ) * (
            solution.η_O + solution.η_A * (1 - solution.norm_B_φ**2)
        ) + solution.deriv.B_φ * (
            solution.η_H * solution.norm_B_θ -
            solution.η_A * solution.norm_B_r * solution.norm_B_φ
        ) + solution.B_φ * (
            solution.η_H * (
                solution.norm_B_r * (1 - initial_conditions.β) -
                solution.norm_B_θ * tan(initial_conditions.angle)
            ) - solution.η_A * solution.norm_B_φ * (
                solution.norm_B_θ * (1 - initial_conditions.β) -
                solution.norm_B_r * tan(initial_conditions.angle)
            )
        )
    )
    test_info(eqn)
    print(eqn, file=regtest)
    assert eqn == approx(0)


@add_initial_conditions
def test_azimuthal_induction_numeric(params, β, norm_kepler_sq, a_0, angle, regtest, test_info):
    initial_conditions = initial_conditions_func(
        params, β, norm_kepler_sq, a_0, angle
    )
    validate_initial_conditions(initial_conditions)
    rhs_func = rhs_func_func(initial_conditions)
    derivs(rhs_func, initial_conditions)
    step = 1e-4
    new_params = initial_conditions.params + derivs * step
    new_angle = initial_conditions.angle + step
    new_derivs = np.zeros(ODE_NUMBER)
    rhs_func(new_angle, new_params, new_derivs)
    dderiv_B_φ_hacked = ((new_derivs - derivs) / step)[ODEIndex.B_φ]
    eqn = dderiv_B_φ_hacked - solution.deriv.B_φ_prime
    test_info(eqn)
    print(eqn, file=regtest)
    assert eqn == approx(0, abs=2e-12)


@add_initial_conditions
def test_dae_continuity(params, β, norm_kepler_sq, a_0, angle, regtest, test_info):
    initial_conditions = initial_conditions_func(
        params, β, norm_kepler_sq, a_0, angle
    )
    validate_initial_conditions(initial_conditions)
    derivs = derivs_func(rhs_func_func(initial_conditions), initial_conditions)
    solution = solution_func(derivs, initial_conditions)
    residual = residual_func(derivs, initial_conditions)
    res = residual[ODEIndex.ρ]
    test_info(res)
    print(res, file=regtest)
    assert res == approx(0)


@add_initial_conditions
def test_dae_solenoid(params, β, norm_kepler_sq, a_0, angle, regtest, test_info):
    initial_conditions = initial_conditions_func(
        params, β, norm_kepler_sq, a_0, angle
    )
    validate_initial_conditions(initial_conditions)
    derivs = derivs_func(rhs_func_func(initial_conditions), initial_conditions)
    solution = solution_func(derivs, initial_conditions)
    residual = residual_func(derivs, initial_conditions)
    res = residual[ODEIndex.B_θ]
    test_info(res)
    print(res, file=regtest)
    assert res == approx(0)


@add_initial_conditions
def test_dae_radial_momentum(params, β, norm_kepler_sq, a_0, angle, regtest, test_info):
    initial_conditions = initial_conditions_func(
        params, β, norm_kepler_sq, a_0, angle
    )
    validate_initial_conditions(initial_conditions)
    derivs = derivs_func(rhs_func_func(initial_conditions), initial_conditions)
    solution = solution_func(derivs, initial_conditions)
    residual = residual_func(derivs, initial_conditions)
    res = residual[ODEIndex.v_r]
    test_info(res)
    print(res, file=regtest)
    assert res == approx(0)


@add_initial_conditions
def test_dae_azimuthal_mometum(params, β, norm_kepler_sq, a_0, angle, regtest, test_info):
    initial_conditions = initial_conditions_func(
        params, β, norm_kepler_sq, a_0, angle
    )
    validate_initial_conditions(initial_conditions)
    derivs = derivs_func(rhs_func_func(initial_conditions), initial_conditions)
    solution = solution_func(derivs, initial_conditions)
    residual = residual_func(derivs, initial_conditions)
    res = residual[ODEIndex.v_φ]
    test_info(res)
    print(res, file=regtest)
    assert res == approx(0)


@add_initial_conditions
def test_dae_polar_momentum(params, β, norm_kepler_sq, a_0, angle, regtest, test_info):
    initial_conditions = initial_conditions_func(
        params, β, norm_kepler_sq, a_0, angle
    )
    validate_initial_conditions(initial_conditions)
    derivs = derivs_func(rhs_func_func(initial_conditions), initial_conditions)
    solution = solution_func(derivs, initial_conditions)
    residual = residual_func(derivs, initial_conditions)
    res = residual[ODEIndex.v_θ]
    test_info(res)
    print(res, file=regtest)
    assert res == approx(0)


@add_initial_conditions
def test_dae_polar_induction(params, β, norm_kepler_sq, a_0, angle, regtest, test_info):
    initial_conditions = initial_conditions_func(
        params, β, norm_kepler_sq, a_0, angle
    )
    validate_initial_conditions(initial_conditions)
    derivs = derivs_func(rhs_func_func(initial_conditions), initial_conditions)
    solution = solution_func(derivs, initial_conditions)
    residual = residual_func(derivs, initial_conditions)
    res = residual[ODEIndex.B_r]
    test_info(res)
    print(res, file=regtest)
    assert res == approx(0)


@add_initial_conditions
def test_dae_azimuthal_induction(params, β, norm_kepler_sq, a_0, angle, regtest, test_info):
    initial_conditions = initial_conditions_func(
        params, β, norm_kepler_sq, a_0, angle
    )
    validate_initial_conditions(initial_conditions)
    derivs = derivs_func(rhs_func_func(initial_conditions), initial_conditions)
    solution = solution_func(derivs, initial_conditions)
    residual = residual_func(derivs, initial_conditions)
    res = residual[ODEIndex.B_φ_prime]
    test_info(res)
    print(res, file=regtest)
    assert res == approx(0)

# This would be useful to do when I have time
#def test_azimuthal_induction_algebraic(self):
#    eqn = 1 # FAIL
#    test_info(eqn)
#    self.assertAlmostEqual(0,eqn)

