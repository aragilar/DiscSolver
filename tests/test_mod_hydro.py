# -*- coding: utf-8 -*-

from types import SimpleNamespace

import pytest
from pytest import approx

import logbook

import numpy as np
from numpy import sqrt, tan

from disc_solver.float_handling import float_type, FLOAT_TYPE
from disc_solver.utils import ODEIndex
from disc_solver.solve.mod_hydro import ode_system
from disc_solver.analyse.j_e_plot import J_func, E_func

ODE_NUMBER = len(ODEIndex)


@pytest.fixture(scope='module')
def initial_conditions():
    values = SimpleNamespace()
    # These are all arbitrary values, this should not affect the result
    values.params = np.array([7 for i in range(ODE_NUMBER)], dtype=FLOAT_TYPE)
    values.β = float_type(5)/float_type(4)
    values.norm_kepler_sq = float_type(1000)
    values.a_0 = float_type(2)
    values.γ = 0

    values.params[ODEIndex.v_θ] = 0

    # This is slightly off the plane, this should mean we don't get
    # cancellation
    values.angle = float_type(0.01)
    return values


@pytest.fixture(scope='module')
def rhs_func(initial_conditions):
    with logbook.NullHandler().applicationbound():
        rhs, internal_data = ode_system(
            a_0=initial_conditions.a_0,
            norm_kepler_sq=initial_conditions.norm_kepler_sq,
            init_con=initial_conditions.params,
        )
    return rhs


@pytest.fixture(scope='module')
def derivs(initial_conditions, rhs_func):
    derivs = np.zeros(ODE_NUMBER)
    with logbook.StderrHandler().applicationbound():
        err = rhs_func(
            initial_conditions.angle,
            initial_conditions.params,
            derivs
        )
    if err != 0:
        raise RuntimeError("rhs failed")
    return derivs


@pytest.fixture(scope='module')
def solution(initial_conditions, derivs):
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

    B_mag = sqrt(values.B_r**2 + values.B_φ**2 + values.B_θ**2)
    values.norm_B_r, values.norm_B_φ, values.norm_B_θ = (
        values.B_r/B_mag, values.B_φ/B_mag, values.B_θ/B_mag
    )
    return values


def test_continuity(initial_conditions, solution, regtest, test_info, test_id):
    eqn = (
        (
            float_type(5)/float_type(2) - 2 * initial_conditions.β
        ) * solution.v_r + solution.deriv.v_θ + (
            solution.v_θ / solution.ρ
        ) * (
            solution.deriv.ρ - solution.ρ * tan(initial_conditions.angle)
        )
    )
    test_info(eqn)
    regtest.identifier = test_id
    print(eqn, file=regtest)
    assert eqn == approx(0)


def test_solenoid(initial_conditions, solution, regtest, test_info, test_id):
    eqn = solution.deriv.B_θ - (
        - 3/4 * solution.B_r + solution.B_θ * tan(initial_conditions.angle)
    )
    test_info(eqn)
    regtest.identifier = test_id
    print(eqn, file=regtest)
    assert eqn == approx(0)


@pytest.mark.xfail
def test_radial_momentum(
    initial_conditions, solution, regtest, test_info, test_id
):
    eqn = (
        solution.v_θ * solution.deriv.v_r -
        float_type(1)/float_type(2) * solution.v_r**2 -
        solution.v_θ**2 - solution.v_φ**2 + initial_conditions.norm_kepler_sq -
        2 * initial_conditions.β - initial_conditions.a_0 / solution.ρ * (
            solution.B_θ * solution.deriv.B_r + (initial_conditions.β - 1) * (
                solution.B_θ**2 + solution.B_φ**2
            )
        )
    )
    test_info(eqn)
    regtest.identifier = test_id
    print(eqn, file=regtest)
    assert eqn == approx(0)


@pytest.mark.xfail
def test_azimuthal_mometum(
    initial_conditions, solution, regtest, test_info, test_id
):
    eqn = (
        solution.v_θ * solution.deriv.v_φ + 1/2 * solution.v_r * solution.v_φ -
        tan(initial_conditions.angle) * solution.v_θ * solution.v_φ -
        initial_conditions.a_0 / solution.ρ * (
            solution.B_θ * solution.deriv.B_φ +
            (1 - initial_conditions.β) * solution.B_r * solution.B_φ -
            tan(initial_conditions.angle) * solution.B_θ * solution.B_φ
        )
    )
    test_info(eqn)
    regtest.identifier = test_id
    print(eqn, file=regtest)
    assert eqn == approx(0)


def test_polar_momentum(
    initial_conditions, solution, regtest, test_info, test_id
):
    eqn = (
        solution.v_r * solution.v_θ / 2 + solution.v_θ * solution.deriv.v_θ +
        tan(initial_conditions.angle) * solution.v_φ ** 2 +
        solution.deriv.ρ / solution.ρ + initial_conditions.a_0 / solution.ρ * (
            (initial_conditions.β - 1) * solution.B_θ * solution.B_r +
            solution.B_r * solution.deriv.B_r +
            solution.B_φ * solution.deriv.B_φ -
            solution.B_φ ** 2 * tan(initial_conditions.angle)
        )
    )
    test_info(eqn)
    regtest.identifier = test_id
    print(eqn, file=regtest)
    assert eqn == approx(0)


def test_polar_induction(
    initial_conditions, solution, regtest, test_info, test_id
):
    eqn = (
        solution.v_θ * solution.B_r -
        solution.v_r * solution.B_θ - (
            solution.deriv.B_r - solution.B_θ * (1 - initial_conditions.β)
        ) * (
            solution.η_O + solution.η_A * (1 - solution.norm_B_φ**2)
        ) - solution.deriv.B_φ * (
            solution.η_H * solution.norm_B_θ +
            solution.η_A * solution.norm_B_r * solution.norm_B_φ
        ) - solution.B_φ * (
            solution.η_H * (
                solution.norm_B_r * (1 - initial_conditions.β) -
                solution.norm_B_θ * tan(initial_conditions.angle)
            ) - solution.η_A * solution.norm_B_φ * (
                solution.norm_B_θ * (1 - initial_conditions.β) +
                solution.norm_B_r * tan(initial_conditions.angle)
            )
        )
    )
    test_info(eqn)
    regtest.identifier = test_id
    print(eqn, file=regtest)
    assert eqn == approx(0)


def test_azimuthal_induction_numeric(
    initial_conditions, derivs, rhs_func, solution, regtest, test_info, test_id
):
    step = float_type(1e-4)
    new_params = initial_conditions.params + derivs * step
    new_angle = initial_conditions.angle + step
    new_derivs = np.zeros(ODE_NUMBER)
    rhs_func(new_angle, new_params, new_derivs)
    dderiv_B_φ_hacked = ((new_derivs - derivs) / step)[ODEIndex.B_φ]
    eqn = dderiv_B_φ_hacked - solution.deriv.B_φ_prime
    test_info(eqn)
    regtest.identifier = test_id
    print(eqn, file=regtest)
    assert eqn == approx(0, abs=5e-12)


def test_E_φ(initial_conditions, solution, regtest, test_info, test_id):
    J_r, J_θ, J_φ = J_func(
        γ=initial_conditions.γ, θ=initial_conditions.angle, B_θ=solution.B_θ,
        B_φ=solution.B_φ, deriv_B_φ=solution.deriv.B_φ,
        deriv_B_r=solution.deriv.B_r,
    )

    E_r, E_θ, E_φ = E_func(
        v_r=solution.v_r, v_θ=solution.v_θ, v_φ=solution.v_φ, B_r=solution.B_r,
        B_θ=solution.B_θ, B_φ=solution.B_φ, J_r=J_r, J_θ=J_θ, J_φ=J_φ,
        η_O=solution.η_O, η_A=solution.η_A, η_H=solution.η_H,
    )

    eqn = E_φ
    test_info(eqn)
    regtest.identifier = test_id
    print(eqn, file=regtest)
    assert eqn == approx(0)
