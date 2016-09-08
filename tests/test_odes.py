
# -*- coding: utf-8 -*-

from math import pi, sqrt, tan
import unittest

import logbook

import numpy as np

from disc_solver.constants import G
from disc_solver.solve.solution import ode_system

ODE_NUMBER = 11

class ODE_Test(unittest.TestCase):
    def setUp(self):
        # These are all arbitrary values, this should not affect the result
        params = np.array([7 for i in range(ODE_NUMBER)])
        self.β = 2
        self.norm_kepler_sq = 2
        self.a_0 = 2

        # This is slightly off the plane, this should mean we don't get
        # cancellation
        self.angle = 0.1

        # Here's the action computation
        derivs = np.zeros(ODE_NUMBER)
        with logbook.NullHandler().applicationbound():
            self.rhs, internal_data = ode_system(
                self.β, self.a_0, self.norm_kepler_sq, params, with_taylor=False
            )
            self.rhs(self.angle, params, derivs)

        # Assign useful names to params, derivs
        self.B_r = params[0]
        self.B_φ = params[1]
        self.B_θ = params[2]
        self.v_r = params[3]
        self.v_φ = params[4]
        self.v_θ = params[5]
        self.ρ = params[6]
        self.η_O = params[8]
        self.η_A = params[9]
        self.η_H = params[10]
        self.deriv_B_r = derivs[0]
        self.deriv_B_φ = derivs[1]
        self.deriv_B_θ = derivs[2]
        self.deriv_v_r = derivs[3]
        self.deriv_v_φ = derivs[4]
        self.deriv_v_θ = derivs[5]
        self.deriv_ρ = derivs[6]
        self.dderiv_B_φ = derivs[7]
        self.deriv_η_O = derivs[8]
        self.deriv_η_A = derivs[9]
        self.deriv_η_H = derivs[10]


        B_mag = sqrt(self.B_r**2 + self.B_φ**2 + self.B_θ **2)
        self.norm_B_r, self.norm_B_φ, self.norm_B_θ = (
            self.B_r/B_mag, self.B_φ/B_mag, self.B_θ/B_mag)

        self.params = params
        self.derivs = derivs

    def test_continuity(self):
        eqn = (
            (5/2 - 2 * self.β) * self.v_r +
            self.deriv_v_θ + (
                self.v_θ / self.ρ
            ) * (
                self.deriv_ρ - self.ρ * tan(self.angle)
            )
        )
        print("continuity", eqn)
        self.assertAlmostEqual(0, eqn)

    def test_solenoid(self):
        eqn = self.deriv_B_θ - (
            (self.β - 2) * self.B_r + self.B_θ * tan(self.angle)
        )
        print("solenoid", eqn)
        self.assertAlmostEqual(0, eqn)

    def test_radial_momentum(self):
        eqn = (self.v_θ * self.deriv_v_r - 1/2 * self.v_r**2 -
            self.v_θ**2 - self.v_φ**2 + self.norm_kepler_sq -
            2 * self.β - self.a_0 / self.ρ * (
                self.B_θ * self.deriv_B_r + (self.β - 1) * (
                    self.B_θ**2 + self.B_φ**2
                )
            )
        )
        print("radial momentum", eqn)
        self.assertAlmostEqual(0, eqn)

    def test_azimuthal_mometum(self):
        eqn = (self.v_θ * self.deriv_v_φ + 1/2 * self.v_r * self.v_φ -
            tan(self.angle) * self.v_θ * self.v_φ - self.a_0 / self.ρ * (
                self.B_θ * self.deriv_B_φ +
                (1 - self.β) * self.B_r * self.B_φ -
                tan(self.angle) * self.B_θ * self.B_φ
            )
        )
        print("azimuthal momentum", eqn)
        self.assertAlmostEqual(0, eqn)

    def test_polar_momentum(self):
        eqn = (self.v_r * self.v_θ / 2 + self.v_θ * self.deriv_v_θ +
            tan(self.angle) * self.v_φ ** 2 + self.deriv_ρ / self.ρ +
            self.a_0 / self.ρ * (
                (self.β - 1) * self.B_θ * self.B_r +
                self.B_r * self.deriv_B_r + self.B_φ * self.deriv_B_φ -
                self.B_φ ** 2 * tan(self.angle)
            )
        )
        print("polar momentum", eqn)
        self.assertAlmostEqual(0, eqn)

    def test_polar_induction(self):
        eqn = (
            self.v_θ * self.B_r -
            self.v_r * self.B_θ + (
                self.B_θ * (1 - self.β) - self.deriv_B_r
            ) * (
                self.η_O + self.η_A * (1 - self.norm_B_φ**2)
            ) + self.deriv_B_φ * (
                self.η_H * self.norm_B_θ -
                self.η_A * self.norm_B_r * self.norm_B_φ
            ) + self.B_φ * (
                self.η_H * (
                    self.norm_B_r * (1 - self.β) -
                    self.norm_B_θ * tan(self.angle)
                ) - self.η_A * self.norm_B_φ * (
                    self.norm_B_θ * (1 - self.β) -
                    self.norm_B_r * tan(self.angle)
                )
            )
        )
        print("polar induction", eqn)
        self.assertAlmostEqual(0, eqn)

    def test_azimuthal_induction_numeric(self):
        step = 1e-4
        new_params = self.params + self.derivs * step
        new_angle = self.angle + step
        new_derivs = np.zeros(ODE_NUMBER)
        self.rhs(new_angle, new_params, new_derivs)
        dderiv_B_φ_hacked = ((new_derivs - self.derivs) / step)[1]
        eqn = dderiv_B_φ_hacked - self.dderiv_B_φ
        print("azimuthal induction (num)", eqn)
        self.assertAlmostEqual(0, eqn)

    # This would be useful to do when I have time
    #def test_azimuthal_induction_algebraic(self):
    #    eqn = 1 # FAIL
    #    print(eqn)
    #    self.assertAlmostEqual(0,eqn)

