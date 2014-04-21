
# -*- coding: utf-8 -*-

from math import pi, sqrt
import unittest

from disc_solver import ode_system, cot, G

ODE_NUMBER = 9

class ODE_Test(unittest.TestCase):
    def setUp(self):
        # These are all arbitrary values, this should not affect the result
        params = [1 for i in range(ODE_NUMBER)]
        self.B_power = 2
        self.sound_speed = 2
        self.central_mass = 2
        self.ohm_diff = 2
        self.abi_diff = 2
        self.hall_diff = 2

        # This is slightly off the plane, this should mean we don't get
        # cancellation
        self.angle = pi/2 + 0.1

        # Here's the action computation
        derivs = [0 for i in range(ODE_NUMBER)]
        rhs = ode_system(
                self.B_power, self.sound_speed, self.central_mass,
                self.ohm_diff, self.abi_diff, self.hall_diff
        )
        rhs(self.angle, params, derivs)

        # Assign useful names to params, derivs
        self.B_r = params[0]
        self.B_phi = params[1]
        self.B_theta = params[2]
        self.v_r = params[3]
        self.v_phi = params[4]
        self.v_theta = params[5]
        self.rho = params[6]
        self.deriv_B_r = derivs[0]
        self.deriv_B_phi = derivs[1]
        self.deriv_B_theta = derivs[2]
        self.deriv_v_r = derivs[3]
        self.deriv_v_phi = derivs[4]
        self.deriv_v_theta = derivs[5]
        self.deriv_rho = derivs[6]
        self.dderiv_B_phi = derivs[7]
        self.dderiv_B_r = derivs[8]


        B_mag = sqrt(self.B_r**2 + self.B_phi**2 + self.B_theta **2)
        self.norm_B_r, self.norm_B_phi, self.norm_B_theta = (
            self.B_r/B_mag, self.B_phi/B_mag, self.B_theta/B_mag)

    def test_continuity(self):
        self.assertAlmostEqual(0,
            (5/2 - 2 * self.B_power) * self.v_r + self.deriv_v_theta +
            (self.v_theta / self.rho) * (self.deriv_rho + self.rho *
                cot(self.angle)
        ))

    def test_solenoid(self):
        self.assertAlmostEqual(self.deriv_B_theta,
            (self.B_power - 2) * self.B_r - self.B_theta * cot(self.angle)
        )

    def test_radial_momentum(self):
        self.assertAlmostEqual(0,
            self.v_theta * self.deriv_v_r - 1/2 * self.v_r**2 -
            self.v_theta**2 - self.v_phi**2 - G * self.central_mass -
            self.sound_speed**2 * 2 * self.B_power - 1/(4 * pi * self.rho) * (
                self.B_theta * self.deriv_B_r + (self.B_power - 1) * (
                    self.B_theta**2 + self.B_phi**2
                )
            )
        )

    def test_azimuthal_mometum(self):
        self.assertAlmostEqual(0,
            self.v_theta * self.deriv_v_phi + 1/2 * self.v_r * self.v_phi -
            cot(self.angle) * self.v_theta * self.v_phi - 
            1 / (4 * pi * self.rho) * (self.B_theta * self.deriv_B_phi +
                (1 - self.B_power) * self.B_r * self.B_phi - 
                cot(self.angle) * self.B_theta * self.B_phi
        ))

    def test_polar_momentum(self):
        self.assertAlmostEqual(0,
            self.v_r * self.v_theta / 2 + self.v_theta * self.deriv_v_theta -
            cot(self.angle) ** self.v_phi ** 2 + 
            self.sound_speed **2 * self.deriv_rho / self.rho +
            1 / (4 * pi * self.rho) * (
                (self.B_power - 1) * self.B_theta * self.B_r +
                self.B_r * self.deriv_B_r + self.B_phi * self.deriv_B_phi +
                self.B_phi ** 2 * cot(self.angle)
        ))

    def test_polar_induction(self):
        self.assertAlmostEqual(0,
            self.v_r * self.B_theta - self.v_theta * self.B_r +
            (self.B_theta * (1 - self.B_power) + self.deriv_B_r) *
            (self.ohm_diff + self.abi_diff * (1 - self.norm_B_phi**2)) +
            self.deriv_B_phi * (
                self.hall_diff * self.norm_B_theta -
                self.abi_diff * self.norm_B_r * self.norm_B_phi
            ) + self.B_phi * (
                self.hall_diff * (
                    self.norm_B_theta * cot(self.angle) -
                    self.norm_B_r * (1 - self.B_power)
                ) + self.abi_diff * self.norm_B_phi * (
                    self.norm_B_theta * (1 - self.B_power) -
                    self.norm_B_r * cot(self.angle)
                )
            )
        )

