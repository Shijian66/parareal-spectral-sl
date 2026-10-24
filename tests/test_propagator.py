"""Tests for the spectral semi-Lagrangian Parareal propagator.

Run with

    pytest -q

The first block checks the building blocks against closed-form values; the
second reproduces the iteration counts reported in the paper (about a minute).
"""

import numpy as np
import pytest

import propagator as sp


@pytest.fixture(scope="module")
def grid():
    return sp.Grid(N=100, L=1.0, c=1.0)


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------
def test_operator_shapes(grid):
    n = grid.N
    assert grid.D1.shape == (n, n)
    assert grid.A.shape == (2 * n, 2 * n)
    np.testing.assert_array_equal(grid.A[:n, n:], np.eye(n))
    np.testing.assert_array_equal(grid.A[n:, :n], grid.D2)


def test_difference_operators_annihilate_constants(grid):
    np.testing.assert_allclose(grid.D1.sum(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(grid.D2.sum(axis=1), 0.0, atol=1e-12)
    # on a periodic grid the symbol of the centred difference is i sin(theta)/dx,
    # which is real in magnitude but not equal to 1: D1 is only second order
    u = np.exp(1j * 2 * np.pi * grid.x / grid.L)
    assert np.allclose(grid.D1 @ u, 1j * np.sin(2 * np.pi / grid.N) / grid.dx * u)


def test_spectral_integral_of_a_constant(grid):
    dT = 0.015
    out = grid.spectral_integral_v(np.ones(grid.N), dT)
    np.testing.assert_allclose(out, 2.0 * grid.c * dT, atol=1e-12)


def test_spectral_integral_matches_the_analytic_symbol(grid):
    """For v = sin(kx) the integral is 2 sin(k c dT) / k times the same mode."""
    for m in (1, 2, 3):
        k = 2.0 * np.pi * m / grid.L
        v = np.sin(k * grid.x)
        exact = 2.0 * np.sin(k * grid.c * 0.015) / k * v
        assert np.allclose(grid.spectral_integral_v(v, 0.015), exact, atol=1e-13)


def test_interpolation_is_exact_on_the_grid(grid):
    values = np.sin(2 * np.pi * grid.x / grid.L)
    integer_positions = np.arange(grid.N, dtype=float)
    np.testing.assert_allclose(grid.interp(integer_positions, values), values,
                               atol=1e-14)
    np.testing.assert_allclose(grid.interp_cubic(integer_positions, values),
                               values, atol=1e-14)


def test_cubic_interpolation_is_more_accurate_than_linear(grid):
    """At the same resolution the four-point stencil is orders of magnitude
    better; the observed ratio is about 10^3 for h = 0.01."""
    values = np.sin(2 * np.pi * grid.x / grid.L)
    positions = np.arange(grid.N, dtype=float) + 0.5
    exact = np.sin(2 * np.pi * positions * grid.dx / grid.L)
    err_linear = np.linalg.norm(grid.interp(positions, values) - exact)
    err_cubic = np.linalg.norm(grid.interp_cubic(positions, values) - exact)
    assert err_cubic < 1e-2 * err_linear
    assert err_cubic < 1e-4


def test_cubic_interpolation_converges_at_fourth_order():
    """Refining the grid by a factor of four must reduce the error by ~4^4."""
    f = lambda x: np.sin(2 * np.pi * x)
    errors = []
    for n in (100, 400):
        g = sp.Grid(N=n, L=1.0, c=1.0)
        positions = np.arange(n, dtype=float) + 0.5
        exact = f(positions * g.dx)
        errors.append(np.max(np.abs(g.interp_cubic(positions, f(g.x)) - exact)))
    ratio = errors[0] / errors[1]
    assert 150.0 < ratio < 350.0, ratio          # 4^4 = 256


def test_integer_shift_reproduces_the_exact_flow(grid):
    """At s = 2 the feet sit on grid points, so a step with v = 0 is exact."""
    dT = 0.02
    u0 = np.sin(2 * np.pi * grid.x / grid.L)
    U0 = np.concatenate([u0, np.zeros(grid.N)])
    out = grid.coarse_improved(U0, dT)
    exact = np.cos(2 * np.pi * grid.c * dT / grid.L) * u0
    assert np.linalg.norm(out[:grid.N] - exact) / np.linalg.norm(exact) < 1e-12


def test_parareal_is_nilpotent_and_monotone(grid):
    """In exact arithmetic the defect vanishes after a finite number of
    iterations, so the error must decrease before it hits the round-off."""
    U0 = grid.initial_condition()
    err = grid.parareal(U0, 0.015, 1e-3, grid.coarse_implicit_euler,
                        max_iter=8, tol=0.0)
    assert np.all(np.diff(err[:5]) < 0.0), err


# ---------------------------------------------------------------------------
# reference iteration counts of the paper (slow: about one minute)
# ---------------------------------------------------------------------------
def _count_to(errors, tol=1e-6):
    for k, err in enumerate(errors, start=1):
        if err < tol:
            return k
    return len(errors)


def test_figure3_iteration_counts(grid):
    """dT = 0.02, s = 2: five iterations for implicit Euler, one for the SL step."""
    U0 = grid.initial_condition()
    err_ie = grid.parareal(U0, 0.02, 1e-3, grid.coarse_implicit_euler, max_iter=8)
    err_sl = grid.parareal(U0, 0.02, 1e-3, grid.coarse_improved, max_iter=8)
    assert _count_to(err_ie) == 5, err_ie
    assert _count_to(err_sl) == 1, err_sl

    floor = grid.parareal(U0, 0.02, 1e-3, grid.coarse_improved,
                          max_iter=3, tol=0.0)
    assert floor[1] < 1e-10, floor          # 5.5e-11 at this shift


def test_figure4_iteration_counts(grid):
    """dT = 0.015, s = 1.5: five, three and one iterations respectively."""
    U0 = grid.initial_condition()
    err_ie = grid.parareal(U0, 0.015, 1e-3, grid.coarse_implicit_euler, max_iter=8)
    err_sl = grid.parareal(U0, 0.015, 1e-3, grid.coarse_improved, max_iter=8)
    err_cu = grid.parareal(
        U0, 0.015, 1e-3,
        lambda U, d: grid.coarse_improved(U, d, use_cubic=True), max_iter=8)
    assert _count_to(err_ie) == 5, err_ie
    assert _count_to(err_sl) == 3, err_sl
    assert _count_to(err_cu) == 1, err_cu

    floor = grid.parareal(U0, 0.015, 1e-3, grid.coarse_improved,
                          max_iter=7, tol=0.0)
    assert floor[5] < 5e-14, floor          # the exact flow, to round-off


def test_variable_speed_iteration_counts():
    tt = sp.TravelTimeGrid(N=100, c0=1.0, amplitude=0.4)
    u0 = (np.sin(2 * np.pi * tt.tau / tt.L)
          + 0.5 * np.sin(6 * np.pi * tt.tau / tt.L))
    U0 = np.concatenate([u0, np.zeros(tt.N)])
    err_sl = tt.parareal(U0, 0.02, tt.coarse_improved, max_iter=6)
    assert _count_to(err_sl) == 3, err_sl
    err_ex = tt.parareal(U0, 0.02, tt.exact_flow, max_iter=3)
    assert _count_to(err_ex) == 1, err_ex
